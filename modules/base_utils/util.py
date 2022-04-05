import sys
import os
import numpy as np
import scipy
import scipy.sparse.linalg
import torch
import tqdm
from functools import partial
from torch import optim
from torch.utils.data import DataLoader, Dataset
from typing import Collection, Dict, Union
from base_utils.model.model import SequentialImageNetwork,\
                                   SequentialImageNetworkMod
import torch.backends.cudnn as cudnn
import toml
from collections import OrderedDict

from base_utils.datasets import make_dataloader
from ranger_opt.ranger import ranger2020 as ranger


if torch.cuda.is_available():
    cudnn.benchmark = True

default_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

DEFAULT_SGD_BATCH_SIZE = 128
DEFAULT_SGD_EPOCHS = 200
DEFAULT_SGD_KWARGS = {
    'lr': 0.1,
    'momentum': 0.9,
    'nesterov': True,
    'weight_decay': 2e-4
}
DEFAULT_SGD_SCHED_KWARGS = {
    'milestones': [75, 100],
    'gamma': 0.1
}

DEFAULT_RANGER_BATCH_SIZE = 128
DEFAULT_RANGER_EPOCHS = 60
DEFAULT_RANGER_KWARGS = {
    'lr': 0.001 * (DEFAULT_RANGER_BATCH_SIZE / 32),
    'betas': (0.9, 0.999),
    'nesterov': True,
    'eps': 1e-1
}


def generate_full_path(path):
    return os.path.join(os.getcwd(), path)


def extract_toml(experiment_name, module_name=None):
    relative_path = "experiments/" + experiment_name + "/" + experiment_name\
                    + ".toml"
    full_path = generate_full_path(relative_path)
    assert os.path.exists(full_path)

    exp_toml = toml.load(full_path, _dict=OrderedDict)
    if module_name is not None:
        return exp_toml[module_name]
    return exp_toml


def load_model(model_flag):
    if model_flag == "r32p":
        import base_utils.model.resnet as resnet

        return SequentialImageNetworkMod(resnet.resnet32()).cuda()
    elif model_flag == "r18":
        from pytorch_cifar.models import resnet

        return SequentialImageNetwork(resnet.ResNet18()).cuda()
    elif model_flag == "l1":
        import base_utils.model.lenet as lenet
        return lenet.LeNet1().cuda()
    else:
        raise NotImplementedError


def custom_svd(A, k=None, *, backend="arpack"):
    assert len(A.shape) == 2
    assert k is None or 1 <= k <= max(A.shape)
    if backend == "arpack" or backend is None:
        if k is None:
            k = min(A.shape)
        if k == A.shape[0]:
            A = np.vstack([A, np.zeros((1, A.shape[1]))])
            U, S, V = custom_svd(A, k)
            return U[:-1, :], S, V

        elif k == A.shape[1]:
            A = np.hstack([A, np.zeros((A.shape[0], 1))])
            U, S, V = custom_svd(A, k)
            return U, S, V[:, :-1]

        U, S, V = scipy.sparse.linalg.svds(A, k=k)
        return np.copy(U[:, ::-1]), np.copy(S[::-1]), np.copy(V[::-1])

    elif backend == "lapack":
        U, S, V = scipy.linalg.svd(A, full_matrices=False)
        if k is None or k == min(A.shape):
            return U, S, V
        return U[:, :k], S[:k], V[:k, :]

    elif backend == "irlb":
        import irlb

        U, S, Vt, _, _ = irlb.irlb(A, k)
        return U, S, Vt.T

    raise ValueError(f"Invalid backend {backend}")


def in_notebook():
    """
    Returns ``True`` if the module is running in IPython kernel,
    ``False`` if in IPython shell or other Python shell.
    """
    return "ipykernel" in sys.modules


if in_notebook():
    import tqdm.notebook


def make_pbar(*args, **kwargs):
    pbar_constructor = (
        tqdm.notebook.tqdm if in_notebook() else partial(tqdm.tqdm, 
                                                         dynamic_ncols=True)
    )
    return pbar_constructor(*args, **kwargs)


def get_module_device(module: torch.nn.Module, check=True):
    if check:
        assert len(set(param.device for param in module.parameters())) == 1
    return next(module.parameters()).device


def either_dataloader_dataset_to_both(
    data: Union[DataLoader, Dataset], *, batch_size=None, eval=False, **kwargs
):
    if isinstance(data, DataLoader):
        dataloader = data
        dataset = data.dataset
    elif isinstance(data, Dataset):
        dataset = data
        dl_kwargs = {}

        if eval:
            dl_kwargs.update(dict(batch_size=1000, shuffle=False,
                                  drop_last=False))
        else:
            dl_kwargs.update(dict(batch_size=128, shuffle=True))

        if batch_size is not None:
            dl_kwargs["batch_size"] = batch_size

        dl_kwargs.update(kwargs)

        dataloader = make_dataloader(data, **dl_kwargs)
    else:
        raise NotImplementedError()
    return dataloader, dataset


clf_loss = torch.nn.CrossEntropyLoss()


def clf_correct(y_pred: torch.Tensor, y: torch.Tensor):
    y_hat = y_pred.data.max(1)[1]
    correct = (y_hat == y).long().cpu().sum()
    return correct


def clf_eval(model: torch.nn.Module, data: Union[DataLoader, Dataset]):
    device = get_module_device(model)
    dataloader, _ = either_dataloader_dataset_to_both(data, eval=True)
    total_correct, total_loss = 0.0, 0.0
    with torch.no_grad():
        model.eval()
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            loss = clf_loss(y_pred, y)
            correct = clf_correct(y_pred, y)

            total_correct += correct.item()
            total_loss += loss.item()

    n = len(dataloader.dataset)
    total_correct /= n
    total_loss /= n
    return total_correct, total_loss


def get_mean_lr(opt: optim.Optimizer):
    return np.mean([group["lr"] for group in opt.param_groups])


class FlatThenCosineAnnealingLR(object):
    def __init__(
        self,
        optimizer,
        T_max,
        eta_min=0,
        last_epoch=-1,
        flat_ratio=0.7
    ):
        self.last_epoch = last_epoch
        self.flat_ratio = flat_ratio
        self.T_max = T_max
        self.inner = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            int(T_max * (1 - flat_ratio)),
            eta_min,
            max(-1, last_epoch - flat_ratio * T_max - 1),
        )

    def step(self):
        self.last_epoch += 1
        if self.last_epoch >= self.flat_ratio * self.T_max:
            self.inner.step()

    def state_dict(self):
        result = {
            "inner." + key: value for key, value in self.inner.state_dict()
                                                              .items()
        }
        result.update(
            {key: value for key, value in self.__dict__.items()
             if key != "inner"}
        )
        return result

    def load_state_dict(self, state_dict):
        self.inner.load_state_dict(
            {k[6:]: v for k, v in state_dict.items() if k.startswith("inner.")}
        )
        self.__dict__.update(
            {k: v for k, v in state_dict.items() if not k.startswith("inner.")}
        )


def mini_train(
    *,
    model: torch.nn.Module,
    train_data: Union[DataLoader, Dataset],
    test_data: Union[DataLoader, Dataset] = None,
    batch_size=32,
    opt: optim.Optimizer,
    scheduler,
    epochs: int,
):
    device = get_module_device(model)
    dataloader, _ = either_dataloader_dataset_to_both(train_data,
                                                      batch_size=batch_size)
    n = len(dataloader.dataset)
    total_examples = epochs * n
    with make_pbar(total=total_examples) as pbar:
        for _ in range(1, epochs + 1):
            train_epoch_loss, train_epoch_correct = 0, 0
            model.train()
            for x, y in dataloader:
                x, y = x.to(device), y.to(device)
                minibatch_size = len(x)
                model.zero_grad()
                y_pred = model(x)
                loss = clf_loss(y_pred, y)
                correct = clf_correct(y_pred, y)
                loss.backward()
                opt.step()
                train_epoch_correct += int(correct.item())
                train_epoch_loss += float(loss.item())
                pbar.update(minibatch_size)

            lr = get_mean_lr(opt)
            if scheduler:
                scheduler.step()

            pbar_postfix = {
                "acc": "%.2f" % (train_epoch_correct / n * 100),
                "loss": "%.4g" % (train_epoch_loss / n),
                "lr": "%.3g" % lr,
            }
            if test_data:
                test_epoch_acc, test_epoch_loss = clf_eval(model, test_data)
                pbar_postfix.update(
                    {
                        "tacc": "%.2f" % (test_epoch_acc * 100),
                        "tloss": "%.4g" % test_epoch_loss,
                    }
                )

            pbar.set_postfix(**pbar_postfix)

    return model


def compute_all_reps(
    model: torch.nn.Sequential,
    data: Union[DataLoader, Dataset],
    *,
    layers: Collection[int],
    flat=False,
) -> Dict[int, np.ndarray]:
    device = get_module_device(model)
    dataloader, dataset = either_dataloader_dataset_to_both(data, eval=True)
    n = len(dataset)
    max_layer = max(layers)
    assert max_layer < len(model)

    reps = {}
    x = dataset[0][0][None, ...].to(device)
    for i, layer in enumerate(model):
        if i > max_layer:
            break
        x = layer(x)
        if i in layers:
            inner_shape = x.shape[1:]
            reps[i] = torch.empty(n, *inner_shape)

    with torch.no_grad():
        model.eval()
        start_index = 0
        for x, _ in dataloader:
            x = x.to(device)
            minibatch_size = len(x)
            for i, layer in enumerate(model):
                if i > max_layer:
                    break
                x = layer(x)
                if i in layers:
                    reps[i][start_index: start_index + minibatch_size] =\
                        x.cpu()

            start_index += minibatch_size

    if flat:
        for layer in reps:
            layer_reps = reps[layer]
            reps[layer] = layer_reps.reshape(layer_reps.shape[0], -1)

    return reps


def compute_grads(
    *,
    model: torch.nn.Module,
    data: Union[DataLoader, Dataset],
):
    device = get_module_device(model)
    dataloader, _ = either_dataloader_dataset_to_both(data,
                                                      batch_size=1,
                                                      eval=True)
    grads = []
    labels = []
    model.eval()
    for x, y in dataloader:
        labels.extend(y.numpy())
        x, y = x.to(device), y.to(device)
        model.zero_grad() 
        y_pred = model(x)
        loss = clf_loss(y_pred, y)
        loss.backward()
        grads_xy = []
        for param in model.parameters():
            grads_xy.append(param.grad.cpu().detach().flatten().numpy())
        grads.append(np.concatenate(grads_xy))

    return np.stack(grads, axis=0), labels


def get_train_info(
    params,
    train_flag,
    batch_size=None,
    epochs=None,
    optim_kwargs={},
    scheduler_kwargs={}
):
    if train_flag == "sgd":
        batch_size = batch_size or DEFAULT_SGD_BATCH_SIZE
        epochs = epochs or DEFAULT_SGD_EPOCHS
        kwargs = {**DEFAULT_SGD_KWARGS, **optim_kwargs}
        sched_kwargs = {**DEFAULT_SGD_SCHED_KWARGS, **scheduler_kwargs}
        opt = optim.SGD(params, **kwargs)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(opt, **sched_kwargs)
    elif train_flag == "ranger":
        batch_size = batch_size or DEFAULT_RANGER_BATCH_SIZE
        epochs = epochs or DEFAULT_RANGER_EPOCHS
        kwargs = {**DEFAULT_RANGER_KWARGS, **optim_kwargs}
        opt = ranger.Ranger(params, **kwargs)
        lr_scheduler = FlatThenCosineAnnealingLR(opt, T_max=epochs)

    return batch_size, epochs, opt, lr_scheduler
