import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader, Dataset
from functorch import make_functional, vmap, grad_and_value
from typing import Union

from modules.bgmd.GeometricMedian import GeometricMedian
from modules.bgmd.utils import dist_grads_to_model, compress
from modules.base_utils.util import clf_correct, clf_eval, get_mean_lr,\
                                    either_dataloader_dataset_to_both,\
                                    get_module_device, make_pbar

clf_loss = torch.nn.CrossEntropyLoss()


def bgmd_train(
    *,
    model: torch.nn.Module,
    train_data: Union[DataLoader, Dataset],
    test_data: Union[DataLoader, Dataset] = None,
    batch_size=1,
    frac=0.1,
    opt: optim.Optimizer,
    scheduler,
    epochs: int,
):
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    device = get_module_device(model)
    dataloader, _ = either_dataloader_dataset_to_both(train_data,
                                                      batch_size=batch_size)
    n = len(dataloader.dataset)
    total_examples = epochs * n
    G = residual_error = None

    # TODO: Hyperparameters
    gar = GeometricMedian('vardi', 1e-5, 500)

    with make_pbar(total=total_examples) as pbar:
        for _ in range(1, epochs + 1):
            train_epoch_loss, train_epoch_correct = 0, 0
            model.train()
            for (x, y) in dataloader:
                # Extract functional model and parameters
                fmodel, params = make_functional(model)
                for param in params:
                    param.requires_grad_(False)

                # Define functions to vmap
                def compute_loss_stateless_model(params, sample, target):
                    batch, targets = sample.unsqueeze(0), target.unsqueeze(0)

                    predictions = fmodel(params, batch)
                    loss = clf_loss(predictions, targets)
                    return loss, predictions.squeeze(0)

                ft_compute_grad = grad_and_value(compute_loss_stateless_model,
                                                 has_aux=True)
                ft_compute_sample_grad = vmap(ft_compute_grad,
                                              in_dims=(None, 0, 0))

                # Forward Pass and Gradient Extraction
                x, y = x.to(device), y.to(device)
                G, (loss, y_pred) = ft_compute_sample_grad(params, x, y)
                correct = clf_correct(y_pred, y)
                G = [torch.flatten(x, start_dim=1) for x in G]
                G = torch.cat(G, dim=1).cpu().numpy()

                # Dimensionality Reduction
                lr = opt.param_groups[0]['lr']
                if residual_error is None:
                    residual_error = np.zeros_like(G, dtype=G.dtype)
                I_k, G_sparse, residual_error =\
                    compress(G, lr, frac, residual_error)
                
                # Gradient aggregation via GM
                agg_g = gar.aggregate(G=G_sparse, ix=I_k)
                gar.agg_time = 0
                gar.num_iter = 0

                # Update Model Grads with aggregated g : i.e. compute \tilde(g)
                model.zero_grad()
                dist_grads_to_model(grads=agg_g, learner=model)
                model.to(device)

                # Now Do an optimizer step with x_t+1 = x_t - \eta \tilde(g)
                opt.step()
                pbar.update(batch_size)
                train_epoch_correct += int(correct)
                train_epoch_loss += float(loss.mean())

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
