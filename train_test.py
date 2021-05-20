import time
import numpy as np
from torch import optim
import torchvision
from util import *
import datasets
from over9000 import over9000, ranger

# import pytorch_resnet_cifar10.resnet as resnet
import pytorch_cifar.models.resnet as resnet
# import resnet


def train_standard(batch_size=256, epochs=60):
    # model = torchvision.models.resnet50().to(default_device)
    # model = resnet.resnet32().to(default_device)
    model = resnet.ResNet18().to(default_device)

    print(model)
    print(f"model layers: {param_layer_count(model)}")
    print(f"model params: {param_count(model)}")

    # opt = over9000.RangerLars(
    #     model.parameters(), lr=0.1, weight_decay=5e-4, betas=(0.9, 0.999)
    # )
    opt = over9000.RangerLars(
        model.parameters(),
        lr=0.001 * (batch_size / 32),
        weight_decay=1e-1,
        betas=(0.9, 0.999),
        eps=0.1,
    )
    # opt = optim.SGD(model.parameters(), lr=1e-1, momentum=0.9, nesterov=True, weight_decay=5e-4)
    lr_scheduler = FlatThenCosineAnnealingLR(opt, T_max=epochs)
    # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, [75, 150], gamma=0.1, last_epoch=-1)

    cifar_train = datasets.load_cifar_train(batch_size)
    cifar_test = datasets.load_cifar_test(batch_size)
    # cifar_train = datasets.load_poisoned_cifar_train(batch_size)
    # cifar_test = datasets.load_cifar_test(batch_size)

    best_test_acc, best_test_loss = MaxMeter(), MinMeter()
    # train_epoch_loss, train_epoch_acc = AverageMeter(), AverageMeter()

    start_time = time.time()
    for epoch in range(1, epochs + 1):
        epoch_ops, epoch_start_time = 0, time.time()
        lr = get_mean_lr(opt)
        print(
            "=" * 20 + f" Epoch {epoch} @ {epoch_start_time - start_time:.2f}s α = {lr}"
        )

        test_epoch_acc, test_epoch_loss = clf_eval(model, cifar_test)
        print(f"Test Accuracy: {test_epoch_acc:.6f}, Test Loss: {test_epoch_loss:.8f}")

        if best_test_acc.update(test_epoch_acc) | best_test_loss.update(
            test_epoch_loss
        ):
            print("==> new best stats, saving")

        train_epoch_loss = 0.0
        train_epoch_acc = 0
        model.train()
        for x, y in cifar_train:
            x, y = x.to(default_device), y.to(default_device)
            minibatch_size = len(x)
            model.zero_grad()
            y_pred = model(x)
            loss = clf_loss(y_pred, y)
            correct = clf_correct(y_pred, y)

            loss.backward()
            opt.step()

            epoch_ops += minibatch_size
            train_epoch_acc += int(correct.item())
            train_epoch_loss += float(loss.item())

        n = len(cifar_train.dataset)
        train_epoch_acc /= n
        train_epoch_loss /= n
        ops_per_sec = epoch_ops // int(time.time() - epoch_start_time)
        print(
            f"Train Accuracy: {train_epoch_acc:.6f}, Train Loss: {train_epoch_loss:.8f}, ops: {ops_per_sec}"
        )

        lr_scheduler.step()

    test_epoch_acc, test_epoch_loss = clf_eval(model, cifar_test)
    print(f"Test Accuracy: {test_epoch_acc:.6f}, Test Loss: {test_epoch_loss:.8f}")


if __name__ == "__main__":
    train_standard()
