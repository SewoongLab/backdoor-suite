import torch
from torch import nn, optim
from pathlib import Path
from ranger_opt.ranger import ranger2020 as ranger

from util import *
from datasets import *
import re

import sys
import os

experiment_name, module_name = sys.argv[1], sys.argv[2]
retrain = module_name == "base_retrainer"

args = extract_toml(experiment_name, module_name)

model_flag = args["model"]
train_flag = args["trainer"]
eps = args["poisons"]
poisoner_flag = args["poisoner"]
clean_label = args["source_label"]
target_label = args["target_label"]
output_path = args["output"]

model = load_model(model_flag)
target_mask_ind = None

print(f"{model_flag=} {clean_label=} {target_label=} {poisoner_flag=} {eps=}")

if retrain:
    input_path = generate_full_path(args["input"])
    target_mask = np.load(input_path)
    assert len(target_mask) == 5000 + eps
    target_mask_ind = [i for i in range(5000 + eps) if not target_mask[i]]
    poison_removed = np.sum(target_mask[-eps:])
    clean_removed = np.sum(target_mask) - poison_removed
    print(f"{poison_removed=} {clean_removed=}")

print("Building datasets...")

poisoner, all_poisoner = pick_poisoner(poisoner_flag, target_label)

poison_cifar_train, cifar_test, poison_cifar_test, all_poison_cifar_test = \
    generate_datasets(poisoner, all_poisoner, eps, clean_label, target_label, target_mask_ind)


if train_flag == "sgd":
    batch_size = 128
    epochs = 200
    opt = torch.optim.SGD(
        model.parameters(), lr=0.1, momentum=0.9, nesterov=True, weight_decay=2e-4
    )
    lr_scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=[75, 150], gamma=0.1)

elif train_flag == "ranger":
    batch_size = 128
    epochs = 60
    opt = ranger.Ranger(
        model.parameters(),
        lr=0.001 * (batch_size / 32),
        weight_decay=1e-1,
        betas=(0.9, 0.999),
        eps=1e-1,
    )
    lr_scheduler = FlatThenCosineAnnealingLR(opt, T_max=epochs)

if __name__ == "__main__":
    print("Training...")

    mini_train(
        model=model,
        train_data=poison_cifar_train,
        test_data=cifar_test,
        batch_size=batch_size,
        opt=opt,
        scheduler=lr_scheduler,
        epochs=epochs,
    )

    print("Evaluating...")

    if not retrain:
        clean_train_acc = clf_eval(model, poison_cifar_train.clean_dataset)[0]
        poison_train_acc = clf_eval(model, poison_cifar_train.poison_dataset)[0]
        print(f"{clean_train_acc=}")
        print(f"{poison_train_acc=}")

    clean_test_acc = clf_eval(model, cifar_test)[0]
    poison_test_acc = clf_eval(model, poison_cifar_test.poison_dataset)[0]
    all_poison_test_acc = clf_eval(model, all_poison_cifar_test.poison_dataset)[0]

    print(f"{clean_test_acc=}")
    print(f"{poison_test_acc=}")
    print(f"{all_poison_test_acc=}")

    print("Saving model...")
    torch.save(model.state_dict(), generate_full_path(output_path))
