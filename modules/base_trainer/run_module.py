"""
Implementation of a basic training module.
Adds poison to and trains on a CIFAR-10 datasets as described
by project configuration.
"""

import sys
import os

import torch
from torch import optim
import numpy as np

sys.path.insert(0, os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..')
    ))

from ranger_opt.ranger import ranger2020 as ranger
from base_utils.datasets import pick_poisoner, generate_datasets
from base_utils.util import extract_toml, load_model, generate_full_path,\
                            FlatThenCosineAnnealingLR, clf_eval, mini_train


def run(experiment_name, module_name):
    """
    Runs poisoning and training.

    :param experiment_name: Name of the experiment in configuration.
    :param module_name: Name of the module in configuration.
    """

    retrain = module_name == "base_retrainer"

    args = extract_toml(experiment_name, module_name)

    model_flag = args["model"]
    dataset_flag = args["dataset"]
    train_flag = args["trainer"]
    eps = args["poisons"]
    poisoner_flag = args["poisoner"]
    clean_label = args["source_label"]
    target_label = args["target_label"]
    output_path = args["output"]

    reduce_amplitude = variant = None
    if "reduce_amplitude" in args:
        reduce_amplitude = None if args['reduce_amplitude'] < 0\
                                else args['reduce_amplitude']
        variant = args['variant']

    model = load_model(model_flag)
    target_mask_ind = None

    print(f"{model_flag=} {clean_label=} {target_label=} {poisoner_flag=} {eps=}")

    if retrain:
        input_path = generate_full_path(args["input"])
        target_mask = np.load(input_path)
        target_mask_ind = [i for i in range(len(target_mask)) if not target_mask[i]]
        poison_removed = np.sum(target_mask[-eps:])
        clean_removed = np.sum(target_mask) - poison_removed
        print(f"{poison_removed=} {clean_removed=}")

    print("Building datasets...")

    poisoner, all_poisoner = pick_poisoner(poisoner_flag,
                                           dataset_flag,
                                           target_label,
                                           reduce_amplitude)

    poison_train, test, poison_test, all_poison_test =\
        generate_datasets(dataset_flag, poisoner, all_poisoner, eps, clean_label,
                          target_label, target_mask_ind, variant)

    if train_flag == "sgd":
        batch_size = 128
        epochs = 200
        opt = torch.optim.SGD(
            model.parameters(),
            lr=0.1,
            momentum=0.9,
            nesterov=True,
            weight_decay=2e-4
        )
        lr_scheduler = optim.lr_scheduler.MultiStepLR(opt,
                                                      milestones=[75, 150],
                                                      gamma=0.1)

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

    print("Training...")

    mini_train(
        model=model,
        train_data=poison_train,
        test_data=test,
        batch_size=batch_size,
        opt=opt,
        scheduler=lr_scheduler,
        epochs=epochs,
    )

    print("Evaluating...")

    if not retrain:
        clean_train_acc = clf_eval(model,
                                   poison_train.clean_dataset)[0]
        poison_train_acc = clf_eval(model,
                                    poison_train.poison_dataset)[0]
        print(f"{clean_train_acc=}")
        print(f"{poison_train_acc=}")

    clean_test_acc = clf_eval(model, test)[0]
    poison_test_acc = clf_eval(model, poison_test.poison_dataset)[0]
    all_poison_test_acc = clf_eval(model,
                                   all_poison_test.poison_dataset)[0]

    print(f"{clean_test_acc=}")
    print(f"{poison_test_acc=}")
    print(f"{all_poison_test_acc=}")

    print("Saving model...")
    torch.save(model.state_dict(), generate_full_path(output_path))


if __name__ == "__main__":
    experiment_name, module_name = sys.argv[1], sys.argv[2]
    run(experiment_name, module_name)
