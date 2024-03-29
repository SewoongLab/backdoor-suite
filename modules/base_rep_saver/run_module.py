"""
Implementation of a basic representation saving module.
Saves representations per class for a previously trained model.
"""

import sys

import torch
from pathlib import Path
from tqdm import trange
import numpy as np

from modules.base_utils.datasets import pick_poisoner, generate_datasets,\
                                LabelSortedDataset
from modules.base_utils.util import extract_toml, generate_full_path, clf_eval,\
                            load_model, compute_all_reps


def run(experiment_name, module_name):
    """
    Extracts representations from a pretrained model.

    :param experiment_name: Name of the experiment in configuration.
    :param module_name: Name of the module in configuration.
    """

    args = extract_toml(experiment_name, module_name)

    model_file = generate_full_path(args["input"])
    model_flag = args["model"]
    model = load_model(model_flag)
    model.load_state_dict(torch.load(model_file))
    dataset_flag = args["dataset"]
    eps = args["poisons"]
    poisoner_flag = args["poisoner"]
    clean_label = args["source_label"]
    target_label = args["target_label"]
    output_folder = args["output"]

    reduce_amplitude = variant = None
    if "reduce_amplitude" in args:
        reduce_amplitude = None if args['reduce_amplitude'] < 0\
                                else args['reduce_amplitude']
        variant = args['variant']

    print("Evaluating...")

    poisoner, all_poisoner = pick_poisoner(poisoner_flag, dataset_flag,
                                           target_label, reduce_amplitude)

    poison_train, test, poison_test, all_poison_test = \
        generate_datasets(dataset_flag, poisoner, all_poisoner, eps, clean_label,
                          target_label, None, variant)

    clean_train_acc = clf_eval(model, poison_train.clean_dataset)[0]
    poison_train_acc = clf_eval(model, poison_train.poison_dataset)[0]
    print(f"{clean_train_acc=}")
    print(f"{poison_train_acc=}")

    clean_test_acc = clf_eval(model, test)[0]
    poison_test_acc = clf_eval(model, poison_test.poison_dataset)[0]
    all_poison_test_acc = clf_eval(model, all_poison_test.poison_dataset)[0]

    print(f"{clean_test_acc=}")
    print(f"{poison_test_acc=}")
    print(f"{all_poison_test_acc=}")

    lsd = LabelSortedDataset(poison_train)

    if model_flag == "r32p":
        layer = 14
    elif model_flag == "r18":
        layer = 13

    for i in trange(lsd.n, dynamic_ncols=True):
        target_reps = compute_all_reps(model, lsd.subset(i), layers=[layer],
                                    flat=True)[
            layer
        ]
        output_folder_path = generate_full_path(output_folder)
        filename = output_folder_path + str(i) + ".npy"
        Path(output_folder_path).mkdir(parents=True, exist_ok=True)
        np.save(filename, target_reps.numpy())


if __name__ == "__main__":
    experiment_name, module_name = sys.argv[1], sys.argv[2]
    run(experiment_name, module_name)
