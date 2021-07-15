import sys

from pathlib import Path
from tqdm import trange

from util import *
from datasets import *

experiment_name, module_name = sys.argv[1], sys.argv[2]

args = extract_toml(experiment_name, module_name)

model_file = generate_full_path(args["input"])
model_flag = args["model"]
model = load_model(model_flag)
model.load_state_dict(torch.load(model_file))
eps = args["poisons"]
poisoner_flag = args["poisoner"]
clean_label = args["source_label"]
target_label = args["target_label"]
output_folder = args["output"]


print("Evaluating...")

poisoner, all_poisoner = pick_poisoner(poisoner_flag, target_label)

poison_cifar_train, cifar_test, poison_cifar_test, all_poison_cifar_test = \
    generate_datasets(poisoner, all_poisoner, eps, clean_label, target_label)

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

lsd = LabelSortedDataset(poison_cifar_train)

if model_flag == "r32p":
    layer = 14
elif model_flag == "r18":
    layer = 13

for i in trange(lsd.n, dynamic_ncols=True):
    target_reps = compute_all_reps(model, lsd.subset(i), layers=[layer], flat=True)[
        layer
    ]
    output_folder_path = generate_full_path(output_folder)
    filename = output_folder_path + str(i) + ".npy"
    Path(output_folder_path).mkdir(parents=True, exist_ok=True)
    np.save(filename, target_reps.numpy())
