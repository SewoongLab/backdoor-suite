import sys

from pathlib import Path
from tqdm import trange
import subprocess

from util import *
from datasets import *

experiment_name, module_name = sys.argv[1], sys.argv[2]

args = extract_toml(experiment_name, module_name)

call = ["julia", "--project=.", "defenses/julia/run_filters.jl"]
call.append(generate_full_path(args["input"]))
call.append(generate_full_path(args["output"]))
call.append(args["defense"])
call.append(str(args["target_label"]))
call.append(str(args["poisons"]))

subprocess.run(call)