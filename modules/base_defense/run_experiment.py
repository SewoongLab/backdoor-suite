import subprocess
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from base_utils.util import *

experiment_name, module_name = sys.argv[1], sys.argv[2]

args = extract_toml(experiment_name, module_name)

call = ["julia", "--project=.", "modules/base_defense/src/julia/run_filters.jl"]
call.append(generate_full_path(args["input"]))
call.append(generate_full_path(args["output"]))
call.append(args["defense"])
call.append(str(args["target_label"]))
call.append(str(args["poisons"]))

subprocess.run(call)