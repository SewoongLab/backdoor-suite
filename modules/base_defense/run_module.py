"""
Implementation of a basic defense module.
Runs a defense using pretrained representations.
"""

import subprocess
import sys
import os

sys.path.insert(0, os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..')
    ))

from base_utils.util import extract_toml, generate_full_path


def run(experiment_name, module_name):
    """
    Runs a defense given pretrained representations.

    :param experiment_name: Name of the experiment in configuration.
    :param module_name: Name of the module in configuration.
    """

    args = extract_toml(experiment_name, module_name)

    file_name = f"run_{args['defense']}.jl"
    call = ["julia",
            "--project=.",
            "modules/base_defense/" + file_name]
    call.append(generate_full_path(args["input"]))
    call.append(generate_full_path(args["output"]))
    call.append(str(args["target_label"]))
    call.append(str(args["poisons"]))

    subprocess.run(call)


if __name__ == "__main__":
    experiment_name, module_name = sys.argv[1], sys.argv[2]
    run(experiment_name, module_name)
