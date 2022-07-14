import os
from re import A
import yaml

import argparse
parser = argparse.ArgumentParser()

parser.add_argument("--config_yaml", type=str, help="yaml file containing config", default="launch_configs/democonfig.yaml")
args = parser.parse_args()

# load yaml file
with open(args.config_yaml, "r") as f:
    config = yaml.safe_load(f)

all_different_args = config["different"]
all_different_args.update(config["different_flags"])

keys = list(all_different_args.keys())
num_processes = len(all_different_args[keys[0]])

# resolve arguments per process
for i in range(num_processes):
    bsub_string = f'bsub -W {config["minutes"]} -n 1 -R "rusage[mem={config["memory"]},ngpus_excl_p=1]" python train.py'
    for item in config["different"].items():
        bsub_string += f" --{item[0]}={item[1][i]}"
    for item in config["different_flags"].items():
        bsub_string += f" --{item[0]}={item[1][i]}"
    for item in config["same"].items():
        if item[1]:
            bsub_string += f" --{item[0]}={item[1]}"
    # same flags
    for item in config["same_flags"].items():
        if item[1]:
            bsub_string += f" --{item[0]}={item[1]}"
    print(bsub_string)
    print()

    # run terminal command with string
    os.system(bsub_string)
