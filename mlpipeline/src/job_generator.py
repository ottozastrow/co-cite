import os
from re import A
import yaml

import argparse
parser = argparse.ArgumentParser()

parser.add_argument("--config_yaml", type=str, help="yaml file containing config", default='launch_configs/debug.yaml')
parser.add_argument("--only", help="of all configs in this file only run number 1,2 ", type=str, default=None)
args = parser.parse_args()


# load yaml file
with open(args.config_yaml, "r") as f:
    config = yaml.safe_load(f)

all_different_args = config["different"].copy()
all_different_args.update(config["different_flags"])

keys = list(all_different_args.keys())
if len(keys) == 0:
    num_processes = 1
else:
    num_processes = len(all_different_args[keys[0]])

processes = list(range(num_processes))
if args.only:
    only_indexes = args.only.split(",")
    processes = [int(i) for i in only_indexes]

# resolve arguments per process
for i in processes:
    # all processes args
    bsub_string = f'bsub -W {config["minutes"]} -n 1 -R "rusage[mem={config["memory"]},ngpus_excl_p=1]" python train.py'
    
    for item in config["different"].items():
        bsub_string += f" --{item[0]}='{item[1][i]}'"

    # per process flags
    if len(config["different_flags"]) > 0:
        for item in config["different_flags"].items():
            if item[1][i]:
                bsub_string += f" --{item[0]}"
    
    # same keyword args
    for item in config["same"].items():
        if item[1]:
            bsub_string += f" --{item[0]}='{item[1]}'"
    # all processes flags
    if len(config["same_flags"]) > 0:
        for flag in config["same_flags"]:
            bsub_string += f" --{flag}"
    print(bsub_string)
    print()

    # run terminal command with string
    os.system(bsub_string)
