import wandb
import yaml

"""Set up a parameter sweep with wandb
Step 1: run this script
Step 2: run sweep.sh
"""
with open("sweep_config.yml", "r") as f: 
    sweep_config = yaml.safe_load(f)

sweep_id = wandb.sweep(sweep=sweep_config, project="jaxley")
