#!/bin/bash
#SBATCH --array=42
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=h100-ferranti
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=1:00:00

# Your commands to run the job
echo "Hello from task $SLURM_ARRAY_TASK_ID"

python run_gd.py --seed $SLURM_ARRAY_TASK_ID --steps=10 --setup 473601979 --t_max=199.5 --momentum=0.0 --cost_fn_power=1.0
