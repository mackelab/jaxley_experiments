#!/bin/bash
#SBATCH --array=0-9
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=h100-ferranti
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=1:00:00

# Your commands to run the job
echo "Hello from task $SLURM_ARRAY_TASK_ID"


python run_ga.py --seed $SLURM_ARRAY_TASK_ID --smc_steps=10 --setup 473601979  --loss_fn="dtw_reg_lowvar" --n_particles=10 --t_max=199.5
