#!/bin/bash
#SBATCH --array=0
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=h100-ferranti
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=8:00:00

# Your commands to run the job
echo "Hello from task $SLURM_ARRAY_TASK_ID"

python run_gd.py --seed $SLURM_ARRAY_TASK_ID --steps=50 --setup 480353286 --n_particles=1000 --t_max=199.5 --lr=0.0005 --momentum=0.0 --cost_fn_power=1.0
