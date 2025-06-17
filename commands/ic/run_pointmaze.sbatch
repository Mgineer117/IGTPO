#!/bin/bash
#SBATCH --job-name=igtpo
#SBATCH --account=huytran1-ic
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH --time=3-00:00:00
#SBATCH --output=igtpo.o%j
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=minjae5@illinois.edu

ulimit -n 4096  # raise file descriptor limit

# Load conda
source ~/.bashrc
# Or: source /sw/apps/anaconda3/2024.10/etc/profile.d/conda.sh  # if ~/.bashrc doesn't source conda

# Activate your conda environment
conda activate igtpo

# === Run Experiments in Parallel ===
CUDA_VISIBLE_DEVICES=0 python3 main.py --env-name pointmaze-medium --project igtpo --algo-name ppo & 
CUDA_VISIBLE_DEVICES=0 python3 main.py --env-name pointmaze-medium --project igtpo --algo-name trpo & 
CUDA_VISIBLE_DEVICES=0 python3 main.py --env-name pointmaze-medium --project igtpo --algo-name psne & 
CUDA_VISIBLE_DEVICES=0 python3 main.py --env-name pointmaze-medium --project igtpo --algo-name drndppo & 
CUDA_VISIBLE_DEVICES=1 python3 main.py --env-name pointmaze-medium --project igtpo --algo-name eigenoption & 
CUDA_VISIBLE_DEVICES=1 python3 main.py --env-name pointmaze-medium --project igtpo --algo-name igtpo --intrinsic-reward-mode eigenpurpose & 
CUDA_VISIBLE_DEVICES=1 python3 main.py --env-name pointmaze-medium --project igtpo --algo-name igtpo --intrinsic-reward-mode drnd & 
CUDA_VISIBLE_DEVICES=1 python3 main.py --env-name pointmaze-medium --project igtpo --algo-name igtpo --intrinsic-reward-mode all & 

# === Wait for all background jobs to finish ===
wait