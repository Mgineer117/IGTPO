#!/bin/bash
#SBATCH --job-name=igtpo
#SBATCH --account=huytran1-ic
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
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
CUDA_VISIBLE_DEVICES=0 python3 main.py --algo-name igtpo --project igtpo --env-name fourrooms-v0 --intrinsic-reward-mode allo &
CUDA_VISIBLE_DEVICES=0 python3 main.py --algo-name igtpo --project igtpo --env-name maze-v2 --intrinsic-reward-mode allo &
CUDA_VISIBLE_DEVICES=0 python3 main.py --algo-name igtpo --project igtpo --env-name fourrooms-v0 --intrinsic-reward-mode allo-drnd &
CUDA_VISIBLE_DEVICES=0 python3 main.py --algo-name igtpo --project igtpo --env-name maze-v2 --intrinsic-reward-mode allo-drnd &

# === Wait for all background jobs to finish ===
wait