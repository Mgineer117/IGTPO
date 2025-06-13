#!/bin/bash
#SBATCH --job-name=igtpo-ablation
#SBATCH --account=huytran1-ic
#SBATCH --partition=csl
#SBATCH --nodes=1
#SBATCH --ntasks=7
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=7-00:00:00
#SBATCH --output=igtpo-ablation.o%j
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=minjae5@illinois.edu

# Load conda
source ~/.bashrc
# Or: source /sw/apps/anaconda3/2024.10/etc/profile.d/conda.sh  # if ~/.bashrc doesn't source conda

# Activate your conda environment
conda activate igtpo

# === Run Experiments in Parallel ===
CUDA_VISIBLE_DEVICES=0 python3 main.py --env-name maze-v0 --project igtpo --algo-name ppo & 
CUDA_VISIBLE_DEVICES=0 python3 main.py --env-name maze-v0 --project igtpo --algo-name trpo & 
CUDA_VISIBLE_DEVICES=0 python3 main.py --env-name maze-v0 --project igtpo --algo-name psne & 
CUDA_VISIBLE_DEVICES=0 python3 main.py --env-name maze-v0 --project igtpo --algo-name drndppo & 
CUDA_VISIBLE_DEVICES=0 python3 main.py --env-name maze-v0 --project igtpo --algo-name eigenoption & 
CUDA_VISIBLE_DEVICES=0 python3 main.py --env-name maze-v0 --project igtpo --algo-name igtpo --intrinsic-reward-mode eigenpurpose & 
CUDA_VISIBLE_DEVICES=0 python3 main.py --env-name maze-v0 --project igtpo --algo-name igtpo --intrinsic-reward-mode drnd & 

# === Wait for all background jobs to finish ===
wait