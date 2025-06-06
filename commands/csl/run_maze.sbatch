#!/bin/bash
#SBATCH --job-name=igtpo-ablation
#SBATCH --account=huytran1-ic
#SBATCH --partition=csl
#SBATCH --nodes=1
#SBATCH --ntasks=6
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
CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 python3 main.py --env-name maze-v0 --project igtpo --algo-name ppo --num-runs 10 & 
CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 python3 main.py --env-name maze-v0 --project igtpo --algo-name trpo --num-runs 10 & 
CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 python3 main.py --env-name maze-v0 --project igtpo --algo-name psne --num-runs 10 & 
CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 python3 main.py --env-name maze-v0 --project igtpo --algo-name drndppo --num-runs 10 & 
CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 python3 main.py --env-name maze-v0 --project igtpo --algo-name eigenoption --num-runs 10 & 
CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 python3 main.py --env-name maze-v0 --project igtpo --algo-name igtpo --num-runs 10 & 

# === Wait for all background jobs to finish ===
wait