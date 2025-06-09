#!/bin/bash
#SBATCH --job-name=igtpo
#SBATCH --account=huytran1-ic
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
#SBATCH --output=igtpo-ablation.o%j
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=minjae5@illinois.edu

# Load conda
source ~/.bashrc
# Or: source /sw/apps/anaconda3/2024.10/etc/profile.d/conda.sh  # if ~/.bashrc doesn't source conda

# Activate your conda environment
conda activate igtpo

# === Run Experiments in Parallel ===
CUDA_VISIBLE_DEVICES=0 python3 main.py --env-name ninerooms-v0 --project igtpo --algo-name igtpo --num-runs 10 & 
CUDA_VISIBLE_DEVICES=0 python3 main.py --env-name maze-v0 --project igtpo --algo-name igtpo --num-runs 10 & 

# === Wait for all background jobs to finish ===
wait