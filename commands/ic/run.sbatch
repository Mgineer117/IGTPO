#!/bin/bash
#SBATCH --job-name=igtpo-ablation
#SBATCH --account=huytran1-ic
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=11
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:4
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
CUDA_VISIBLE_DEVICES=0 python3 main.py --env-name pointmaze-medium --project igtpo --algo-name eigenoption & 
CUDA_VISIBLE_DEVICES=0 python3 main.py --env-name pointmaze-medium --project igtpo --algo-name ppo & 
CUDA_VISIBLE_DEVICES=0 python3 main.py --env-name pointmaze-medium --project igtpo --algo-name igtpo & 
CUDA_VISIBLE_DEVICES=0 python3 main.py --env-name pointmaze-medium --project igtpo --algo-name drndppo & 

CUDA_VISIBLE_DEVICES=1 python3 main.py --env-name fetch-reach --project igtpo --algo-name eigenoption & 
CUDA_VISIBLE_DEVICES=1 python3 main.py --env-name fetch-reach --project igtpo --algo-name ppo & 
CUDA_VISIBLE_DEVICES=1 python3 main.py --env-name fetch-reach --project igtpo --algo-name igtpo & 
CUDA_VISIBLE_DEVICES=1 python3 main.py --env-name fetch-reach --project igtpo --algo-name drndppo & 

CUDA_VISIBLE_DEVICES=2 python3 main.py --env-name maze-v0 --project igtpo --algo-name eigenoption & 
sleep 1800
CUDA_VISIBLE_DEVICES=2 python3 main.py --env-name maze-v0 --project igtpo --algo-name ppo & 
CUDA_VISIBLE_DEVICES=2 python3 main.py --env-name maze-v0 --project igtpo --algo-name igtpo & 
sleep 1800
CUDA_VISIBLE_DEVICES=2 python3 main.py --env-name maze-v0 --project igtpo --algo-name drndppo & 

CUDA_VISIBLE_DEVICES=3 python3 main.py --env-name ninerooms-v0 --project igtpo --algo-name eigenoption & 
sleep 1800
CUDA_VISIBLE_DEVICES=3 python3 main.py --env-name ninerooms-v0 --project igtpo --algo-name ppo & 
CUDA_VISIBLE_DEVICES=3 python3 main.py --env-name ninerooms-v0 --project igtpo --algo-name igtpo & 
sleep 1800
CUDA_VISIBLE_DEVICES=3 python3 main.py --env-name ninerooms-v0 --project igtpo --algo-name drndppo & 

# === Wait for all background jobs to finish ===
wait