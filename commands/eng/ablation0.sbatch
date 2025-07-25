#!/bin/bash
#SBATCH --job-name=igtpo-ablation
#SBATCH --account=huytran1-ic
#SBATCH --partition=eng-research-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=7
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:4
#SBATCH --time=2-00:00:00
#SBATCH --output=igtpo-ablation.o%j
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=minjae5@illinois.edu

ulimit -n 4096  # raise file descriptor limit

# Load conda
source ~/.bashrc
# Or: source /sw/apps/anaconda3/2024.10/etc/profile.d/conda.sh  # if ~/.bashrc doesn't source conda

# Activate your conda environment
conda activate igtpo

# === Run Experiments in Parallel ===
CUDA_VISIBLE_DEVICES=0 python3 main.py --env-name maze-v0 --project igtpo-ablation --algo-name igtpo --num-inner-updates 5 --intrinsic-reward-mode allo & 
CUDA_VISIBLE_DEVICES=0 python3 main.py --env-name maze-v0 --project igtpo-ablation --algo-name igtpo --num-inner-updates 10 --intrinsic-reward-mode allo & 
CUDA_VISIBLE_DEVICES=1 python3 main.py --env-name maze-v0 --project igtpo-ablation --algo-name igtpo --num-inner-updates 15 --intrinsic-reward-mode allo & 
CUDA_VISIBLE_DEVICES=1 python3 main.py --env-name maze-v0 --project igtpo-ablation --algo-name igtpo --num-inner-updates 20 --intrinsic-reward-mode allo & 

CUDA_VISIBLE_DEVICES=2 python3 main.py --env-name maze-v0 --project igtpo-ablation --algo-name igtpo --igtpo-actor-lr 0.0001 --intrinsic-reward-mode allo &
CUDA_VISIBLE_DEVICES=2 python3 main.py --env-name maze-v0 --project igtpo-ablation --algo-name igtpo --igtpo-actor-lr 0.005 --intrinsic-reward-mode allo &
CUDA_VISIBLE_DEVICES=3 python3 main.py --env-name maze-v0 --project igtpo-ablation --algo-name igtpo --igtpo-actor-lr 0.01 --intrinsic-reward-mode allo &

# === Wait for all background jobs to finish ===
wait