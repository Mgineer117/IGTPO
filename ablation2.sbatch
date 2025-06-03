#!/bin/bash
#SBATCH --job-name=igtpo-ablation
#SBATCH --account=huytran1-ic
#SBATCH --partition=eng-research-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=7
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH --time=2-00:00:00
#SBATCH --output=igtpo-ablation.o%j
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=minjae5@illinois.edu


# === Run Experiments in Parallel ===  
CUDA_VISIBLE_DEVICES=0 python3 main.py --env-name pointmaze-medium --project igtpo-ablation --algo-name igtpo --igtpo-actor-lr 0.1 & 
CUDA_VISIBLE_DEVICES=0 python3 main.py --env-name pointmaze-medium --project igtpo-ablation --algo-name igtpo --igtpo-actor-lr 0.001 & 
CUDA_VISIBLE_DEVICES=0 python3 main.py --env-name pointmaze-medium --project igtpo-ablation --algo-name igtpo --igtpo-actor-lr 0.0001 & 

CUDA_VISIBLE_DEVICES=1 python3 main.py --env-name pointmaze-medium --project igtpo-ablation --algo-name igtpo --batch-size 1024 & 
CUDA_VISIBLE_DEVICES=1 python3 main.py --env-name pointmaze-medium --project igtpo-ablation --algo-name igtpo --batch-size 2048 & 
CUDA_VISIBLE_DEVICES=1 python3 main.py --env-name pointmaze-medium --project igtpo-ablation --algo-name igtpo --batch-size 4096 & 
CUDA_VISIBLE_DEVICES=1 python3 main.py --env-name pointmaze-medium --project igtpo-ablation --algo-name igtpo --batch-size 16384 & 


# === Wait for all background jobs to finish ===
wait