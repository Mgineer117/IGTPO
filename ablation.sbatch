#!/bin/bash
#SBATCH --job-name=igtpo-ablation
#SBATCH --account=huytran1-ic
#SBATCH --partition=eng-research-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=17
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:5
#SBATCH --time=2-00:00:00
#SBATCH --output=igtpo-ablation.o%j
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=minjae5@illinois.edu


# === Run Experiments in Parallel ===
CUDA_VISIBLE_DEVICES=0 python3 main.py --env-name pointmaze-medium --project igtpo-ablation --algo-name igtpo --num-local-updates 5 & 
CUDA_VISIBLE_DEVICES=0 python3 main.py --env-name pointmaze-medium --project igtpo-ablation --algo-name igtpo --num-local-updates 10 & 
CUDA_VISIBLE_DEVICES=0 python3 main.py --env-name pointmaze-medium --project igtpo-ablation --algo-name igtpo --num-local-updates 20 & 

CUDA_VISIBLE_DEVICES=1 python3 main.py --env-name pointmaze-medium --project igtpo-ablation --algo-name igtpo --entropy-scaler 0.0 & 
CUDA_VISIBLE_DEVICES=1 python3 main.py --env-name pointmaze-medium --project igtpo-ablation --algo-name igtpo --entropy-scaler 0.1 &
CUDA_VISIBLE_DEVICES=1 python3 main.py --env-name pointmaze-medium --project igtpo-ablation --algo-name igtpo --entropy-scaler 0.01 &
CUDA_VISIBLE_DEVICES=1 python3 main.py --env-name pointmaze-medium --project igtpo-ablation --algo-name igtpo --entropy-scaler 0.0001 &  

CUDA_VISIBLE_DEVICES=2 python3 main.py --env-name pointmaze-medium --project igtpo-ablation --algo-name igtpo --target-kl 0.01 &
CUDA_VISIBLE_DEVICES=2 python3 main.py --env-name pointmaze-medium --project igtpo-ablation --algo-name igtpo --target-kl 0.03 &
CUDA_VISIBLE_DEVICES=2 python3 main.py --env-name pointmaze-medium --project igtpo-ablation --algo-name igtpo --target-kl 0.1 &
  
CUDA_VISIBLE_DEVICES=3 python3 main.py --env-name pointmaze-medium --project igtpo-ablation --algo-name igtpo --igtpo-actor-lr 0.1 & 
CUDA_VISIBLE_DEVICES=3 python3 main.py --env-name pointmaze-medium --project igtpo-ablation --algo-name igtpo --igtpo-actor-lr 0.001 & 
CUDA_VISIBLE_DEVICES=3 python3 main.py --env-name pointmaze-medium --project igtpo-ablation --algo-name igtpo --igtpo-actor-lr 0.0001 & 

CUDA_VISIBLE_DEVICES=4 python3 main.py --env-name pointmaze-medium --project igtpo-ablation --algo-name igtpo --batch-size 1024 & 
CUDA_VISIBLE_DEVICES=4 python3 main.py --env-name pointmaze-medium --project igtpo-ablation --algo-name igtpo --batch-size 2048 & 
CUDA_VISIBLE_DEVICES=4 python3 main.py --env-name pointmaze-medium --project igtpo-ablation --algo-name igtpo --batch-size 4096 & 
CUDA_VISIBLE_DEVICES=4 python3 main.py --env-name pointmaze-medium --project igtpo-ablation --algo-name igtpo --batch-size 16384 & 


# === Wait for all background jobs to finish ===
wait