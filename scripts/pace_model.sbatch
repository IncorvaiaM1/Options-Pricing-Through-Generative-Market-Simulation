#!/bin/bash
#SBATCH -J model
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -t 06:00:00
#SBATCH -o logs/%x-%j.out

# One model on one data folder, on a GPU node:
#   sbatch -J a6_A scripts/pace_model.sbatch pure_diffusion bench_data/A
#   MARKET=bench_data/market/2026-10-27 sbatch --export=ALL -J a6_tech scripts/pace_model.sbatch pure_diffusion bench_data/real/megacap_tech
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
module load anaconda3
conda activate gdl
nvidia-smi --query-gpu=name --format=csv,noheader || true
bash scripts/run_model.sh "$@"
