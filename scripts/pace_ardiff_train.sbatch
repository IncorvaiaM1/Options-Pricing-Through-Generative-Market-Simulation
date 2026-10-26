#!/bin/bash
#SBATCH -J ardiff_train
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -t 04:00:00
#SBATCH -o logs/%x-%j.out

# Usage (from repo root): sbatch scripts/pace_ardiff_train.sbatch --head diffusion --out runs/diff
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs runs results
module load anaconda3
conda activate gdl
nvidia-smi --query-gpu=name,memory.total --format=csv
python -u -m ardiff.train "$@"
