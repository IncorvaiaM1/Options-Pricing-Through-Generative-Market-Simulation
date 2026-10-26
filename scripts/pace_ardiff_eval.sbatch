#!/bin/bash
#SBATCH -J ardiff_eval
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH -t 02:00:00
#SBATCH -o logs/%x-%j.out

# Usage: sbatch scripts/pace_ardiff_eval.sbatch --ckpts runs/diff/ckpt.pkl runs/tok256q/ckpt.pkl --out results/v0
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs results
module load anaconda3
conda activate gdl
python -u -m ardiff.evaluate "$@"
