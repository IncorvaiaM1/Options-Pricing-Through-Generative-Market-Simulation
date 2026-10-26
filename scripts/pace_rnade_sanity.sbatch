#!/bin/bash
#SBATCH -J rnade_sanity
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t 01:00:00
#SBATCH -o logs/%x-%j.out
## RNADE is tiny, so CPU is fine and schedules faster. Uncomment for a GPU:
##SBATCH --gres=gpu:1

# Submit from the repo root:  sbatch scripts/pace_rnade_sanity.sbatch
# Extra args pass through:    sbatch scripts/pace_rnade_sanity.sbatch --dataset gbm --epochs 20000

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs results

module load anaconda3
conda activate gdl

python -u experiments/rnade_sanity.py "$@"
