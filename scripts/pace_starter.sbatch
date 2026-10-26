#!/bin/bash
#SBATCH -J starter
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -t 04:00:00
#SBATCH -o logs/%x-%j.out

# Generic GPU job for any starter or harness command. From the repo root:
#   sbatch -J a6 scripts/pace_starter.sbatch starters.a6_pure_diffusion --data bench_data/A --out runs/a6
#   sbatch -J np scripts/pace_starter.sbatch starters.b3_neural_process
# First argument is the python module, the rest are its arguments.
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs runs results
module load anaconda3
conda activate gdl
MOD=$1; shift
python -u -m "$MOD" "$@"
