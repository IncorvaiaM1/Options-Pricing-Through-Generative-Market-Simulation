#!/bin/bash
# Submits the first full comparison on PACE: AR+diffusion vs two tokenized-AR baselines
# (same backbone), then one eval job that waits for all three.
#   bash scripts/run_thesis_v0.sh
set -e
cd "$(dirname "$0")/.."
mkdir -p logs

COMMON="--n_train 20000 --steps 20000 --batch 256 --d 128 --n_layers 4"

j1=$(sbatch --parsable scripts/pace_ardiff_train.sbatch --head diffusion --out runs/v0_diff $COMMON)
j2=$(sbatch --parsable scripts/pace_ardiff_train.sbatch --head token --n_bins 256 --bin_scheme quantile --out runs/v0_tok256q $COMMON)
j3=$(sbatch --parsable scripts/pace_ardiff_train.sbatch --head token --n_bins 1024 --bin_scheme uniform --bin_clip 8 --out runs/v0_tok1024u $COMMON)
echo "train jobs: $j1 $j2 $j3"

j4=$(sbatch --parsable --dependency=afterok:$j1:$j2:$j3 scripts/pace_ardiff_eval.sbatch \
  --ckpts runs/v0_diff/ckpt.pkl runs/v0_tok256q/ckpt.pkl runs/v0_tok1024u/ckpt.pkl \
  --labels ar_diffusion ar_token256q ar_token1024u \
  --n_paths 8192 --out results/v0)
echo "eval job: $j4   (watch: squeue -u \$USER ; results in results/v0/report.md)"
