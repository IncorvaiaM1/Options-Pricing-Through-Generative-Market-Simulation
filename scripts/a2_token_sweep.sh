#!/bin/bash
# A2 starter: tokenized-AR sweep on PACE, so "tokens lose" can't be blamed on one bad bin choice.
#   n_bins in {64, 256, 1024, 4096} x scheme in {quantile, uniform}  -> 8 training jobs
#   then each checkpoint is exported with 3 decode modes (mean, center, dequant) and scored on the shared harness.
# Prereq (once): python -m bench.trackA export --out bench_data/A
#   bash scripts/a2_token_sweep.sh            # submit
#   bash scripts/a2_token_sweep.sh --score    # after jobs finish: export samples + score + table
set -e
cd "$(dirname "$0")/.."
mkdir -p logs
COMMON="--n_train 20000 --n_val 2000 --n_eval 32 --steps 20000 --batch 256 --d 128 --n_layers 4"
BINS="64 256 1024 4096"
SCHEMES="quantile uniform"

if [ "$1" != "--score" ]; then
  for nb in $BINS; do for sc in $SCHEMES; do
    sbatch scripts/pace_ardiff_train.sbatch --head token --n_bins $nb --bin_scheme $sc --bin_clip 8 \
      --out runs/a2_tok${nb}${sc:0:1} $COMMON
  done; done
  echo "submitted 8 jobs; when done: bash scripts/a2_token_sweep.sh --score (run inside a GPU salloc)"
  exit 0
fi

for nb in $BINS; do for sc in $SCHEMES; do
  ck=runs/a2_tok${nb}${sc:0:1}/ckpt.pkl
  [ -f $ck ] || { echo "missing $ck"; continue; }
  for dec in mean center dequant; do
    out=runs/a2_tok${nb}${sc:0:1}_$dec
    python -m ardiff.export_samples --ckpt $ck --data bench_data/A --token_decode $dec --out $out --n_paths 8192
    python -m bench.trackA score --data bench_data/A --samples $out/samples.npy
  done
done; done
python -m bench.trackA table "results/trackA/*.json"
