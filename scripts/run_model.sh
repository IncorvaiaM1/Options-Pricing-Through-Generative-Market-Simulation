#!/bin/bash
# Train/generate one Track A model on one data folder, score it, and draw its path plot.
#
#   bash scripts/run_model.sh <model> <data_dir> [extra args passed to the model]
#
# models:  bsm | bootstrap | garch                      classical baselines (seconds, CPU)
#          ar_diffusion | ar_token                      A1 / A2 (ours + token baseline, JAX)
#          rnade                                        A3 (PyTorch, team's RNADE)
#          gan | sde_gan | pure_diffusion | path_flow | path_flow_t     A4 A5 A6 A7 (JAX)
# data_dir: a synthetic export (bench.trackA export) or a real one (bench.real export).
# Optional env:  MARKET=bench_data/market/2026-10-27_megacap_tech   also price that market snapshot (bench.market compare)
#                SEED=1   pass --seed 1 and keep this run apart: runs/<data>/<model>_s1, label <label>_s1, log <model>_s1.log
#                N_PATHS=8192   QUICK=1 (tiny settings for a plumbing check)
#
# Outputs: runs/<data name>/<model>/samples.npy, results/<data name>/{trackA|backtest|market}/<label>.json,
#          figs/<data name>/paths_<label>.png
set -e
cd "$(dirname "$0")/.."
MODEL=$1; DATA=$2; shift 2 || { echo "usage: bash scripts/run_model.sh <model> <data_dir> [args]"; exit 1; }
NAME=$(basename "$DATA")
SFX=""; if [ -n "$SEED" ]; then SFX=_s$SEED; set -- "$@" --seed "$SEED"; fi
OUT=runs/$NAME/$MODEL$SFX
N=${N_PATHS:-8192}
KIND=$(python -c "import json;print(json.load(open('$DATA/config.json')).get('kind','synthetic'))")
ALSO=""; if [ -n "$MARKET" ]; then ALSO="--also_eval $MARKET"; fi
Q=""
if [ -n "$QUICK" ]; then N=256; fi

# Full stdout/stderr of this model always lands in logs/models/<data name>/<model>.log (overwritten by the next run of
# the same model; python -m bench.archive copies it into archive/<RUN_TAG>/ so nothing is lost).
mkdir -p logs/models/$NAME
LOG=logs/models/$NAME/$MODEL$SFX.log
exec > >(tee "$LOG") 2>&1
echo "# run_model.sh $MODEL $DATA $*"
echo "# start $(date -Is)  host $(hostname)  git $(git rev-parse --short HEAD 2>/dev/null)$(git diff --quiet HEAD 2>/dev/null || echo ' (dirty)')  RUN_TAG=${RUN_TAG:-none}"
echo "# env QUICK=${QUICK:-} N_PATHS=$N MARKET=${MARKET:-}  data $(python -c "import json;c=json.load(open('$DATA/config.json'));print({k:c[k] for k in ('kind','n_train','n_eval','n_hedge_paths','tickers') if k in c})" 2>/dev/null)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | sed 's/^/# gpu /' || true
trap 'rc=$?; echo "# end $(date -Is) exit $rc elapsed ${SECONDS}s"' EXIT
# stale outputs of this model would otherwise end up in the report figures if this run fails
rm -f $OUT/samples.npy $OUT/meta.json

case $MODEL in
  bsm|bootstrap|garch)
    python -m starters.a0_baselines --method $MODEL --data $DATA --out $OUT --n_paths $N "$@"
    if [ -n "$MARKET" ]; then
      python -m starters.a0_baselines --method $MODEL --data $MARKET --out $OUT/also_$(basename $MARKET) --n_paths $N "$@"
    fi ;;
  ar_diffusion|ar_token)
    HEAD=diffusion; if [ $MODEL = ar_token ]; then HEAD=token; fi
    if [ -n "$QUICK" ]; then Q="--steps 200 --d 32 --n_layers 1 --batch 64 --log_every 100"; fi
    python -m ardiff.train --head $HEAD --data $DATA --out $OUT $Q "$@"
    python -m ardiff.export_samples --ckpt $OUT/ckpt.pkl --data $DATA --out $OUT --n_paths $N
    if [ -n "$MARKET" ]; then
      python -m ardiff.export_samples --ckpt $OUT/ckpt.pkl --data $MARKET --out $OUT/also_$(basename $MARKET) --n_paths $N
    fi ;;
  rnade)
    if [ -n "$QUICK" ]; then Q="--steps 200 --hidden 16 --log_every 100"; fi
    python -m starters.a3_rnade --data $DATA --out $OUT --n_paths $N $ALSO $Q "$@" ;;
  gan)
    if [ -n "$QUICK" ]; then Q="--steps 100 --width 64 --batch 64 --log_every 50"; fi
    python -m starters.a4_gan --data $DATA --out $OUT --n_paths $N $ALSO $Q "$@" ;;
  sde_gan)
    if [ -n "$QUICK" ]; then Q="--steps 50 --dh 8 --dw 4 --width 32 --substeps 1 --batch 64 --log_every 25"; fi
    python -m starters.a5_sde_gan --data $DATA --out $OUT --n_paths $N $ALSO $Q "$@" ;;
  pure_diffusion)
    if [ -n "$QUICK" ]; then Q="--steps 300 --width 64 --depth 1 --T 100 --log_every 150"; fi
    python -m starters.a6_pure_diffusion --data $DATA --out $OUT --n_paths $N $ALSO $Q "$@" ;;
  path_flow|path_flow_t)
    B=gauss; if [ $MODEL = path_flow_t ]; then B=t; fi
    if [ -n "$QUICK" ]; then Q="--steps 300 --n_layers 4 --width 64 --log_every 150"; fi
    python -m starters.a7_path_flow --data $DATA --out $OUT --base $B --n_paths $N $ALSO $Q "$@" ;;
  *) echo "unknown model $MODEL"; exit 1 ;;
esac

if [ -n "$SFX" ]; then  # seed suffix on the label, so scores and figures of different seeds don't overwrite each other
  for mj in $OUT/meta.json $OUT/also_*/meta.json; do
    [ -f "$mj" ] && python -c "import json,sys;p,s=sys.argv[1:];m=json.load(open(p));l=m.get('label','$MODEL');m['label']=l if l.endswith(s) else l+s;json.dump(m,open(p,'w'),indent=1)" "$mj" "$SFX"
  done
fi
if [ "$KIND" = "real" ]; then
  python -m bench.backtest --data $DATA --samples $OUT/samples.npy --out results/$NAME/backtest
else
  python -m bench.trackA score --data $DATA --samples $OUT/samples.npy --out results/$NAME/trackA
fi
if [ -n "$MARKET" ]; then
  python -m bench.market compare --market $MARKET --samples $OUT/also_$(basename $MARKET)/samples.npy --out results/$NAME/market
fi
python -m bench.plots paths --data $DATA --samples $OUT/samples.npy --out figs/$NAME
echo "done: $MODEL on $NAME"
