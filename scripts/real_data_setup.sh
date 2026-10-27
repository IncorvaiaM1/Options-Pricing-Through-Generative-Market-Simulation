#!/bin/bash
# Real data setup. Run on a machine WITH internet (PACE login node is fine; compute nodes may not have it).
#   bash scripts/real_data_setup.sh megacap_tech          # a preset cluster (see bench/real.py PRESETS)
#   TICKERS="AAPL" bash scripts/real_data_setup.sh aapl   # a single stock (or any custom list)
# It prints the exact submit command at the end (copy it; don't type <date> placeholders, bash reads < as a redirect).
set -e
cd "$(dirname "$0")/.."
NAME=$1
if [ -n "$TICKERS" ]; then
  python -m bench.real download --tickers $TICKERS --start ${START:-2010-01-01} --out bench_data/real_raw/$NAME
else
  python -m bench.real download --preset $NAME --start ${START:-2010-01-01} --out bench_data/real_raw/$NAME
fi
python -m bench.real export --raw bench_data/real_raw/$NAME --out bench_data/real/$NAME \
  --train_end ${TRAIN_END:-2021-12-31} --val_end ${VAL_END:-2023-06-30} --eval_stride ${EVAL_STRIDE:-5}
python -m bench.market snapshot --export bench_data/real/$NAME --out bench_data/market/$(date +%F)_$NAME
SNAP=bench_data/market/$(date +%F)_$NAME
echo "ready: data bench_data/real/$NAME, market snapshot $SNAP"
echo "next:  MARKET=$SNAP bash scripts/submit_all.sh bench_data/real/$NAME"
