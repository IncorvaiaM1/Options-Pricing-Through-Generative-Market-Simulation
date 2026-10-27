#!/bin/bash
# Leaderboard tables + all comparison figures for one data folder, after models have run.
#   bash scripts/report.sh bench_data/A
#   bash scripts/report.sh bench_data/real/megacap_tech
set -e
cd "$(dirname "$0")/.."
DATA=$1; NAME=$(basename "$DATA")
KIND=$(python -c "import json;print(json.load(open('$DATA/config.json')).get('kind','synthetic'))")
SAMPLES=$(ls runs/$NAME/*/samples.npy 2>/dev/null | tr '\n' ' ')
mkdir -p figs/$NAME
if [ "$KIND" = "real" ]; then
  python -m bench.backtest table "results/$NAME/backtest/*.json" | tee figs/$NAME/backtest_table.md
  python -m bench.plots backtest --results "results/$NAME/backtest/*.json" --out figs/$NAME
  ls results/$NAME/market/*.json >/dev/null 2>&1 && python -m bench.plots market --results "results/$NAME/market/*.json" --out figs/$NAME
else
  python -m bench.trackA table "results/$NAME/trackA/*.json" | tee figs/$NAME/trackA_table.md
  python -m bench.plots leaderboard --results "results/$NAME/trackA/*.json" --out figs/$NAME
  python -m bench.plots smile --data $DATA --samples $SAMPLES --out figs/$NAME
fi
python -m bench.plots terminal --data $DATA --samples $SAMPLES --out figs/$NAME
echo "figures and tables in figs/$NAME"
