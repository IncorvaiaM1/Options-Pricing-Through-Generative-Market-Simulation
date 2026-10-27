#!/bin/bash
# Submit every Track A model for one data folder as separate PACE jobs, then a report job that waits for all.
#   bash scripts/submit_all.sh bench_data/A
#   MARKET=bench_data/market/2026-10-27 bash scripts/submit_all.sh bench_data/real/megacap_tech
#   MODELS="pure_diffusion path_flow" bash scripts/submit_all.sh bench_data/A     # a subset
set -e
cd "$(dirname "$0")/.."
DATA=$1; NAME=$(basename "$DATA")
MODELS=${MODELS:-"bsm bootstrap garch ar_diffusion ar_token rnade gan sde_gan pure_diffusion path_flow path_flow_t"}
mkdir -p logs
ids=""
for m in $MODELS; do
  id=$(sbatch --parsable --export=ALL -J ${m}_${NAME} scripts/pace_model.sbatch $m $DATA)
  echo "$m -> job $id"; ids="$ids:$id"
done
rid=$(sbatch --parsable --dependency=afterany$ids -J report_${NAME} -t 00:30:00 \
      --wrap "module load anaconda3; conda activate gdl; bash scripts/report.sh $DATA" -o logs/report_${NAME}-%j.out)
echo "report job $rid (runs when all finish): figs/$NAME/"
