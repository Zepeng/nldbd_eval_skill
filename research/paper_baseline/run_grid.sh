#!/bin/bash
# Paper Region + MLP Transformer baseline (user request 2026-10-24 23:00): 1 %, 10 %, 100 % labels on all four
# detectors, under the evaluation skill. Usage: run_grid.sh "<ds>:<fraction> ..."  (one lane, sequential)
cd /home/tidmad/foundation_model; source research/paper_baseline/env.sh
export NLDBD_FINETUNE_RAM_GB=16
mkdir -p runs/paper_region_mlp/logs
for job in $1; do
  ds=${job%%:*}; lf=${job##*:}; hrs=8; [ "$lf" = "1.0" ] && hrs=14
  if [ -f runs/paper_region_mlp/$ds/L$lf/s42/metrics.json ]; then echo "$ds L$lf already done"; continue; fi
  t0=$(date +%s); log=runs/paper_region_mlp/logs/$ds.L$lf.log
  $PY -B research/paper_baseline/train_supervised.py --model-module research/paper_baseline/region_mlp.py \
      --dataset $ds --label-fraction $lf --max-hours $hrs > $log 2>&1
  echo "$(date +%H:%M) $ds L$lf exit $? in $(( ($(date +%s) - t0) / 60 )) min: $(grep '"mode": "full"' $log | tail -1)"
done
