#!/bin/bash
# Energy-regression baseline (user request 2026-10-25 01:00): the paper's Region + MLP backbone trained from scratch
# under the frozen regression recipe (validation_regression, --mode scratch), 1 %, 10 %, 100 % labels, MJD / EXO-200 / SuperNEMO.
cd /home/tidmad/foundation_model; source research/paper_baseline/env.sh
export NLDBD_FINETUNE_RAM_GB=12 NLDBD_REGRESSION_RUNS=/home/tidmad/foundation_model/runs_regression
mkdir -p runs_regression/logs
for lf in 0.01 0.1 1.0; do for ds in mjd exo200 supernemo; do
  hrs=3; [ "$lf" = "1.0" ] && hrs=10
  if [ -f runs_regression/paper_region_mlp__scratch/$ds/L$lf/s42/metrics.json ]; then echo "$ds L$lf already done"; continue; fi
  t0=$(date +%s); log=runs_regression/logs/paper_region_mlp.scratch.$ds.L$lf.log
  $PY -B -m validation_regression.finetune --fm-module research/paper_baseline/region_mlp_encoder.py --name paper_region_mlp \
      --dataset $ds --label-fraction $lf --mode scratch --max-hours $hrs > $log 2>&1
  echo "$(date +%H:%M) $ds L$lf exit $? in $(( ($(date +%s) - t0) / 60 )) min: $(tail -1 $log)"
done; done
