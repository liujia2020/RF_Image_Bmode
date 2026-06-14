#!/usr/bin/env bash
set -euo pipefail
cd /mnt/g/Code/RF_Image_Bmode
/home/liujia/miniconda3/bin/conda run -n rf-bmode \
  python -m nbconvert \
  --to notebook \
  --execute \
  --inplace \
  experiments/dual_input_v1/2026-06-13_dualbranch_baseline_50ep/02_train/train.ipynb \
  --ExecutePreprocessor.timeout=-1 \
  > experiments/dual_input_v1/2026-06-13_dualbranch_baseline_50ep/02_train/full_run_nbconvert.log \
  2>&1
