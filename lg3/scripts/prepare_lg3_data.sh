EREPORT_COLS="Power"
SMARTCARE_COLS="Tod"

python lg3/prepare_lg3_data.py \
  --freq 1H \
  --output_dir lg3/data/processed \
  --train_ratio 0.7 \
  --val_ratio 0.1 \
  --ereport_cols "${EREPORT_COLS}" \
  --smartcare_process_cols "${SMARTCARE_COLS}"
