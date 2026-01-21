set -e

SOURCES="elec1_f2,ohsung_f2"
DATA_ROOT="data"
PROCESSED_ROOT="lg3/data/processed_sources"
REVIN_ROOT="lg3/data/revin_sources"
COMBINED_BASE="lg3/data/combined"
COMBINED_REVIN="${COMBINED_BASE}/revin"
PROCESSED_COMBINED="lg3/data/processed_combined"

EREPORT_COLS="Power"
SMARTCARE_COLS=""
FREQ="30min"
EXCLUDE_FROM_MONTH=10
DROP_ZERO_RATIO=0.9

SEQ_LEN=48
PRED_LEN=48
TIN=48  # SEQ LEN이랑 같은 값
TOUT=48 # PRED LEN이랑 같은 값
BATCH_SIZE=2048
GPU=1

VQVAE_CONFIG="lg3/scripts/lg3.json"
VQVAE_SAVE="lg3/saved_models/"

COMPRESSION=16
FORECAST_SAVE="lg3/data/forecasting/Tin${SEQ_LEN}_Tout${PRED_LEN}"
TRAINED_VQVAE_PATH="lg3/saved_models/CD64_CW256_CF16_BS2048_ITR5000/checkpoints/final_model.pth"

IFS=',' read -r -a SOURCE_ARR <<< "$SOURCES"

# 1) per-source prepare
for SOURCE in "${SOURCE_ARR[@]}"; do
  python lg3/prepare_lg3_data.py \
    --ereport_dir "${DATA_ROOT}/${SOURCE}/EREPORT" \
    --smartcare_dir "${DATA_ROOT}/${SOURCE}/SMARTCARE" \
    --freq "${FREQ}" \
    --output_dir "${PROCESSED_ROOT}/${SOURCE}" \
    --train_ratio 0.7 \
    --val_ratio 0.1 \
  --ereport_cols "${EREPORT_COLS}" \
  --smartcare_process_cols "${SMARTCARE_COLS}" \
  --exclude_from_month ${EXCLUDE_FROM_MONTH} \
  --drop_zero_ratio_threshold ${DROP_ZERO_RATIO}
 done

# 2) per-source revin
for SOURCE in "${SOURCE_ARR[@]}"; do
  PYTHONPATH=. python -m lg3.save_revin_data \
    --input_dir "${PROCESSED_ROOT}/${SOURCE}" \
    --output_dir "${REVIN_ROOT}/${SOURCE}" \
    --seq_len ${SEQ_LEN} \
    --pred_len ${PRED_LEN} \
    --batch_size ${BATCH_SIZE} \
    --gpu ${GPU}
 done

# 3) combine revin for VQ-VAE
python - <<PY
import os
import shutil
import numpy as np

sources = "${SOURCES}".split(",")
revin_root = "${REVIN_ROOT}"
combined_revin = "${COMBINED_REVIN}"

if os.path.isdir(combined_revin):
    shutil.rmtree(combined_revin)
os.makedirs(combined_revin, exist_ok=True)

for split in ["train", "val", "test"]:
    xs, ys = [], []
    for source in sources:
        src_dir = os.path.join(revin_root, source)
        xs.append(np.load(os.path.join(src_dir, f"{split}_data_x.npy")))
        ys.append(np.load(os.path.join(src_dir, f"{split}_data_y.npy")))
    x = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0)
    np.save(os.path.join(combined_revin, f"{split}_data_x.npy"), x)
    np.save(os.path.join(combined_revin, f"{split}_data_y.npy"), y)
print("[DONE] combined revin ->", combined_revin)
PY

# 4) train VQ-VAE (uses combined revin)
PYTHONPATH=. python -m lg3.train_vqvae \
  --config_path "${VQVAE_CONFIG}" \
  --model_init_num_gpus ${GPU} \
  --data_init_cpu_or_gpu cpu \
  --save_path "${VQVAE_SAVE}" \
  --base_path "${COMBINED_BASE}" \
  --batchsize ${BATCH_SIZE}

# 5) combine per-source splits for forecaster input
python - <<PY
import os
import shutil
import pandas as pd

sources = "${SOURCES}".split(",")
processed_root = "${PROCESSED_ROOT}"
output_dir = "${PROCESSED_COMBINED}"

if os.path.isdir(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(os.path.join(output_dir, "smartcare_units"), exist_ok=True)

for source in sources:
    unit_dir = os.path.join(output_dir, "smartcare_units", f"unit_{source}")
    os.makedirs(unit_dir, exist_ok=True)
    for split in ["train", "val", "test"]:
        src = os.path.join(processed_root, source, f"lg3_{split}.csv")
        df = pd.read_csv(src, index_col=0, parse_dates=True)
        df.to_csv(os.path.join(unit_dir, f"lg3_{split}.csv"))
print("[DONE] combined forecaster splits ->", output_dir)
PY

# 6) extract forecasting data
PYTHONPATH=. python -m lg3.extract_forecasting_data \
  --input_dir "${PROCESSED_COMBINED}" \
  --save_path "${FORECAST_SAVE}/" \
  --seq_len ${SEQ_LEN} \
  --pred_len ${PRED_LEN} \
  --enc_in -1 \
  --gpu ${GPU} \
  --batch_size 1024 \
  --compression_factor ${COMPRESSION} \
  --trained_vqvae_model_path "${TRAINED_VQVAE_PATH}"

# 7) train forecaster
PYTHONPATH=. python -m lg3.train_forecaster \
  --data-type lg3 \
  --Tin ${SEQ_LEN} \
  --Tout ${PRED_LEN} \
  --compression ${COMPRESSION} \
  --cuda-id ${GPU} \
  --seed 2021 \
  --data_path "${FORECAST_SAVE}" \
  --codebook_size 128 \
  --checkpoint \
  --checkpoint_path "lg3/saved_models/lg3/forecaster_checkpoints/lg3_Tin${SEQ_LEN}_Tout${PRED_LEN}_seed2021" \
  --file_save_path "lg3/results/" \
  --patience 20 \
  --d-model 128 \
  --d_hid 512 \
  --nlayers 8 \
  --nhead 8 \
  --baselr 0.0005 \
  --batchsize 64

# 8) eval
DATA_DIR="lg3/data/forecasting/Tin${TIN}_Tout${TOUT}"
CHECKPOINT_DIR="lg3/saved_models/lg3/forecaster_checkpoints/lg3_Tin${TIN}_Tout${TOUT}_seed2021"
FEATURE_NAMES_PATH="${DATA_DIR}/feature_names.json"
FEATURE_NAME="Power"

PYTHONPATH=. python -m lg3.eval_forecaster_metrics \
  --data_dir "${DATA_DIR}" \
  --checkpoint_dir "${CHECKPOINT_DIR}" \
  --batch_size 32 \
  --cuda_id 0 \
  --scheme 1 \
  --feature_names_path "${FEATURE_NAMES_PATH}" \
  --output_csv "lg3/results/forecast_metrics.csv" \
  --output_seq_mse_csv "lg3/results/forecast_seq_mse.csv" \
  --output_seq_mae_csv "lg3/results/forecast_seq_mae.csv" \
  --output_seq_mape_csv "lg3/results/forecast_seq_mape.csv" \
  --output_seq_mase_csv "lg3/results/forecast_seq_mase.csv"

PYTHONPATH=. python -m lg3.plot_forecaster_overlap \
  --data_dir "${DATA_DIR}" \
  --feature_names_path "${FEATURE_NAMES_PATH}" \
  --checkpoint_dir "${CHECKPOINT_DIR}" \
  --feature "${FEATURE_NAME}" \
  --plot_mode pred_only \
  --start_idx 0 \
  --num_sequences 200 \
  --batch_size 32 \
  --cuda_id 0 \
  --scheme 1 \
  --output "lg3/results/forecast_overlap_${FEATURE_NAME}.png"

PYTHONPATH=. python -m lg3.plot_forecaster_overlap \
  --data_dir "${DATA_DIR}" \
  --feature_names_path "${FEATURE_NAMES_PATH}" \
  --checkpoint_dir "${CHECKPOINT_DIR}" \
  --feature "${FEATURE_NAME}" \
  --plot_mode pred_only \
  --start_idx 0 \
  --num_sequences 50 \
  --batch_size 16 \
  --cuda_id 0 \
  --scheme 1 \
  --save_each \
  --output_dir "lg3/results/non_overlap"
