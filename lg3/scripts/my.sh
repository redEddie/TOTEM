set -e

SOURCES="elec1_f2,ohsung_f2,snu"
DATA_ROOT="data"
PROCESSED_ROOT="lg3/data/processed_sources"
REVIN_ROOT="lg3/data/revin_sources"
COMBINED_BASE="lg3/data/combined"
COMBINED_REVIN="${COMBINED_BASE}/revin"
PROCESSED_COMBINED="lg3/data/processed_combined"
EXOG_ROOT="lg3/data/exog_sources"

# EREPORT_COLS="MFR_068,Comp1 Hz_1,Comp1 Hz_0,Power,VAP_Entha,LIQ_Entha,Tcond"
EREPORT_COLS="Power"
SMARTCARE_COLS="Tod"
FREQ="5min"
EXCLUDE_FROM_MONTH=10

SEQ_LEN=288
PRED_LEN=288
BATCH_SIZE=2048
GPU=1
LRS="0.0001,0.001"

VQVAE_CONFIG="lg3/scripts/lg3.json"
VQVAE_SAVE="lg3/saved_models/"

COMPRESSION=4
FORECAST_SAVE="lg3/data/forecasting/Tin${SEQ_LEN}_Tout${PRED_LEN}"
TRAINED_VQVAE_PATH="lg3/saved_models/CD64_CW256_CF4_BS2048_ITR15000/checkpoints/final_model.pth"

IFS=',' read -r -a SOURCE_ARR <<< "$SOURCES"

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
  --exog_dir "${EXOG_ROOT}" \
  --save_path "${FORECAST_SAVE}/" \
  --seq_len ${SEQ_LEN} \
  --pred_len ${PRED_LEN} \
  --enc_in -1 \
  --gpu ${GPU} \
  --batch_size 1024 \
  --compression_factor ${COMPRESSION} \
  --trained_vqvae_model_path "${TRAINED_VQVAE_PATH}"

# 7) train forecaster (multiple learning rates)
IFS=',' read -r -a LR_ARR <<< "$LRS"
for lr in "${LR_ARR[@]}"; do
  PYTHONPATH=. python -m lg3.train_forecaster \
    --data-type lg3 \
    --Tin ${SEQ_LEN} \
    --Tout ${PRED_LEN} \
    --compression ${COMPRESSION} \
    --cuda-id ${GPU} \
    --seed 2021 \
    --data_path "${FORECAST_SAVE}" \
    --codebook_size 256 \
    --checkpoint \
    --checkpoint_path "lg3/saved_models/lg3/forecaster_checkpoints/lg3_Tin${SEQ_LEN}_Tout${PRED_LEN}_seed2021_lr${lr}" \
    --file_save_path "lg3/results/" \
    --patience 5 \
    --d-model 128 \
    --d_hid 512 \
    --nlayers 8 \
    --nhead 8 \
    --baselr ${lr} \
    --batchsize 128
done

# 8) eval
PYTHONPATH=. python -m lg3.plot_forecaster_overlap \
  --data_dir lg3/data/forecasting/Tin288_Tout288 \
  --feature_names_path lg3/data/forecasting/Tin288_Tout288/feature_names.json \
  --checkpoint_dir lg3/saved_models/lg3/forecaster_checkpoints/lg3_Tin288_Tout288_seed2021 \
  --feature Power \
  --plot_mode pred_only \
  --save_each \
  --output_dir lg3/results/non_overlap
