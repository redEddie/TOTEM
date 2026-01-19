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

VQVAE_CONFIG="lg3/scripts/lg3.json"
VQVAE_SAVE="lg3/saved_models/"

COMPRESSION=4
FORECAST_SAVE="lg3/data/forecasting/Tin${SEQ_LEN}_Tout${PRED_LEN}"
TRAINED_VQVAE_PATH="lg3/saved_models/CD64_CW256_CF4_BS2048_ITR15000/checkpoints/final_model.pth"

IFS=',' read -r -a SOURCE_ARR <<< "$SOURCES"


# 7) train forecaster
# PYTHONPATH=. python -m lg3.train_forecaster \
#   --data-type lg3 \
#   --Tin ${SEQ_LEN} \
#   --Tout ${PRED_LEN} \
#   --compression ${COMPRESSION} \
#   --cuda-id ${GPU} \
#   --seed 2021 \
#   --data_path "${FORECAST_SAVE}" \
#   --codebook_size 256 \
#   --checkpoint \
#   --checkpoint_path "lg3/saved_models/lg3/forecaster_checkpoints/lg3_Tin${SEQ_LEN}_Tout${PRED_LEN}_seed2021" \
#   --file_save_path "lg3/results/" \
#   --patience 5 \
#   --d-model 128 \
#   --d_hid 512 \
#   --nlayers 8 \
#   --nhead 8 \
#   --baselr 0.001 \
#   --batchsize 128

# 8) eval
PYTHONPATH=. python -m lg3.plot_forecaster_overlap \
  --data_dir lg3/data/forecasting/Tin288_Tout288 \
  --feature_names_path lg3/data/forecasting/Tin288_Tout288/feature_names.json \
  --checkpoint_dir lg3/saved_models/lg3/forecaster_checkpoints/lg3_Tin288_Tout288_seed2021 \
  --feature Power \
  --plot_mode pred_only \
  --save_each \
  --output_dir lg3/results/non_overlap