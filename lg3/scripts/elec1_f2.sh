set -e

DATA_ROOT="data/elec1_f2"
PROCESSED_DIR="lg3/data/processed"
REVIN_DIR="lg3/data/revin"

EREPORT_COLS="MFR_068,Comp1 Hz_1,Comp1 Hz_0,Power,VAP_Entha,LIQ_Entha,Tcond"
SMARTCARE_COLS="Tod"
FREQ="5min"
EXCLUDE_FROM_MONTH=10

SEQ_LEN=288
PRED_LEN=288
BATCH_SIZE=2048
GPU=0

VQVAE_CONFIG="lg3/scripts/lg3.json"
VQVAE_SAVE="lg3/saved_models/"

COMPRESSION=16
FORECAST_SAVE="lg3/data/forecasting/Tin${SEQ_LEN}_Tout${PRED_LEN}"
TRAINED_VQVAE_PATH="lg3/saved_models/CD64_CW256_CF16_BS2048_ITR15000/checkpoints/final_model.pth"

python lg3/prepare_lg3_data.py \
  --ereport_dir "${DATA_ROOT}/EREPORT" \
  --smartcare_dir "${DATA_ROOT}/SMARTCARE" \
  --freq "${FREQ}" \
  --output_dir "${PROCESSED_DIR}" \
  --train_ratio 0.7 \
  --val_ratio 0.1 \
  --ereport_cols "${EREPORT_COLS}" \
  --smartcare_process_cols "${SMARTCARE_COLS}" \
  --exclude_from_month ${EXCLUDE_FROM_MONTH}

PYTHONPATH=. python -m lg3.save_revin_data \
  --input_dir "${PROCESSED_DIR}" \
  --output_dir "${REVIN_DIR}" \
  --seq_len ${SEQ_LEN} \
  --pred_len ${PRED_LEN} \
  --batch_size ${BATCH_SIZE} \
  --gpu ${GPU}

PYTHONPATH=. python -m lg3.train_vqvae \
  --config_path "${VQVAE_CONFIG}" \
  --model_init_num_gpus ${GPU} \
  --data_init_cpu_or_gpu cpu \
  --save_path "${VQVAE_SAVE}" \
  --base_path "lg3/data" \
  --batchsize ${BATCH_SIZE}

PYTHONPATH=. python -m lg3.extract_forecasting_data \
  --input_dir "${PROCESSED_DIR}" \
  --save_path "${FORECAST_SAVE}/" \
  --seq_len ${SEQ_LEN} \
  --pred_len ${PRED_LEN} \
  --enc_in -1 \
  --gpu ${GPU} \
  --batch_size 256 \
  --compression_factor ${COMPRESSION} \
  --trained_vqvae_model_path "${TRAINED_VQVAE_PATH}" \
  --norm_stats "${REVIN_DIR}/norm_stats.json"

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
  --checkpoint_path "lg3/saved_models/lg3/forecaster_checkpoints/lg3_Tin${SEQ_LEN}_Tout${PRED_LEN}_seed2021" \
  --file_save_path "lg3/results/" \
  --patience 20 \
  --d-model 128 \
  --d_hid 512 \
  --nlayers 8 \
  --nhead 8 \
  --baselr 0.0005 \
  --batchsize 64

bash lg3/scripts/eval_forecaster_metrics.sh
