TIN=288
TOUT=288
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
