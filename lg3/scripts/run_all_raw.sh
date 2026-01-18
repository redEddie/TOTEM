set -e

# bash lg3/scripts/prepare_lg3_data.sh
bash lg3/scripts/extract_forecasting_data_raw.sh
bash lg3/scripts/train_forecaster_raw.sh
bash lg3/scripts/eval_forecaster_metrics_raw.sh
