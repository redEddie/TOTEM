set -e

bash lg3/scripts/prepare_lg3_data.sh
# bash lg3/scripts/save_revin_data.sh
# bash lg3/scripts/train_vqvae.sh
bash lg3/scripts/extract_forecasting_data.sh
bash lg3/scripts/train_forecaster.sh
bash lg3/scripts/eval_forecaster_metrics.sh
