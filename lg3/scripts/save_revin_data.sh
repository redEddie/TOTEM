PYTHONPATH=. python -m lg3.save_revin_data \
  --input_dir "lg3/data/processed" \
  --output_dir "lg3/data/revin" \
  --seq_len 96 \
  --pred_len 96 \
  --batch_size 2048 \
  --gpu 0
