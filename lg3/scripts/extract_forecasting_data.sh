PYTHONPATH=. python -m lg3.extract_forecasting_data \
  --input_dir "lg3/data/processed" \
  --save_path "lg3/data/forecasting/Tin192_Tout192/" \
  --seq_len 192 \
  --pred_len 192 \
  --enc_in -1 \
  --gpu 0 \
  --batch_size 256 \
  --compression_factor 4 \
  --trained_vqvae_model_path "lg3/saved_models/CD64_CW256_CF4_BS2048_ITR15000/checkpoints/final_model.pth"
