PYTHONPATH=. python -m lg3.train_vqvae \
  --config_path lg3/scripts/lg3.json \
  --model_init_num_gpus 0 \
  --data_init_cpu_or_gpu cpu \
  --save_path "lg3/saved_models/" \
  --base_path "lg3/data" \
  --batchsize 2048
