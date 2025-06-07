# 1 GPU
# conda activate flow_grpo
# accelerate launch --config_file scripts/accelerate_configs/multi_gpu.yaml --num_processes=1 --main_process_port 29503 scripts/train_sd3.py --config config/dgx.py:geneval_sd3_debug
# accelerate launch --config_file scripts/accelerate_configs/multi_gpu.yaml --num_processes=1 --main_process_port 29503 scripts/train_sd3.py --config config/dgx.py:geneval_sd3_counting
# 4 GPU
# accelerate launch --config_file scripts/accelerate_configs/multi_gpu.yaml --num_processes=4 --main_process_port 29503 scripts/train_sd3.py --config config/dgx.py:geneval_sd3_counting

accelerate launch --config_file scripts/accelerate_configs/multi_gpu.yaml --num_processes=4 --main_process_port 29503 scripts/train_sd3.py --config config/count_10_aba.py:geneval_sd3_counting_10_step20_init_same_noise

# gunicorn "app_geneval:create_app()"
