#!/bin/bash
accelerate launch --config_file scripts/accelerate_configs/multi_gpu.yaml --num_processes=4 --main_process_port 29503 scripts/train_sd3.py --config config/dgx.py:geneval_sd3_counting

# cold_start from 3k
accelerate launch --config_file scripts/accelerate_configs/multi_gpu.yaml --num_processes=4 --main_process_port 29503 scripts/train_sd3.py --config config/count_10_aba.py:geneval_sd3_counting_10_step20_cold_start_from_3k
