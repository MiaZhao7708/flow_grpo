#! /bin/bash

# NAME="sd-3.5-m-base"
# STEP=316
# STEP=596

cleanup() {
    echo "Cleaning up..."
    pkill -p $$
    exit 1
}

trap cleanup SIGINT

# CUDA_VISIBLE_DEVICES=0 python generation/diffusers_generate.py --outdir $NAME &
# CUDA_VISIBLE_DEVICES=1 python generation/diffusers_generate.py --outdir $NAME &
# CUDA_VISIBLE_DEVICES=2 python generation/diffusers_generate.py --outdir $NAME &
# CUDA_VISIBLE_DEVICES=3 python generation/diffusers_generate.py --outdir $NAME &


# python generation/diffusers_generate.py \
#     --metadata_file "/openseg_blob/zhaoyaqi/flow_grpo/dataset/counting/test_metadata.jsonl" \
#     --outdir "sd-3.5-m-base-coco80-8k"
#     --lora_step '/openseg_blob/zhaoyaqi/workspace/coco80_grpo_counting_sd3_5_medium/output/sd_3_5_medium_base_data_8k_coco80/checkpoint-1000'

# CUDA_VISIBLE_DEVICES=0 python generation/diffusers_generate.py --lora_step '/openseg_blob/zhaoyaqi/workspace/coco80_grpo_counting_sd3_5_medium/output/sd_3_5_medium_base_data_8k_coco80_v5/checkpoint-1000' --outdir "sd-3.5-m-base-coco80-8k-v5-ft-1k-denoise-step10" --steps 10 --n_samples 4 --metadata_file "/openseg_blob/zhaoyaqi/flow_grpo/dataset/counting/test_metadata.jsonl" &
# CUDA_VISIBLE_DEVICES=1 python generation/diffusers_generate.py --lora_step '/openseg_blob/zhaoyaqi/workspace/coco80_grpo_counting_sd3_5_medium/output/sd_3_5_medium_base_data_8k_coco80_v5/checkpoint-1000' --outdir "sd-3.5-m-base-coco80-8k-v5-ft-1k-denoise-step15" --steps 15 --n_samples 4 --metadata_file "/openseg_blob/zhaoyaqi/flow_grpo/dataset/counting/test_metadata.jsonl" &
# CUDA_VISIBLE_DEVICES=2 python generation/diffusers_generate.py --lora_step '/openseg_blob/zhaoyaqi/workspace/coco80_grpo_counting_sd3_5_medium/output/sd_3_5_medium_base_data_8k_coco80_v5/checkpoint-1000' --outdir "sd-3.5-m-base-coco80-8k-v5-ft-1k-denoise-step20" --steps 20 --n_samples 4 --metadata_file "/openseg_blob/zhaoyaqi/flow_grpo/dataset/counting/test_metadata.jsonl" &
# CUDA_VISIBLE_DEVICES=3 python generation/diffusers_generate.py --lora_step '/openseg_blob/zhaoyaqi/workspace/coco80_grpo_counting_sd3_5_medium/output/sd_3_5_medium_base_data_8k_coco80_v5/checkpoint-1000' --outdir "sd-3.5-m-base-coco80-8k-v5-ft-1k-denoise-step40" --steps 40 --n_samples 4 --metadata_file "/openseg_blob/zhaoyaqi/flow_grpo/dataset/counting/test_metadata.jsonl" &


CUDA_VISIBLE_DEVICES=0 python generation/diffusers_generate.py --lora_step '/openseg_blob/zhaoyaqi/workspace/coco80_grpo_counting_sd3_5_medium/output/sd_3_5_medium_base_data_8k_coco80_v5/checkpoint-2000' --outdir "sd-3.5-m-base-coco80-8k-v5-ft-step2k" --n_samples 4 &
CUDA_VISIBLE_DEVICES=1 python generation/diffusers_generate.py --lora_step '/openseg_blob/zhaoyaqi/workspace/coco80_grpo_counting_sd3_5_medium/output/sd_3_5_medium_base_data_8k_coco80_v5/checkpoint-2000' --outdir "sd-3.5-m-base-coco80-8k-v5-ft-step2k" --n_samples 4 &
CUDA_VISIBLE_DEVICES=2 python generation/diffusers_generate.py --lora_step '/openseg_blob/zhaoyaqi/workspace/coco80_grpo_counting_sd3_5_medium/output/sd_3_5_medium_base_data_8k_coco80_v5/checkpoint-4000' --outdir "sd-3.5-m-base-coco80-8k-v5-ft-step4k" --n_samples 4 &
CUDA_VISIBLE_DEVICES=3 python generation/diffusers_generate.py --lora_step '/openseg_blob/zhaoyaqi/workspace/coco80_grpo_counting_sd3_5_medium/output/sd_3_5_medium_base_data_8k_coco80_v5/checkpoint-4000' --outdir "sd-3.5-m-base-coco80-8k-v5-ft-step4k" --n_samples 4 &

# CUDA_VISIBLE_DEVICES=0 python generation/diffusers_generate.py --outdir "sd-3.5-m-base" --n_samples 4 &
# CUDA_VISIBLE_DEVICES=1 python generation/diffusers_generate.py --outdir "sd-3.5-m-base" --n_samples 4 &
# CUDA_VISIBLE_DEVICES=2 python generation/diffusers_generate.py --outdir "sd-3.5-m-base" --n_samples 4 &
# CUDA_VISIBLE_DEVICES=3 python generation/diffusers_generate.py --outdir "sd-3.5-m-base" --n_samples 4 &

# CUDA_VISIBLE_DEVICES=0 python generation/diffusers_generate.py --lora_step 316 --outdir "counting_coco80_10_step20_guidance_7" --metadata_file "/openseg_blob/zhaoyaqi/flow_grpo/dataset/counting/test_metadata.jsonl" &
# CUDA_VISIBLE_DEVICES=1 python generation/diffusers_generate.py --lora_step 316 --outdir "counting_coco80_10_step20_guidance_7" --metadata_file "/openseg_blob/zhaoyaqi/flow_grpo/dataset/counting/test_metadata.jsonl" &
# CUDA_VISIBLE_DEVICES=2 python generation/diffusers_generate.py --lora_step 316 --outdir "counting_coco80_10_step20_guidance_7" --metadata_file "/openseg_blob/zhaoyaqi/flow_grpo/dataset/counting/test_metadata.jsonl" &
# CUDA_VISIBLE_DEVICES=3 python generation/diffusers_generate.py --lora_step 316 --outdir "counting_coco80_10_step20_guidance_7" --metadata_file "/openseg_blob/zhaoyaqi/flow_grpo/dataset/counting/test_metadata.jsonl" &

# conda activate reward_server

# CUDA_VISIBLE_DEVICES=1 python evaluation/evaluate_images.py "sd-3.5-m-base-coco80-8k-ft-step3k" &
# CUDA_VISIBLE_DEVICES=2 python evaluation/evaluate_images.py "sd-3.5-m-base-coco80-8k-ft-step5k" &

# python evaluation/summary_scores.py "sd-3.5-m-base-coco80-8k-ft-step3k" 
# python evaluation/summary_scores.py "sd-3.5-m-base-coco80-8k-ft-step5k" 

wait

# python evaluation/evaluate_images.py $NAME 
# CUDA_VISIBLE_DEVICES=0 python evaluation/evaluate_images.py sd-3.5-m-base-coco80-8k-v5-ft-1k-denoise-step10
# python evaluation/summary_scores.py sd-3.5-m-base-coco80-8k-v5-ft-1k-denoise-step40

# python evaluation/summary_scores.py "<RESULTS_FOLDER>/results.jsonl"
# python evaluation/summary_scores.py sd-3.5-m-base
