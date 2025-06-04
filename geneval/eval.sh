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

# CUDA_VISIBLE_DEVICES=0 python generation/diffusers_generate.py --lora_step '/openseg_blob/zhaoyaqi/workspace/coco80_grpo_counting_sd3_5_medium/output/sd_3_5_medium_base_data_8k_coco80/checkpoint-1000' --outdir "sd-3.5-m-base-coco80-8k-step10" --steps 10 --n_samples 4 --metadata_file "/openseg_blob/zhaoyaqi/flow_grpo/dataset/counting/test_metadata.jsonl" &
# CUDA_VISIBLE_DEVICES=1 python generation/diffusers_generate.py --lora_step '/openseg_blob/zhaoyaqi/workspace/coco80_grpo_counting_sd3_5_medium/output/sd_3_5_medium_base_data_8k_coco80/checkpoint-1000' --outdir "sd-3.5-m-base-coco80-8k-step15" --steps 15 --n_samples 4 --metadata_file "/openseg_blob/zhaoyaqi/flow_grpo/dataset/counting/test_metadata.jsonl" &
# CUDA_VISIBLE_DEVICES=2 python generation/diffusers_generate.py --lora_step '/openseg_blob/zhaoyaqi/workspace/coco80_grpo_counting_sd3_5_medium/output/sd_3_5_medium_base_data_8k_coco80/checkpoint-1000' --outdir "sd-3.5-m-base-coco80-8k-step20" --steps 20 --n_samples 4 --metadata_file "/openseg_blob/zhaoyaqi/flow_grpo/dataset/counting/test_metadata.jsonl" &
# CUDA_VISIBLE_DEVICES=3 python generation/diffusers_generate.py --lora_step '/openseg_blob/zhaoyaqi/workspace/coco80_grpo_counting_sd3_5_medium/output/sd_3_5_medium_base_data_8k_coco80/checkpoint-1000' --outdir "sd-3.5-m-base-coco80-8k-step20" --steps 20 --n_samples 4 --metadata_file "/openseg_blob/zhaoyaqi/flow_grpo/dataset/counting/test_metadata.jsonl" &


# CUDA_VISIBLE_DEVICES=0 python generation/diffusers_generate.py --lora_step '/openseg_blob/zhaoyaqi/workspace/coco80_grpo_counting_sd3_5_medium/output/sd_3_5_medium_base_data_8k_coco80/checkpoint-1000' --outdir "sd-3.5-m-base-coco80-8k" --n_samples 4 &
# CUDA_VISIBLE_DEVICES=1 python generation/diffusers_generate.py --lora_step '/openseg_blob/zhaoyaqi/workspace/coco80_grpo_counting_sd3_5_medium/output/sd_3_5_medium_base_data_8k_coco80/checkpoint-1000' --outdir "sd-3.5-m-base-coco80-8k" --n_samples 4 &
# CUDA_VISIBLE_DEVICES=2 python generation/diffusers_generate.py --lora_step '/openseg_blob/zhaoyaqi/workspace/coco80_grpo_counting_sd3_5_medium/output/sd_3_5_medium_base_data_8k_coco80/checkpoint-1000' --outdir "sd-3.5-m-base-coco80-8k" --n_samples 4 &
# CUDA_VISIBLE_DEVICES=3 python generation/diffusers_generate.py --lora_step '/openseg_blob/zhaoyaqi/workspace/coco80_grpo_counting_sd3_5_medium/output/sd_3_5_medium_base_data_8k_coco80/checkpoint-1000' --outdir "sd-3.5-m-base-coco80-8k" --n_samples 4 &

CUDA_VISIBLE_DEVICES=0 python generation/diffusers_generate.py --outdir "sd-3.5-m-base" --n_samples 4 &
CUDA_VISIBLE_DEVICES=1 python generation/diffusers_generate.py --outdir "sd-3.5-m-base" --n_samples 4 &
CUDA_VISIBLE_DEVICES=2 python generation/diffusers_generate.py --outdir "sd-3.5-m-base" --n_samples 4 &
CUDA_VISIBLE_DEVICES=3 python generation/diffusers_generate.py --outdir "sd-3.5-m-base" --n_samples 4 &

# CUDA_VISIBLE_DEVICES=0 python generation/diffusers_generate.py --lora_step 316 --outdir "counting_coco80_10_step20_guidance_7" --metadata_file "/openseg_blob/zhaoyaqi/flow_grpo/dataset/counting/test_metadata.jsonl" &
# CUDA_VISIBLE_DEVICES=1 python generation/diffusers_generate.py --lora_step 316 --outdir "counting_coco80_10_step20_guidance_7" --metadata_file "/openseg_blob/zhaoyaqi/flow_grpo/dataset/counting/test_metadata.jsonl" &
# CUDA_VISIBLE_DEVICES=2 python generation/diffusers_generate.py --lora_step 316 --outdir "counting_coco80_10_step20_guidance_7" --metadata_file "/openseg_blob/zhaoyaqi/flow_grpo/dataset/counting/test_metadata.jsonl" &
# CUDA_VISIBLE_DEVICES=3 python generation/diffusers_generate.py --lora_step 316 --outdir "counting_coco80_10_step20_guidance_7" --metadata_file "/openseg_blob/zhaoyaqi/flow_grpo/dataset/counting/test_metadata.jsonl" &


wait

# python evaluation/evaluate_images.py $NAME 
# python evaluation/evaluate_images.py sd-3.5-m-flow-grpo-step${STEP}

# python evaluation/summary_scores.py "<RESULTS_FOLDER>/results.jsonl"
# python evaluation/summary_scores.py sd-3.5-m-base
