#! /bin/bash

# NAME="sd-3.5-m-base"
STEP=316

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



CUDA_VISIBLE_DEVICES=0 python generation/diffusers_generate.py --lora_step $STEP &
CUDA_VISIBLE_DEVICES=1 python generation/diffusers_generate.py --lora_step $STEP &
CUDA_VISIBLE_DEVICES=2 python generation/diffusers_generate.py --lora_step $STEP &
CUDA_VISIBLE_DEVICES=3 python generation/diffusers_generate.py --lora_step $STEP &

wait

# python evaluation/evaluate_images.py $NAME 
# python evaluation/evaluate_images.py sd-3.5-m-flow-grpo-step${STEP}

# python evaluation/summary_scores.py "<RESULTS_FOLDER>/results.jsonl"
# python evaluation/summary_scores.py sd-3.5-m-base
