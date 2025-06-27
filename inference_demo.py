#!/usr/bin/env python3
"""
Simple inference demo for Flow-GRPO counting model with LoRA weights.
"""

import argparse
import json
import os
import random
import torch
from PIL import Image
from diffusers import StableDiffusion3Pipeline
from peft import PeftModel
import sys
from safetensors.torch import load_file

# Add the current directory to Python path
sys.path.append('/openseg_blob/zhaoyaqi/flow_grpo')

# Import the custom pipeline function used in training
from flow_grpo.diffusers_patch.sd3_pipeline_with_logprob import pipeline_with_logprob

# Use environment variable for Hugging Face token
huggingface_token = os.getenv('HUGGINGFACE_TOKEN')

# Set HuggingFace cache directory
os.environ['HF_HOME'] = '/detr_blob/liuzeyu/checkpoints/huggingface'

def build_pipeline(model_name, device_id=0):
    """
    Build the diffusion pipeline.
    
    Args:
        model_name: Name or path of the base model
        device_id: GPU device ID
    
    Returns:
        pipeline: Initialized StableDiffusion3Pipeline
    """
    device = torch.device("cuda", index=device_id)
    
    pipeline = StableDiffusion3Pipeline.from_pretrained(
        model_name, 
        torch_dtype=torch.float16, 
        token=huggingface_token
    )

    
    pipeline.to(device=device)
    return pipeline

def check_lora_loading_status(transformer):
    """检查LoRA加载状态 - 推理版本"""
    print("=== LoRA Loading Status ===")
    
    # 检查是否有adapter
    if hasattr(transformer, 'peft_config'):
        print("✓ PEFT Config found:")
        for adapter_name, config in transformer.peft_config.items():
            print(f"  - Adapter: {adapter_name}, Type: {type(config).__name__}")
            print(f"    Rank: {config.r}, Alpha: {config.lora_alpha}")
            print(f"    Target modules: {config.target_modules}")
    else:
        print("❌ No PEFT Config found!")
    
    # 检查当前激活的adapter
    if hasattr(transformer, 'active_adapters'):
        print(f"✓ Active adapters: {transformer.active_adapters}")
    else:
        print("❌ No active adapters found!")
    
    # 检查总参数数量
    total_params = sum(p.numel() for p in transformer.parameters())
    print(f"✓ Total parameters: {total_params:,}")
    
    print("-" * 50)

def load_lora_weights(pipeline, lora_path, device_id=0):
    """
    Load LoRA weights using PeftModel.
    """
    if not os.path.exists(lora_path):
        raise ValueError(f"LoRA path does not exist: {lora_path}")

    print(f"Loading LoRA weights from: {lora_path}")
    print(f"Transformer type before loading: {type(pipeline.transformer)}")
    
    try:
        # Load LoRA weights using PeftModel
        pipeline.transformer = PeftModel.from_pretrained(
            pipeline.transformer, 
            lora_path,
            torch_dtype=torch.float16,
            device_map={"": device_id}
        )
        print(f"Transformer type after loading: {type(pipeline.transformer)}")
        print("LoRA config:", pipeline.transformer.peft_config)
        
        # Merge and unload LoRA weights
        print("Merging LoRA weights into the base model...")
        pipeline.transformer = pipeline.transformer.merge_and_unload()
        print(f"Transformer type after merging: {type(pipeline.transformer)}")
        
        device = torch.device("cuda", index=device_id)
        pipeline = pipeline.to(device)
        pipeline.transformer.eval()
        print("✓ LoRA weights loaded and merged successfully!")
        return pipeline
    except Exception as e:
        print(f"Error loading LoRA weights: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

def load_random_test_prompt(test_file_path):
    """
    Load a random prompt from the test metadata file.
    
    Args:
        test_file_path: Path to the test_metadata.jsonl file
    
    Returns:
        dict: Selected test sample with prompt and metadata
    """
    if not os.path.exists(test_file_path):
        raise ValueError(f"Test file does not exist: {test_file_path}")
    
    test_samples = []
    with open(test_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            test_samples.append(json.loads(line.strip()))
    
    if not test_samples:
        raise ValueError("No test samples found in the file")
    
    # Randomly select a sample
    selected_sample = random.choice(test_samples)
    print(f"Selected test sample:")
    print(f"  - Class: {selected_sample['class']}")
    print(f"  - Expected count: {selected_sample['include'][0]['count']}")
    print(f"  - Prompt: {selected_sample['prompt']}")
    
    return selected_sample

def run_inference(pipeline, prompt, num_images=1, height=512, width=512, num_steps=40, guidance_scale=7.5, seed=42):
    """
    Run inference with the given prompt.
    
    Args:
        pipeline: The diffusion pipeline
        prompt: Text prompt for generation
        num_images: Number of images to generate
        height: Image height
        width: Image width
        num_steps: Number of inference steps
        guidance_scale: Guidance scale for CFG
        seed: Random seed for reproducibility
    
    Returns:
        list: Generated PIL images
    """
    # Set seed for reproducibility
    torch.manual_seed(seed)
    
    print(f"\nGenerating {num_images} image(s) with prompt:")
    print(f"'{prompt}'")
    print(f"Parameters: {height}x{width}, {num_steps} steps, guidance={guidance_scale}")
    
    with torch.no_grad():
        images = pipeline(
            prompt=prompt,
            negative_prompt="",
            num_images_per_prompt=num_images,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            num_inference_steps=num_steps
        ).images
    
    return images

def save_images(images, output_dir, prefix="demo"):
    """
    Save generated images to the output directory.
    
    Args:
        images: List of PIL images
        output_dir: Output directory path
        prefix: Filename prefix
    """
    os.makedirs(output_dir, exist_ok=True)
    
    saved_paths = []
    for i, image in enumerate(images):
        filename = f"{prefix}_{i:03d}.png"
        filepath = os.path.join(output_dir, filename)
        image.save(filepath)
        saved_paths.append(filepath)
        print(f"✓ Saved: {filepath}")
    
    return saved_paths


def main():
    parser = argparse.ArgumentParser(description="Flow-GRPO Inference Demo")
    parser.add_argument("--lora_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="stabilityai/stable-diffusion-3.5-medium")
    parser.add_argument("--test_file", type=str, default="/openseg_blob/zhaoyaqi/flow_grpo/dataset/counting/test_metadata.jsonl")
    parser.add_argument("--output_dir", type=str, default="./demo_outputs")
    parser.add_argument("--num_images", type=int, default=4)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--num_steps", type=int, default=40)
    parser.add_argument("--guidance_scale", type=float, default=4.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--custom_prompt", type=str, default=None)
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Flow-GRPO Inference Demo")
    print("=" * 60)
    
    try:
        # 创建base和lora子文件夹
        base_output_dir = os.path.join(args.output_dir, "base")
        lora_output_dir = os.path.join(args.output_dir, "lora")
        os.makedirs(base_output_dir, exist_ok=True)
        os.makedirs(lora_output_dir, exist_ok=True)
        
        # 1. Get prompt first
        print("\n1. Preparing prompt...")
        if args.custom_prompt:
            prompt = args.custom_prompt
            print(f"Using custom prompt: {prompt}")
        else:
            test_sample = load_random_test_prompt(args.test_file)
            prompt = test_sample['prompt']
        
        # 2. 首先使用base model进行推理
        print("\n=== Running inference with base model ===")
        print("\n2. Building base pipeline...")
        base_pipeline = build_pipeline(args.model_name, args.device_id)
        print(f"✓ Base pipeline built with model: {args.model_name}")
        
        # 3. Run inference with base model
        print("\n3. Running inference with base model...")
        base_images = run_inference(
            pipeline=base_pipeline,
            prompt=prompt,
            num_images=args.num_images,
            height=args.height,
            width=args.width,
            num_steps=args.num_steps,
            guidance_scale=args.guidance_scale,
            seed=args.seed
        )
        
        # 4. Save base model results
        print("\n4. Saving base model results...")
        base_paths = save_images(base_images, base_output_dir, "base")
        
        # 5. Now create a new pipeline for LoRA
        print("\n=== Running inference with LoRA model ===")
        print("\n5. Building LoRA pipeline...")
        lora_pipeline = build_pipeline(args.model_name, args.device_id)
        print(f"✓ LoRA pipeline built with model: {args.model_name}")
        
        print("\n6. Loading LoRA weights...")
        # load_lora_weights(lora_pipeline, args.lora_path)
        lora_pipeline = load_lora_weights(lora_pipeline, args.lora_path, args.device_id)
        
        # 7. Run inference with LoRA model
        print("\n7. Running inference with LoRA model...")
        lora_images = run_inference(
            pipeline=lora_pipeline,
            prompt=prompt,
            num_images=args.num_images,
            height=args.height,
            width=args.width,
            num_steps=args.num_steps,
            guidance_scale=args.guidance_scale,
            seed=args.seed
        )
        
        # 8. Save LoRA model results
        print("\n8. Saving LoRA model results...")
        lora_paths = save_images(lora_images, lora_output_dir, "lora")
        
        print("\n" + "=" * 60)
        print("✓ Demo completed successfully!")
        print(f"✓ Generated {len(base_images)} images with base model")
        print(f"✓ Generated {len(lora_images)} images with LoRA model")
        print(f"✓ Base model results saved to: {base_output_dir}")
        print(f"✓ LoRA model results saved to: {lora_output_dir}")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 