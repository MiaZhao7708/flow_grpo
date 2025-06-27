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
from peft import PeftModel, LoraConfig, get_peft_model
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
    
    if 'medium' in model_name:
        pipeline = StableDiffusion3Pipeline.from_pretrained(
            model_name, 
            torch_dtype=torch.bfloat16, 
            token=huggingface_token
        )
    else:
        pipeline = StableDiffusion3Pipeline.from_pretrained(
            model_name, 
            torch_dtype=torch.bfloat16, 
            cache_dir="/detr_blob/liuzeyu/checkpoints/huggingface"
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
    
    # 检查LoRA参数是否存在
    lora_params = {}
    for name, param in transformer.named_parameters():
        if 'lora_' in name:
            lora_params[name] = param.data.abs().mean().item()
    
    if lora_params:
        print(f"✓ Found {len(lora_params)} LoRA parameters:")
        for i, (name, mean_val) in enumerate(lora_params.items()):
            if i < 5:  # 只显示前5个
                print(f"  - {name}: mean_abs_value = {mean_val:.6f}")
            elif i == 5:
                print(f"  - ... and {len(lora_params)-5} more")
                break
    else:
        print("❌ No LoRA parameters found!")
    
    print("-" * 50)

def load_lora_weights(pipeline, lora_path):
    """
    Load LoRA weights by manually correcting state dict keys to match diffusers' expectations.
    """
    if not os.path.exists(lora_path):
        raise ValueError(f"LoRA path does not exist: {lora_path}")

    if os.path.isdir(lora_path) and os.path.exists(os.path.join(lora_path, "lora")):
        lora_weights_path = os.path.join(lora_path, "lora")
    else:
        lora_weights_path = lora_path

    print(f"Loading LoRA weights from: {lora_weights_path} with manual key correction.")
    
    lora_checkpoint_file = os.path.join(lora_weights_path, "adapter_model.safetensors")
    if not os.path.exists(lora_checkpoint_file):
        raise ValueError(f"adapter_model.safetensors not found in {lora_weights_path}")

    # 1. Load the state dict from the safetensors file
    lora_state_dict = load_file(lora_checkpoint_file, device="cpu")
    
    # 2. Correct the keys
    # The warning "No LoRA keys... with prefix='transformer'" indicates the keys in the
    # checkpoint don't have the prefix diffusers expects for the transformer model.
    # We will manually add the 'transformer.' prefix.
    # The PEFT library adds its own prefixes like 'base_model.model.' during training.
    pipeline_state_dict = {}
    for key, value in lora_state_dict.items():
        if key.startswith("base_model.model."):
            # Replace the PEFT prefix with the diffusers-expected prefix
            new_key = "transformer." + key[len("base_model.model."):]
            pipeline_state_dict[new_key] = value
        else:
            # Keep other keys if any, though we only expect transformer keys
            pipeline_state_dict[key] = value

    # 3. Load the corrected state dict into the pipeline.
    # This should now work without warnings for the transformer.
    pipeline.load_lora_weights(pipeline_state_dict)

    # 4. Fuse the weights
    print("Fusing LoRA weights into the base model...")
    pipeline.fuse_lora()
    
    pipeline.transformer.eval()

    print("✓ LoRA weights loaded via corrected state_dict and fused successfully!")


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

def run_inference(pipeline, prompt, num_images=1, height=512, width=512, num_steps=20, guidance_scale=7.5, seed=42):
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
        # Use the same pipeline_with_logprob function as training code
        # images, _, _, _ = pipeline_with_logprob(
        #     pipeline,
        #     prompt=prompt,
        #     negative_prompt="",
        #     num_images_per_prompt=num_images,
        #     height=height,
        #     width=width,
        #     guidance_scale=guidance_scale,
        #     num_inference_steps=num_steps,
        #     output_type="pt",
        #     return_dict=False,
        #     determistic=True,  # For reproducible results
        # )
        images = pipeline(
                prompt=prompt,
                num_images_per_prompt=num_images,
                height=height,
                width=width,
                guidance_scale=guidance_scale,
                num_inference_steps=40
            ).images
        
        # Convert from tensor to PIL images
        # images = pipeline.image_processor.postprocess(images, output_type="pil")
    
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
    parser.add_argument("--guidance_scale", type=float, default=7.5)
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
        load_lora_weights(lora_pipeline, args.lora_path)
        
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