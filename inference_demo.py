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

# Add the current directory to Python path
sys.path.append('/openseg_blob/zhaoyaqi/flow_grpo')

huggingface_token = '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'

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
    Load LoRA weights - 完全按照训练代码的逻辑实现
    """
    if not os.path.exists(lora_path):
        raise ValueError(f"LoRA path does not exist: {lora_path}")
    
    # 检查路径结构
    if os.path.isdir(lora_path) and os.path.exists(os.path.join(lora_path, "lora")):
        lora_weights_path = os.path.join(lora_path, "lora")
    else:
        lora_weights_path = lora_path
    
    # 验证LoRA权重文件存在
    if not (os.path.exists(os.path.join(lora_weights_path, "adapter_config.json")) and 
            os.path.exists(os.path.join(lora_weights_path, "adapter_model.safetensors"))):
        raise ValueError(f"Invalid LoRA weights at {lora_weights_path}. Missing adapter_config.json or adapter_model.safetensors")
    
    print(f"Loading LoRA weights from: {lora_weights_path}")
    
    # 1. 先初始化PEFT模型 - 与训练代码完全一致
    target_modules = [
        "attn.add_k_proj",
        "attn.add_q_proj",
        "attn.add_v_proj",
        "attn.to_add_out",
        "attn.to_k",
        "attn.to_out.0",
        "attn.to_q",
        "attn.to_v",
    ]
    transformer_lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        init_lora_weights="gaussian",
        target_modules=target_modules,
    )
    
    # 2. 使用get_peft_model初始化PEFT模型 - 与训练代码一致
    pipeline.transformer = get_peft_model(pipeline.transformer, transformer_lora_config)
    
    # 3. 加载LoRA权重 - 与训练代码load_model_hook_partial一致
    pipeline.transformer.load_adapter(lora_weights_path, adapter_name="default")
    pipeline.transformer.set_adapter("default")
    
    # 4. 设置为推理模式 - 所有参数不可训练
    pipeline.transformer.eval()
    for param in pipeline.transformer.parameters():
        param.requires_grad_(False)
    
    print("✓ LoRA weights loaded successfully!")
    print("✓ Model set to inference mode (all parameters frozen)")
    
    # 5. 检查加载状态
    # check_lora_loading_status(pipeline.transformer)

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
        images = pipeline(
            prompt=prompt,
            num_images_per_prompt=num_images,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            num_inference_steps=num_steps,
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

# def get_model_name_from_lora_path(lora_path):
#     """从LoRA路径中提取模型名称作为子文件夹名"""
#     # 从路径中提取有意义的名称
#     path_parts = lora_path.strip('/').split('/')
    
#     # 寻找包含有意义信息的部分
#     for part in reversed(path_parts):
#         if 'checkpoint-' in part:
#             continue  # 跳过checkpoint部分
#         if part in ['output', 'checkpoints', 'lora']:
#             continue  # 跳过通用目录名
#         if len(part) > 10:  # 选择较长的目录名，通常包含更多信息
#             return part
    
#     # 如果没找到合适的，使用倒数第二个非通用目录名
#     meaningful_parts = [p for p in path_parts if p not in ['output', 'checkpoints', 'lora'] and not p.startswith('checkpoint-')]
#     if meaningful_parts:
#         return meaningful_parts[-1]
    
#     return "lora_model"

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
        # 获取模型名称并创建子文件夹
        # model_name = get_model_name_from_lora_path(args.lora_path)
        final_output_dir = args.output_dir
        # final_output_dir = os.path.join(args.output_dir, model_name)
        # print(f"Model name: {model_name}")
        # print(f"Output directory: {final_output_dir}")
        
        # 1. Build pipeline
        print("\n1. Building pipeline...")
        pipeline = build_pipeline(args.model_name, args.device_id)
        print(f"✓ Pipeline built with model: {args.model_name}")
        
        # 2. Load LoRA weights
        print("\n2. Loading LoRA weights...")
        load_lora_weights(pipeline, args.lora_path)
        
        # 3. Get prompt
        print("\n3. Preparing prompt...")
        if args.custom_prompt:
            prompt = args.custom_prompt
            print(f"Using custom prompt: {prompt}")
        else:
            test_sample = load_random_test_prompt(args.test_file)
            prompt = test_sample['prompt']
        
        # 4. Run inference
        print("\n4. Running inference...")
        images = run_inference(
            pipeline=pipeline,
            prompt=prompt,
            num_images=args.num_images,
            height=args.height,
            width=args.width,
            num_steps=args.num_steps,
            guidance_scale=args.guidance_scale,
            seed=args.seed
        )
        
        # 5. Save results
        print("\n5. Saving results...")
        saved_paths = save_images(images, final_output_dir, "demo")
        
        print("\n" + "=" * 60)
        print("✓ Demo completed successfully!")
        print(f"✓ Generated {len(images)} images")
        print(f"✓ Results saved to: {final_output_dir}")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 