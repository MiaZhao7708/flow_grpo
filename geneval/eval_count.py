import json
import argparse
import os
import os.path as osp
import torch
import json
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich import print as rprint
from torchvision.utils import save_image
import os
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import *
import sys
sys.path.append('/openseg_blob/zhaoyaqi/Count-FLUX/multilayer_diffusion')
try:
    from multilayer.utils import (
        huggingface_token, 
        huggingface_cache_dir,
    )
except:
    print("============= import multilayer.utils failed ==============")
    huggingface_token = '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'
os.environ['HF_HOME'] = '/detr_blob/liuzeyu/checkpoints/huggingface'

DEMO_PROMPT = {
        'prompt': 'A serene mountain lake at sunset, with snow-capped peaks reflected in the crystal clear water, warm golden light illuminating the scene.',
        'gt_count': 0,
        'animals_str': 'peak'
        }

def number_to_words(n):
    # 定义阿拉伯数字到字母的映射
    num_to_word = {
        1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 
        6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten", 
        11: "eleven", 12: "twelve", 13: "thirteen", 14: "fourteen", 
        15: "fifteen", 16: "sixteen", 17: "seventeen", 18: "eighteen", 
        19: "nineteen", 20: "twenty", 30: "thirty", 40: "forty", 
        50: "fifty"
    }
    
    if n <= 20:
        return num_to_word[n]
    elif 21 <= n <= 50:
        tens = (n // 10) * 10
        ones = n % 10
        if ones == 0:
            return num_to_word[tens]
        else:
            return f"{num_to_word[tens]}-{num_to_word[ones]}"
    else:
        return "Number out of range"  # 可以根据需要调整


def build_pipeline(name, gpu_idx):

    device = torch.device("cuda", index=gpu_idx)  
    if 'medium' in name:
        pipeline = StableDiffusion3Pipeline.from_pretrained(name, torch_dtype=torch.bfloat16, token=huggingface_token)
    else:
        pipeline = StableDiffusion3Pipeline.from_pretrained(name, torch_dtype=torch.bfloat16, cache_dir="/detr_blob/liuzeyu/checkpoints/huggingface")
    pipeline.to(device=device)
    return pipeline


def generate_image(prompts, pipeline, args):
    torch.manual_seed(args.seed)
    
    rprint(f"[bold green]开始生成图像,共{len(prompts)}个prompt需要处理,每个生成{args.batch_size}张图片[/bold green]")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("[cyan]{task.fields[status]}[/cyan]"),
        transient=True,
    ) as progress:
        task = progress.add_task("生成进度", total=len(prompts), status="")
        
        for idx, prompt in enumerate(prompts):
            gt_count = prompt["gt_count"]
            animals_str = prompt["animals_str"].replace("s", "")
            prompt = prompt["prompt"]
            progress.update(task, status=f"处理 Prompt {idx}: {prompt}")

            gt = f"{gt_count}_{animals_str}"
            
            # 检查是否已经存在该prompt的生成图像或占位文件
            if any(f == f"case_{gt}_tmp" or f.startswith(f"case_{gt}_output") for f in os.listdir(args.output_dir)):
                progress.update(task, status=f"跳过 {gt} - 已有其他进程在处理")
                progress.update(task, advance=1)
                continue
            
            # 创建单个占位文件
            placeholder_path = os.path.join(args.output_dir, f"case_{gt}_tmp")
            with open(placeholder_path, 'w') as f:
                f.write('placeholder')
            
            try:
                # 生成图像
                progress.update(task, status=f"正在生成 {gt} 的图像...")
                images = pipeline(
                    prompt=prompt,
                    num_images_per_prompt=args.batch_size,
                    height=args.height,
                    width=args.width,
                    negative_prompt="",
                ).images
                
                # 删除占位文件并保存所有生成的图像
                if os.path.exists(placeholder_path):
                    os.remove(placeholder_path)
                for i, image in enumerate(images):
                    image.save(os.path.join(args.output_dir, f"case_{gt}_output_{i}.png"))
                
            except Exception as e:
                # 如果生成失败,保留占位文件用于后续统计
                progress.update(task, status=f"生成 {gt} 的图像失败: {str(e)}")
            
            progress.update(task, status=f"完成 {gt} 的图像生成")
            progress.update(task, advance=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="v01_lora_r64_bs16_flux_count")
    parser.add_argument("--height", type=int, default=512) 
    parser.add_argument("--width", type=int, default=512) 
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=10) # yaqi
    parser.add_argument("--prompt_file", type=str, default="eval_prompt_one_animal_50.json") # yaqi
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--ckpt_step", type=str, default="checkpoint-0")
    parser.add_argument("--folder", type=str, default="test_sd3")
    parser.add_argument("--model_name", type=str, default="stabilityai/stable-diffusion-3.5-large")
    parser.add_argument("--demo", type=bool, default=False)
    args = parser.parse_args()

    # 设置ckpt_dir和output_dir
    ckpt_folder = f"/openseg_blob/zhaoyaqi/workspace/sd3_5_counting/output/{args.folder}"
    args.ckpt_dir = os.path.join(ckpt_folder, args.ckpt_step)
    args.output_dir = f"/openseg_blob/zhaoyaqi/workspace/sd3_5_counting/output_eval/{args.ckpt_dir.split('output/')[-1].split('/')[0]}/{args.ckpt_dir.split('output/')[-1].split('/')[-1]}"

    os.makedirs(args.output_dir, exist_ok=True)
    print(f'============= output_dir: {args.output_dir} =============')
    # 1. 读取eval_in_domain_500.json
    with open(f'/openseg_blob/zhaoyaqi/Count-FLUX/playground/data/{args.prompt_file}', 'r') as f:
        prompts = json.load(f)
    
    if 'grid_layout' in args.folder:
        print(f"============= grid layout exp ==============")
        new_prompts = []
        for prompt in prompts:
            if int(prompt['gt_count'])==50:
                continue
            else:
                new_prompts.append(prompt)
        prompts = new_prompts
        print(f"============= use {len(prompts)} prompts ==============")
    
    else:
        print(f"============= use {len(prompts)} prompts ==============")
    
    if 'medium' in args.folder and '3_5' in args.folder:
        args.model_name = 'stabilityai/stable-diffusion-3.5-medium'
    
    print(f"============= use {args.model_name} ==============")
    
    pipeline = build_pipeline(args.model_name, args.gpu_id)
    
    # 添加判断：只有当ckpt_step不为"0"时才加载LoRA权重
    if args.ckpt_step != "checkpoint-0":
        pipeline.load_lora_weights(args.ckpt_dir)
    if args.demo:
        prompts = [DEMO_PROMPT]
    generate_image(prompts, pipeline, args)