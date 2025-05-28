"""Adapted from TODO"""

import argparse
import json
import os

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm, trange
from einops import rearrange
from torchvision.utils import make_grid
from torchvision.transforms import ToTensor
# from pytorch_lightning import seed_everything
from diffusers import DiffusionPipeline, StableDiffusionPipeline, StableDiffusion3Pipeline
from peft import PeftModel
import random
import torch
import numpy as np
import os

huggingface_token = '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'
torch.set_grad_enabled(False)

# python generation/diffusers_generate.py \
#     --metadata_file "/openseg_blob/zhaoyaqi/flow_grpo/dataset/counting/test_metadata.jsonl" \
#     --outdir "sd-3.5-m-base-coco80-8k"
#     --lora_step '/openseg_blob/zhaoyaqi/workspace/coco80_grpo_counting_sd3_5_medium/output/sd_3_5_medium_base_data_8k_coco80/checkpoint-1000'
 
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metadata_file",
        default="/openseg_blob/zhaoyaqi/workspace/flow_grpo/geneval/prompts/evaluation_metadata.jsonl",
        type=str,
        help="JSONL file containing lines of metadata for each prompt"
    )
    parser.add_argument(
        "--model",
        type=str,
        # default="runwayml/stable-diffusion-v1-5",
        default="stabilityai/stable-diffusion-3.5-medium",
        help="Huggingface model name"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=10,
        help="number of samples",
    )
    parser.add_argument(
        "--steps",
        type=int,
        # default=50,
        default=40,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--negative-prompt",
        type=str,
        nargs="?",
        const="ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, extra limbs, disfigured, deformed, body out of frame, bad anatomy, watermark, signature, cut off, low contrast, underexposed, overexposed, bad art, beginner, amateur, distorted face",
        # default=None,
        default="",
        help="negative prompt for guidance"
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--lora_step",
        type=int,
        default=None,
        help="path to lora file",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--scale",
        type=float,
        # default=9.0,
        default=4.5,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=10,
        help="how many samples can be produced simultaneously",
    )
    parser.add_argument(
        "--skip_grid",
        action="store_true",
        help="skip saving grid",
    )
    opt = parser.parse_args()
    return opt


def main(opt):
    # Load prompts
    with open(opt.metadata_file) as fp:
        metadatas = [json.loads(line) for line in fp]

    # 保持一致的index输出
    index_list = list(range(len(metadatas)))
    random.shuffle(index_list)

    # Load model
    if opt.model == "stabilityai/stable-diffusion-xl-base-1.0":
        model = DiffusionPipeline.from_pretrained(opt.model, torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
        model.enable_xformers_memory_efficient_attention()
    elif opt.model == "stabilityai/stable-diffusion-3.5-medium":
        print("---------- Using StableDiffusion3Pipeline ----------")
        model = StableDiffusion3Pipeline.from_pretrained(opt.model, torch_dtype=torch.bfloat16, token=huggingface_token)
    else:
        model = StableDiffusionPipeline.from_pretrained(opt.model, torch_dtype=torch.float16)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    # if opt.lora_step:
    #     print(f'---------- Using LoRA model: {opt.lora_step} ----------')
    #     lora_path = f'/openseg_blob/zhaoyaqi/workspace/flow_grpo/logs/geneval/sd3.5-M/checkpoints/checkpoint-{opt.lora_step}/lora'
    #     model.transformer = PeftModel.from_pretrained(model.transformer, lora_path)
    #     opt.outdir = f"sd-3.5-m-flow-grpo-step{opt.lora_step}"
    if opt.lora_step:
        print(f'---------- Using LoRA model: {opt.lora_step} ----------')
        model.load_lora_weights(opt.lora_step)
    
    
    # base_outdir = "/openseg_blob/zhaoyaqi/workspace/flow_grpo/coco_counting/output_eval"
    base_outdir = '/openseg_blob/zhaoyaqi/workspace/coco80_grpo_counting_sd3_5_medium/output_eval'
    opt.outdir = os.path.join(base_outdir, opt.outdir)
    print(f"---------- Output directory: {opt.outdir} ----------")
    # model.enable_attention_slicing()

    # for index, metadata in tqdm(enumerate(metadatas), total=len(metadatas)):
    for index in tqdm(index_list, total=len(index_list), desc="Generating"):
        metadata = metadatas[index]
        seed_everything(opt.seed)

        outpath = os.path.join(opt.outdir, f"{index:0>5}")
        os.makedirs(outpath, exist_ok=True)
        
        # multi-gpu generation, skip if the output already exists
        if os.path.exists(os.path.join(opt.outdir, f"{index:0>5}_tmp")) or os.path.exists(os.path.join(opt.outdir, f"{index:0>5}", "samples", '00003.png')):
            continue
        
        placeholder_path = os.path.join(opt.outdir, f"{index:0>5}_tmp")
        with open(placeholder_path, "w") as fp:
            fp.write('placeholder')
        
        prompt = metadata['prompt']
        n_rows = batch_size = opt.batch_size
        # print(f"Prompt ({index: >3}/{len(metadatas)}): '{prompt}'")

        sample_path = os.path.join(outpath, "samples")
        os.makedirs(sample_path, exist_ok=True)
        with open(os.path.join(outpath, "metadata.jsonl"), "w") as fp:
            json.dump(metadata, fp)

        sample_count = 0

        with torch.no_grad():
            all_samples = list()
            # for n in trange((opt.n_samples + batch_size - 1) // batch_size, desc="Sampling"):
            for n in trange(
                (opt.n_samples + batch_size - 1) // batch_size,
                desc=f"Sampling {prompt}",
                leave=False,
                position=1
            ):
                # Generate images
                samples = model(
                    prompt,
                    height=opt.H,
                    width=opt.W,
                    num_inference_steps=opt.steps,
                    guidance_scale=opt.scale,
                    num_images_per_prompt=min(batch_size, opt.n_samples - sample_count),
                    negative_prompt=opt.negative_prompt or None
                ).images
                for sample in samples:
                    sample.save(os.path.join(sample_path, f"{sample_count:05}.png"))
                    sample_count += 1
                if not opt.skip_grid:
                    all_samples.append(torch.stack([ToTensor()(sample) for sample in samples], 0))

            if not opt.skip_grid:
                # additionally, save as grid
                grid = torch.stack(all_samples, 0)
                grid = rearrange(grid, 'n b c h w -> (n b) c h w')
                grid = make_grid(grid, nrow=n_rows)

                # to image
                grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                grid = Image.fromarray(grid.astype(np.uint8))
                grid.save(os.path.join(outpath, f'grid.png'))
                del grid
        del all_samples

        try:
            if os.path.exists(placeholder_path):
                os.remove(placeholder_path)
        except Exception as e:
            print(f"---------- Error removing placeholder_path: {e} ----------")

    print("Done.")


if __name__ == "__main__":
    opt = parse_args()
    main(opt)
