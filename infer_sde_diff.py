from collections import defaultdict
import contextlib
import os
import datetime
from concurrent import futures
import time
import json
import debugpy

import os
os.environ['PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT'] = '10.0'  # 设置为5秒
from accelerate.logging import get_logger
from diffusers import StableDiffusion3Pipeline
import numpy as np
import sys
sys.path.append("/openseg_blob/zhaoyaqi/flow_grpo")
import os
from flow_grpo.diffusers_patch.sd3_pipeline_with_logprob import pipeline_with_logprob
from flow_grpo.diffusers_patch.train_dreambooth_lora_sd3 import encode_prompt
import torch
from functools import partial
import tqdm
import random
from torch.utils.data import Dataset, DataLoader

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)
huggingface_token = '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'
torch.set_grad_enabled(False)


logger = get_logger(__name__)


class GenevalPromptDataset(Dataset):
    def __init__(self, dataset, split='train'):
        self.file_path = os.path.join(dataset, f'{split}_metadata.jsonl')
        with open(self.file_path, 'r', encoding='utf-8') as f:
            self.metadatas = [json.loads(line) for line in f]
            self.prompts = [item['prompt'] for item in self.metadatas]
        
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx], "metadata": self.metadatas[idx]}

    @staticmethod
    def collate_fn(examples):
        prompts = [example["prompt"] for example in examples]
        metadatas = [example["metadata"] for example in examples]
        return prompts, metadatas


def compute_text_embeddings(prompt, text_encoders, tokenizers, max_sequence_length, device):
    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds = encode_prompt(
            text_encoders, tokenizers, prompt, max_sequence_length
        )
        prompt_embeds = prompt_embeds.to(device)
        pooled_prompt_embeds = pooled_prompt_embeds.to(device)
    return prompt_embeds, pooled_prompt_embeds

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def load_lora_weights(pipeline, merge_lora_path):    
    """Load and fuse LoRA weights into the pipeline's transformer.
    
    Args:
        pipeline: The diffusion pipeline
        merge_lora_path: Path to LoRA weights, can be a string or list of strings
    """
    if isinstance(merge_lora_path, str):
        pretrained_lora_ckpt_path_list = [merge_lora_path]
    else:
        pretrained_lora_ckpt_path_list = merge_lora_path  # fuse multiple lora weights
        
    for pretrained_lora_ckpt in pretrained_lora_ckpt_path_list:
        lora_state_dict = pipeline.lora_state_dict(pretrained_lora_ckpt,strict=False)
        pipeline.load_lora_into_transformer(lora_state_dict, pipeline.transformer)
        pipeline.transformer.fuse_lora(safe_fusing=True)

        pipeline.transformer.unload_lora()
        logger.info(f"[INFO] loaded pretrained lora weights from {pretrained_lora_ckpt} and fused the lora to the base model")
        print(f"[INFO] loaded pretrained lora weights from {pretrained_lora_ckpt} and fused the lora to the base model")


def eval(pipeline, test_dataloader, text_encoders, tokenizers, args, device):
    neg_prompt_embed, neg_pooled_prompt_embed = compute_text_embeddings([""], text_encoders, tokenizers, max_sequence_length=128, device=device)

    sample_neg_prompt_embeds = neg_prompt_embed.repeat(args.test_batch_size, 1, 1)
    sample_neg_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(args.test_batch_size, 1)

    # test_dataloader = itertools.islice(test_dataloader, 2)
    for index_sample,test_batch in enumerate(tqdm(
            test_dataloader,
            desc="Eval: ",
            position=0,
        )):
        prompts, prompt_metadata = test_batch
        print(f"sample {index_sample} prompts: {prompts}")
        prompt_embeds, pooled_prompt_embeds = compute_text_embeddings(
            prompts, 
            text_encoders, 
            tokenizers, 
            max_sequence_length=128, 
            device=device
        )
        # shape = (16,64,64)
        latent = torch.randn(16,64,64).to(device)
        # latent = torch.load("/openseg_blob/zhaoyaqi/flow_grpo/latent.pt").to(device)
        latents = latent.unsqueeze(0).repeat(args.num_images_per_prompt, 1, 1, 1)
        latents_sde = latents[0].unsqueeze(0)
        # ode sample
        with torch.no_grad():
            samples = pipeline(
                prompts,
                height=args.resolution,
                width=args.resolution,
                num_inference_steps=args.eval_num_steps,
                guidance_scale=args.guidance_scale,
                num_images_per_prompt=args.num_images_per_prompt,
                negative_prompt="",
                latents=latents,
            ).images

        for index_ode,sample in enumerate(samples):
            output_path_ode = os.path.join(args.output_dir, f"sample_{index_sample:02d}_image_{index_ode:02d}_ode.png")
            sample.save(output_path_ode)

        # sde sample
        for index_sde in tqdm(range(args.num_images_per_prompt), desc="SDE sample"):
            output_path_sde = os.path.join(args.output_dir, f"sample_{index_sample:02d}_image_{index_sde:02d}_sde.png")
            if os.path.exists(output_path_sde):
                continue
            with torch.no_grad():
                images, latents, log_probs, _ = pipeline_with_logprob(
                    pipeline,
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    negative_prompt_embeds=sample_neg_prompt_embeds,
                    negative_pooled_prompt_embeds=sample_neg_pooled_prompt_embeds,
                    num_inference_steps=args.eval_num_steps,
                    guidance_scale=args.guidance_scale,
                    latents=latents_sde,
                    output_type="pil",
                    return_dict=False,
                    height=args.resolution,
                    width=args.resolution, 
                    determistic=False,
                )
                images[0].save(output_path_sde)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="/openseg_blob/zhaoyaqi/flow_grpo/dataset/geneval")
    parser.add_argument("--test_batch_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_num_steps", type=int, default=40)
    parser.add_argument("--guidance_scale", type=float, default=7.0)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--output_dir", type=str, default="/openseg_blob/zhaoyaqi/flow_grpo/sde_diff/")
    parser.add_argument("--num_images_per_prompt", type=int, default=5)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    pretrained_model = "stabilityai/stable-diffusion-3.5-medium" 

    seed_everything(args.seed)

    pipeline = StableDiffusion3Pipeline.from_pretrained(pretrained_model,torch_dtype=torch.bfloat16, token=huggingface_token)
    pipeline = pipeline.to(device)
    pipeline.transformer.eval()

    text_encoders = [pipeline.text_encoder, pipeline.text_encoder_2, pipeline.text_encoder_3]
    tokenizers = [pipeline.tokenizer, pipeline.tokenizer_2, pipeline.tokenizer_3]

    test_dataset = GenevalPromptDataset(args.dataset, 'test')

    # 创建正常的DataLoader
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        collate_fn=GenevalPromptDataset.collate_fn,
        shuffle=False,
        num_workers=8,
    )
    
    eval(pipeline, test_dataloader, text_encoders, tokenizers, args, device)


if __name__ == "__main__":
    main()