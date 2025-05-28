#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software 
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import copy
import time
import gc
import logging
import math
import os
import shutil
import argparse
import itertools
import torch.nn.functional as F
from pathlib import Path
from mmengine.config import Config
from functools import partial

import numpy as np
import torch
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from peft import LoraConfig, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict
from tqdm.auto import tqdm
from transformers import CLIPTokenizer, PretrainedConfig, T5TokenizerFast

import diffusers
from diffusers import (
    AutoencoderKL,
    FluxTransformer2DModel,
    FlowMatchEulerDiscreteScheduler,
    FluxPipeline,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    cast_training_params,
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
    _set_state_dict_into_text_encoder
)
from diffusers.utils import (
    check_min_version,
    convert_unet_state_dict_to_peft,
    is_wandb_available,
)
from diffusers.utils.torch_utils import is_compiled_module
from diffusers.configuration_utils import FrozenDict

from custom_dataset import Text2ImageDataset,Text2ImageRGBDataset
from custom_pipeline import encode_prompt

if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.31.0.dev0")

logger = get_logger(__name__)
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["WANDB__SERVICE_WAIT"] = "300" # added by yaqi, wandb service wait time,mm05: 300
print(f"WANDB__SERVICE_WAIT: {os.environ['WANDB__SERVICE_WAIT']}")

def parse_config(path=None):
    
    if path is None:
        parser = argparse.ArgumentParser()
        parser.add_argument('config_dir', type=str)
        args = parser.parse_args()
        path = args.config_dir
    config = Config.fromfile(path)
    
    config.config_dir = path

    if "LOCAL_RANK" in os.environ:
        config.local_rank = int(os.environ["LOCAL_RANK"])
    elif "OMPI_COMM_WORLD_LOCAL_RANK" in os.environ:
        config.local_rank = int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])
    else:
        config.local_rank = -1

    return config

def load_text_encoders(class_one, class_two):
    text_encoder_one = class_one.from_pretrained(
        config.pretrained_model_name_or_path, subfolder="text_encoder", revision=config.revision, variant=config.variant,
        cache_dir=config.get("cache_dir", None),
    )
    text_encoder_two = class_two.from_pretrained(
        config.pretrained_model_name_or_path, subfolder="text_encoder_2", revision=config.revision, variant=config.variant,
        cache_dir=config.get("cache_dir", None),
    )

    return text_encoder_one, text_encoder_two

def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision,
        cache_dir=config.get("cache_dir", None),
    )
    model_class = text_encoder_config.architectures[0]
    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "T5EncoderModel":
        from transformers import T5EncoderModel

        return T5EncoderModel
    else:
        raise ValueError(f"{model_class} is not supported.")

def unwrap_model(model, accelerator):
    model = accelerator.unwrap_model(model)
    model = model._orig_mod if is_compiled_module(model) else model
    return model

def save_model_hook_partial(models, weights, output_dir, accelerator, transformer, text_encoder_one):
    if accelerator.is_main_process and len(weights) > 0:
        transformer_lora_layers_to_save = None
        text_encoder_one_lora_layers_to_save = None

        for model in models:
            if isinstance(model, type(unwrap_model(transformer, accelerator))):
                transformer_lora_layers_to_save = get_peft_model_state_dict(model)
            elif isinstance(model, type(unwrap_model(text_encoder_one, accelerator))):
                text_encoder_one_lora_layers_to_save = get_peft_model_state_dict(model)
            else:
                raise ValueError(f"unexpected save model: {model.__class__}")

            # make sure to pop weight so that corresponding model is not saved again
            weights.pop()

        FluxPipeline.save_lora_weights(
            output_dir,
            transformer_lora_layers=transformer_lora_layers_to_save,
            text_encoder_lora_layers=text_encoder_one_lora_layers_to_save,
        )

def load_model_hook_partial(models, input_dir, accelerator, transformer, text_encoder_one):
    if len(models) > 0:
        transformer_ = None
        text_encoder_one_ = None

        while len(models) > 0:
            model = models.pop()

            if isinstance(model, type(unwrap_model(transformer, accelerator))):
                transformer_ = model
            elif isinstance(model, type(unwrap_model(text_encoder_one, accelerator))):
                text_encoder_one_ = model
            else:
                raise ValueError(f"unexpected save model: {model.__class__}")

        lora_state_dict = FluxPipeline.lora_state_dict(input_dir)

        transformer_state_dict = {
            f'{k.replace("transformer.", "")}': v for k, v in lora_state_dict.items() if k.startswith("transformer.")
        }
        transformer_state_dict = convert_unet_state_dict_to_peft(transformer_state_dict)
        incompatible_keys = set_peft_model_state_dict(transformer_, transformer_state_dict, adapter_name="default")
        if incompatible_keys is not None:
            # check only for unexpected keys
            unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
            if unexpected_keys:
                logger.warning(
                    f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                    f" {unexpected_keys}. "
                )

        if config.train_text_encoder:
            # Do we need to call `scale_lora_layers()` here?
            _set_state_dict_into_text_encoder(lora_state_dict, prefix="text_encoder.", text_encoder=text_encoder_one_)

        # Make sure the trainable params are in float32. This is again needed since the base models
        # are in `weight_dtype`. More details:
        # https://github.com/huggingface/diffusers/pull/6514#discussion_r1449796804
        if config.mixed_precision in ["fp16", "bf16"]:
            models = [transformer_]
            if config.train_text_encoder:
                models.extend([text_encoder_one_])
            # only upcast trainable parameters (LoRA) into fp32
            cast_training_params(models)

def initialize_all_models(config, accelerator):

    # Load the tokenizers
    logger.info(f"[INFO] start load tokenizers")
    tokenizer_one = CLIPTokenizer.from_pretrained(
        config.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=config.revision,
        cache_dir=config.get("cache_dir", None),
    )
    tokenizer_two = T5TokenizerFast.from_pretrained(
        config.pretrained_model_name_or_path,
        subfolder="tokenizer_2",
        revision=config.revision,
        cache_dir=config.get("cache_dir", None),
    )
    
    # import correct text encoder classes
    logger.info(f"[INFO] start load text encoders")
    text_encoder_cls_one = import_model_class_from_model_name_or_path(
        config.pretrained_model_name_or_path, config.revision
    )
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        config.pretrained_model_name_or_path, config.revision, subfolder="text_encoder_2"
    )
    text_encoder_one, text_encoder_two = load_text_encoders(text_encoder_cls_one, text_encoder_cls_two)

    logger.info(f"[INFO] start load vae")
    vae: AutoencoderKL = AutoencoderKL.from_pretrained(
        config.pretrained_model_name_or_path,
        subfolder="vae",
        revision=config.revision,
        variant=config.variant,
        cache_dir=config.get("cache_dir", None),
    )
    vae.enable_slicing()

    logger.info(f"[INFO] start load mmdit")
    transformer = FluxTransformer2DModel.from_pretrained(
        config.transformer_varient if hasattr(config, "transformer_varient") else config.pretrained_model_name_or_path, 
        subfolder="" if hasattr(config, "transformer_varient") else "transformer", 
        revision=config.revision, 
        variant=config.variant,
        cache_dir=config.get("cache_dir", None),
    )

    # lora pretrained lora weights
    if hasattr(config, "pretrained_lora_dir"):
        lora_state_dict = FluxPipeline.lora_state_dict(config.pretrained_lora_dir)
        FluxPipeline.load_lora_into_transformer(lora_state_dict, None, transformer)
        transformer.fuse_lora(safe_fusing=True)
        transformer.unload_lora() # don't forget to unload the lora params
        logger.info(f"[INFO] fused pretrained lora weights from {config.pretrained_lora_dir}")

    vae.requires_grad_(False)
    transformer.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)

    logger.info(f"[INFO] move models to cuda")
    vae.to(accelerator.device, dtype=config.weight_dtype)
    transformer.to(accelerator.device, dtype=config.weight_dtype)
    text_encoder_one.to(accelerator.device, dtype=config.weight_dtype)
    text_encoder_two.to(accelerator.device, dtype=config.weight_dtype)

    if config.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()
        if config.train_text_encoder:
            text_encoder_one.gradient_checkpointing_enable()

    # now we will add new LoRA weights to the attention layers
    logger.info(f"[INFO] add lora in mmdit")
    target_modules = []
    # transformer_blocks
    module_names = ["to_k", "to_q", "to_v", "to_out.0"]
    for name, _ in transformer.transformer_blocks.named_modules():
        if any([name.endswith(n) for n in module_names]):
            target_modules.append("transformer_blocks." + name)
    # single_transformer_blocks
    module_names = ["to_k", "to_q", "to_v"]
    for name, _ in transformer.single_transformer_blocks.named_modules():
        if any([name.endswith(n) for n in module_names]):
            target_modules.append("single_transformer_blocks." + name)

    transformer_lora_config = LoraConfig(
        r=config.rank,
        lora_alpha=config.rank,
        init_lora_weights="gaussian",
        target_modules=target_modules,
    )
    transformer.add_adapter(transformer_lora_config)

    if config.train_text_encoder:
        text_lora_config = LoraConfig(
            r=config.text_encoder_rank,
            lora_alpha=config.text_encoder_rank,
            init_lora_weights="gaussian",
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        )
        text_encoder_one.add_adapter(text_lora_config)

    tokenizers = [tokenizer_one, tokenizer_two]
    text_encoders = [text_encoder_one, text_encoder_two]

    # Make sure the trainable params are in float32.  
    logger.info(f"[INFO] cast_training_params to fp32")
    if config.mixed_precision in ["fp16", "bf16"]:
        models = [transformer]
        if config.train_text_encoder:
            models.extend([text_encoder_one])
        # only upcast trainable parameters (LoRA) into fp32
        cast_training_params(models, dtype=torch.float32)
    
    return vae, transformer, tokenizers, text_encoders

def get_trainable_params(config, accelerator, transformer, text_encoder_one):

    transformer_lora_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))

    # Optimization parameters
    transformer_parameters_with_lr = {"params": transformer_lora_parameters, "lr": config.learning_rate}
    params_to_optimize = [transformer_parameters_with_lr]

    if config.train_text_encoder:
        text_encoder_one_params = list(filter(lambda p: p.requires_grad, text_encoder_one.parameters()))
        text_parameters_one_with_lr = {
            "params": text_encoder_one_params,
            "weight_decay": config.adam_weight_decay_text_encoder,
            "lr": config.text_encoder_lr,
        }
        params_to_optimize.extend([text_parameters_one_with_lr])
    
    if accelerator.is_main_process:
        for i, param_set in enumerate(params_to_optimize):
            num_params = sum([p.numel() for p in param_set["params"]]) / 1e+6
            print(f"Trainable Params Set {i}: {num_params:02f}M")
    
    return params_to_optimize

def get_sigmas(timesteps, accelerator, noise_scheduler_copy, n_dim=4, dtype=torch.float32):
    sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
    schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
    timesteps = timesteps.to(accelerator.device)
    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma    

def log_validation(
    pipeline,
    config,
    accelerator,
    global_step
):
    logger.info(f"Running validation... \n Generating {config.num_validation_images} images per case")
    pipeline = pipeline.to(accelerator.device)
    # pipeline.set_progress_bar_config(disable=True)

    # run inference
    image_logs = []
    generator = torch.Generator(device=accelerator.device).manual_seed(config.seed) if config.seed else None
    for validation_prompt in config.validation_prompts:
        with torch.autocast(accelerator.device.type, dtype=torch.bfloat16):
            images = [
                pipeline(
                    prompt=validation_prompt,
                    generator=generator,
                    height=config.resolution,
                    width=config.resolution,
                ).images[0]
                for _ in range(config.num_validation_images)
            ]
        image_logs.append(
            {
                "images": images, 
                "caption": validation_prompt,
            }
        )

    for tracker in accelerator.trackers:
        assert tracker.name == "wandb"
        formatted_images = []
        
        for log in image_logs:
            images = log["images"]
            validation_prompt = log["caption"]
            for idx, image in enumerate(images):
                image = wandb.Image(image, caption=validation_prompt)
                formatted_images.append(image)

        tracker.log({"validation": formatted_images})

    del pipeline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def collate_fn(examples):
    # print(type(examples), len(examples))
    # print(type(examples[0]), len(examples[0]), type(examples[0][0]), type(examples[0][1]), type(examples[0][2]))
    # print(type(examples[1]), len(examples[1]), type(examples[1][0]), type(examples[1][1]), type(examples[1][2]))
    # print(type(examples[2]), len(examples[2]), type(examples[2][0]), type(examples[2][1]), type(examples[2][2]))
    # print(type(examples[3]), len(examples[3]), type(examples[3][0]), type(examples[3][1]), type(examples[3][2]))
    pixel_values = torch.stack([example["image_pt"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    caption = [example["caption"] for example in examples]
    result = {
        "pixel_values": pixel_values,
        "caption": caption,
    }
    return result

def train(
        accelerator, progress_bar, first_epoch, global_step,
        vae, transformer, text_encoder_one, text_encoder_two, text_encoders, tokenizers,
        noise_scheduler_copy, optimizer, lr_scheduler, train_dataloader, config,
    ):

    # 添加时间步记录列表，但只记录当前区间的
    current_timesteps = []
    # 记录上一个检查点的步数
    last_checkpoint_step = global_step
    
    for epoch in range(first_epoch, config.num_train_epochs):
        transformer.train()
        if config.train_text_encoder:
            text_encoder_one.train()
            # set top parameter requires_grad = True for gradient checkpointing works
            accelerator.unwrap_model(text_encoder_one).text_model.embeddings.requires_grad_(True)

        for step, batch in enumerate(train_dataloader):
            models_to_accumulate = [transformer]
            if config.train_text_encoder:
                models_to_accumulate += [text_encoder_one]
            with accelerator.accumulate(models_to_accumulate):
                
                merged_pt = batch["pixel_values"].to(dtype=vae.dtype)

                # changed by yaqi,for RGB image
                # pixel_values_vae_input = merged_pt.to(accelerator.device)  # [bs, c_img, H, W]
                # pixel_values_vae_input = pixel_values_vae_input[:, :3] * ((pixel_values_vae_input[:, 3:4] + 1) / 2.) # [16,3,512,512]
                pixel_values_vae_input = merged_pt.to(accelerator.device)[:, :3] 

                # 编码提示词
                prompt_embeds, pooled_prompt_embeds, text_ids = encode_prompt(
                    text_encoders=text_encoders,
                    tokenizers=tokenizers,
                    prompt=batch["caption"],
                )

                # 支持single_sample模式，从单个样本创建多个不同噪声版本
                if config.get("single_sample", False) and merged_pt.shape[0] == 1:
                    # 确定目标批次大小
                    target_batch_size = config.train_target_batch_size                    
                    pixel_values_vae_input = pixel_values_vae_input.repeat(target_batch_size, 1, 1, 1)                    
                    prompt_embeds = prompt_embeds.repeat(target_batch_size, 1, 1)
                    pooled_prompt_embeds = pooled_prompt_embeds.repeat(target_batch_size, 1)
                    # text_ids 不需要重复，因为它会在函数中自动处理


                # Convert images to latent space
                model_input = vae.encode(pixel_values_vae_input).latent_dist.sample()
                model_input = (model_input - vae.config.shift_factor) * vae.config.scaling_factor
                model_input = model_input.to(dtype=config.weight_dtype) # 16,16,64,64
                vae_scale_factor = 2 ** (len(vae.config.block_out_channels))
                latent_image_ids = FluxPipeline._prepare_latent_image_ids(
                    model_input.shape[0],
                    model_input.shape[2],
                    model_input.shape[3],
                    accelerator.device,
                    config.weight_dtype,
                ) # 1024,3

                # Sample noise that we'll add to the latents
                
                noise = torch.randn_like(model_input) # 16,16,64,64
                bsz = model_input.shape[0]

                # Sample a random timestep for each image
                # for weighting schemes where we sample timesteps non-uniformly
                u = compute_density_for_timestep_sampling(
                    weighting_scheme=config.weighting_scheme,
                    batch_size=bsz,
                    logit_mean=config.logit_mean,
                    logit_std=config.logit_std,
                    mode_scale=config.mode_scale,
                )

                indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                timesteps = noise_scheduler_copy.timesteps[indices].to(device=model_input.device) # 16
                
                # added by yaqi
                if accelerator.is_main_process:
                    current_timesteps.extend(timesteps.cpu().numpy().tolist())

                # Add noise according to flow matching.
                sigmas = get_sigmas(timesteps, accelerator, noise_scheduler_copy, n_dim=model_input.ndim, dtype=model_input.dtype) # 16,1,1,1
                noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise # 16,16,64,64

                packed_noisy_model_input = FluxPipeline._pack_latents(
                    noisy_model_input, # 16,16,64,64
                    batch_size=model_input.shape[0],
                    num_channels_latents=model_input.shape[1],
                    height=model_input.shape[2],
                    width=model_input.shape[3],
                ) # 16,1024,64

                # handle guidance
                if config.get("guidance_scale", None):
                    guidance = torch.tensor([config.guidance_scale], device=accelerator.device)
                    guidance = guidance.expand(model_input.shape[0])
                else:
                    guidance = None
                

                # Predict the noise residual
                model_pred = transformer(
                    hidden_states=packed_noisy_model_input, # 16,1024,64
                    timestep=timesteps / 1000, # 16
                    guidance=guidance, # 16
                    encoder_hidden_states=prompt_embeds, # 16,512,4096
                    pooled_projections=pooled_prompt_embeds, # 16,768
                    txt_ids=text_ids, # 512,3
                    img_ids=latent_image_ids, # 1024,3
                    return_dict=False,
                )[0]

                model_pred = FluxPipeline._unpack_latents(
                    model_pred,
                    height=int(model_input.shape[2] * vae_scale_factor / 2),
                    width=int(model_input.shape[3] * vae_scale_factor / 2),
                    vae_scale_factor=vae_scale_factor,
                )

                if config.get("precondition_outputs", None):
                    model_pred = model_pred * (-sigmas) + noisy_model_input

                # these weighting schemes use a uniform timestep sampling
                # and instead post-weight the loss
                weighting = compute_loss_weighting_for_sd3(weighting_scheme=config.weighting_scheme, sigmas=sigmas)

                # flow matching loss
                if config.get("precondition_outputs", None):
                    target = model_input
                else:
                    target = noise - model_input  # 从噪声到原始数据的方向量

                # Compute regular loss.
                loss = torch.mean(
                    (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                    1,
                )
                loss = loss.mean()

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = (
                        itertools.chain(transformer.parameters(), text_encoder_one.parameters())
                        if config.train_text_encoder
                        else transformer.parameters()
                    )
                    accelerator.clip_grad_norm_(params_to_clip, config.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                
                if accelerator.is_main_process or accelerator.state.distributed_type == DistributedType.DEEPSPEED:
                    if global_step % config.checkpointing_steps == 0:
                        if accelerator.is_main_process:
                            # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                            if config.checkpoints_total_limit is not None:
                                checkpoints = os.listdir(config.output_dir)
                                checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                                checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                                # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                                if len(checkpoints) >= config.checkpoints_total_limit:
                                    num_to_remove = len(checkpoints) - config.checkpoints_total_limit + 1
                                    removing_checkpoints = checkpoints[0:num_to_remove]

                                    logger.info(
                                        f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                    )
                                    logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                    for removing_checkpoint in removing_checkpoints:
                                        removing_checkpoint = os.path.join(config.output_dir, removing_checkpoint)
                                        shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(config.output_dir, f"checkpoint-{global_step}")
                        if accelerator.state.distributed_type == DistributedType.DEEPSPEED:
                            extra_kwargs = {'exclude_frozen_parameters': True}
                        else:
                            extra_kwargs = {}
                        accelerator.save_state(save_path, **extra_kwargs)
                        logger.info(f"[Rank{accelerator.process_index}] saved state to {save_path}", main_process_only=False)

                        if accelerator.is_main_process and accelerator.state.distributed_type == DistributedType.DEEPSPEED:
                            transformer_lora_layers_to_save = get_peft_model_state_dict(accelerator.unwrap_model(transformer))
                            text_encoder_one_lora_layers_to_save = get_peft_model_state_dict(accelerator.unwrap_model(text_encoder_one)) if config.train_text_encoder else None
                            FluxPipeline.save_lora_weights(
                                save_path,
                                transformer_lora_layers=transformer_lora_layers_to_save,
                                text_encoder_lora_layers=text_encoder_one_lora_layers_to_save,
                            )
                        
                logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)

                # added by yaqi
                if accelerator.is_main_process and global_step % config.checkpointing_steps == 0:
                    # 创建ckpt目录
                    checkpoint_dir = os.path.join(config.output_dir, f"checkpoint-{global_step}")
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    
                    # 只保存当前区间的时间步
                    timestep_save_path = os.path.join(checkpoint_dir, f"timesteps_{last_checkpoint_step}_to_{global_step}.npy")
                    np.save(timestep_save_path, np.array(current_timesteps))
                    
                    logger.info(f"Saved timesteps from step {last_checkpoint_step} to {global_step} at {timestep_save_path}")
                    
                    # 更新上一个检查点步数
                    last_checkpoint_step = global_step
                    
                    # 清空当前区间的时间步列表，为下一个区间做准备
                    current_timesteps = []

                if global_step >= config.max_train_steps:
                    break

                if accelerator.is_main_process:
                    if config.validation_prompts is not None and (global_step % config.validation_steps == 0 or global_step == 1): # or global_step == 1
                        # create pipeline
                        pipeline = FluxPipeline.from_pretrained(
                            config.pretrained_model_name_or_path,
                            vae=vae,
                            text_encoder=accelerator.unwrap_model(text_encoder_one),
                            text_encoder_2=accelerator.unwrap_model(text_encoder_two),
                            transformer=accelerator.unwrap_model(transformer),
                            revision=config.revision,
                            variant=config.variant,
                            torch_dtype=config.weight_dtype,
                        )
                        log_validation(
                            pipeline=pipeline,
                            config=config,
                            accelerator=accelerator,
                            global_step=global_step,
                        )
                        torch.cuda.empty_cache()
                        gc.collect()

    # 训练结束后保存最后一个区间的时间步
    if accelerator.is_main_process and len(current_timesteps) > 0:
        final_dir = os.path.join(config.output_dir, "checkpoint-final")
        os.makedirs(final_dir, exist_ok=True)
        
        # 保存最后区间的时间步
        timestep_save_path = os.path.join(final_dir, f"timesteps_{last_checkpoint_step}_to_final.npy")
        np.save(timestep_save_path, np.array(current_timesteps))
        
        logger.info(f"Saved final timesteps from step {last_checkpoint_step} to end at {timestep_save_path}")

def main(config):
    if 'basecode_flux' in config.output_dir:
        config.output_dir = config.output_dir.replace('Count-FLUX/basecode_flux', 'workspace/cache_nips')
    print(f"config.output_dir: {config.output_dir}")
    logging_dir = Path(config.output_dir, config.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=config.output_dir, logging_dir=logging_dir)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision=config.mixed_precision,
        log_with=config.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    # Disable AMP for MPS. 
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    if config.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if config.seed is not None:
        set_seed(config.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if config.output_dir is not None:
            os.makedirs(config.output_dir, exist_ok=True)

    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora transformer) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    config.weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        config.weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        config.weight_dtype = torch.bfloat16

    if torch.backends.mps.is_available() and config.weight_dtype == torch.bfloat16:
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    # Load scheduler
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        config.pretrained_model_name_or_path, subfolder="scheduler",
        cache_dir=config.cache_dir if hasattr(config, "cache_dir") else None
    )
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if config.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    if config.scale_lr:
        config.learning_rate = (
            config.learning_rate * config.gradient_accumulation_steps * config.train_batch_size * accelerator.num_processes
        )

    # Initialize all models
    vae, transformer, tokenizers, text_encoders = initialize_all_models(config, accelerator)
    text_encoder_one, text_encoder_two = text_encoders

    save_model_hook = partial(
        save_model_hook_partial,
        accelerator=accelerator,
        transformer=transformer,
        text_encoder_one=text_encoder_one,
    )
    load_model_hook = partial(
        load_model_hook_partial,
        accelerator=accelerator,
        transformer=transformer,
        text_encoder_one=text_encoder_one,
    )
    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    # Get trainable parameters
    params_to_optimize = get_trainable_params(
        config, 
        accelerator, 
        transformer, 
        text_encoder_one, 
    )

    # Optimizer
    if config.get("optimizer", None) == "prodigy":
        import prodigyopt # type: ignore
        optimizer_class = prodigyopt.Prodigy
        if config.train_text_encoder and config.text_encoder_lr:
            params_to_optimize[1]["lr"] = config.learning_rate
        optimizer = optimizer_class(
            params_to_optimize,
            lr=config.learning_rate,
            betas=(config.adam_beta1, config.adam_beta2),
            beta3=config.prodigy_beta3,
            weight_decay=config.adam_weight_decay,
            eps=config.adam_epsilon,
            decouple=config.prodigy_decouple,
            use_bias_correction=config.prodigy_use_bias_correction,
            safeguard_warmup=config.prodigy_safeguard_warmup,
        )
    else:
        if config.get("use_8bit_adam", None):
            import bitsandbytes as bnb # type: ignore
            optimizer_class = bnb.optim.AdamW8bit
        else:
            optimizer_class = torch.optim.AdamW
        optimizer = optimizer_class(
            params_to_optimize,
            betas=(config.adam_beta1, config.adam_beta2),
            weight_decay=config.adam_weight_decay,
            eps=config.adam_epsilon,
        )

    # Dataset and DataLoaders creation:
    train_dataset = Text2ImageRGBDataset(**config.dataset_cfg)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=config.train_batch_size,
        num_workers=config.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / config.gradient_accumulation_steps)
    if config.max_train_steps is None:
        config.max_train_steps = config.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        config.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=config.max_train_steps * accelerator.num_processes,
        num_cycles=config.lr_num_cycles,
        power=config.lr_power,
    )

    # Prepare everything with our `accelerator`.
    if accelerator.state.distributed_type == DistributedType.DEEPSPEED:
        accelerator.state.deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = config.train_batch_size # mute annoying deepspeed errors
    transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, train_dataloader, lr_scheduler
    )
    if config.train_text_encoder:
        text_encoder_one = accelerator.prepare(text_encoder_one)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / config.gradient_accumulation_steps)
    if overrode_max_train_steps:
        config.max_train_steps = config.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    config.num_train_epochs = math.ceil(config.max_train_steps / num_update_steps_per_epoch)

    # Initialize trackers, also store our configuration.
    if accelerator.is_main_process:
        tracker_config = dict(copy.deepcopy(config))
        accelerator.init_trackers(
            project_name=config.tracker_project_name, 
            config=tracker_config, 
            init_kwargs={"wandb": {"name": config.wandb_job_name}},
        )

    # Train!
    total_batch_size = config.train_batch_size * accelerator.num_processes * config.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {config.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {config.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {config.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {config.max_train_steps}")
    
    # 检查并打印single_sample配置
    single_sample = config.get("single_sample", False)
    if single_sample:
        logger.info("***** Single Sample Mode Enabled *****")
        logger.info(f"  Training Mode: Using a single sample with {config.train_batch_size} different noise patterns")
        logger.info(f"  This mode will use the same image and prompt for all {config.train_batch_size} items in a batch")
        logger.info(f"  Each sample will have a different noise pattern and timestep")
    else:
        logger.info("***** Regular Batch Mode Enabled *****")
        logger.info(f"  Training Mode: Using {config.train_batch_size} different samples per batch")
    
    global_step = 0
    first_epoch = 0

    if config.base_checkpoint:
        accelerator.print(f"Resuming from checkpoint {config.base_checkpoint}")
        if accelerator.state.distributed_type == DistributedType.DEEPSPEED:
            extra_kwargs = {'load_module_strict': False}
        else:
            extra_kwargs = {}
        accelerator.load_state(config.base_checkpoint, **extra_kwargs)
            
    # Potentially load in the weights and states from a previous save
    if config.resume_from_checkpoint and config.base_checkpoint is None:
        if config.resume_from_checkpoint != "latest":
            path = os.path.basename(config.resume_from_checkpoint)
        else:
            # Get the mos recent checkpoint
            dirs = os.listdir(config.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{config.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            config.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            if accelerator.state.distributed_type == DistributedType.DEEPSPEED:
                extra_kwargs = {'load_module_strict': False}
            else:
                extra_kwargs = {}
            accelerator.load_state(os.path.join(config.output_dir, path), **extra_kwargs)
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, config.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    single_sample = config.get("single_sample", False)

    print(f"train_batch_size: {config.train_batch_size}")
    print(f"single_sample_multi_noise: {single_sample}")
    train(
        accelerator, progress_bar, first_epoch, global_step,
        vae, transformer, text_encoder_one, text_encoder_two, text_encoders, tokenizers,
        noise_scheduler_copy, optimizer, lr_scheduler, train_dataloader, config,
    )

    # finally, save the lora layers
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_path = os.path.join(config.output_dir, f"checkpoint-final")
        accelerator.save_state(save_path)
        logger.info(f"Final checkpoints is saved to {save_path}")

    accelerator.end_training()


if __name__ == "__main__":
    config = parse_config()
    main(config)
