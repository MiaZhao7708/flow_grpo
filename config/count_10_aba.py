import ml_collections
import imp
import os
# run_name; merge_lora_path; save_dir

base = imp.load_source("base", os.path.join(os.path.dirname(__file__), "base.py"))

def compressibility():
    config = base.get_config()

    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.dataset = os.path.join(os.getcwd(), "dataset/pickscore")

    config.num_epochs = 100
    config.use_lora = True

    config.sample.batch_size = 8
    config.sample.num_batches_per_epoch = 4

    config.train.batch_size = 4
    config.train.gradient_accumulation_steps = 2

    # prompting
    config.prompt_fn = "general_ocr"

    # rewards
    config.reward_fn = {"jpeg_compressibility": 1}
    config.per_prompt_stat_tracking = True
    return config


##### cold start #####
def geneval_sd3_counting_10_step20_cold_start_from_3k():
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/counting")
    config.run_name = "0607_counting_coco80_10_step20_guidance_7_v5_cold_start_from_3k_aigc10"

    # sd3.5 medium
    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.merge_lora_path = '/openseg_blob/zhaoyaqi/workspace/coco80_grpo_counting_sd3_5_medium/output/sd_3_5_medium_base_data_8k_coco80_v5/checkpoint-3000'
    config.sample.num_steps = 20 # 40
    config.sample.eval_num_steps = 40
    # config.sample.guidance_scale = 4.5
    config.sample.guidance_scale = 7.0
    config.resume_from = '/openseg_blob/zhaoyaqi/workspace/coco80_grpo_counting_sd3_5_medium/output/counting_coco80_10_step20_guidance_7_v5_cold_start_from_3k/checkpoints'

    # 8 cards to start LLaVA Server
    config.resolution = 512
    config.sample.train_batch_size = 6
    config.sample.num_image_per_prompt = 12 # device_num = 4
    # config.sample.num_image_per_prompt = 3 # device_num = 1 
    config.sample.num_batches_per_epoch = 24
    config.sample.test_batch_size = 10 # This bs is a special design, the test set has a total of 2212, to make gpu_num*bs*n as close as possible to 2212, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2 # 24/2=12
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    config.train.beta = 0.01
    config.sample.kl_reward = 0
    config.sample.global_std=True
    config.train.ema=True
    config.num_epochs = 100000
    # config.save_freq = 30 # epoch
    # config.eval_freq = 60
    config.save_freq = 10 # epoch
    config.eval_freq = 20
    config.save_dir = '/openseg_blob/zhaoyaqi/workspace/coco80_grpo_counting_sd3_5_medium/output/counting_coco80_10_step20_guidance_7_v5_cold_start_from_3k'
    config.reward_fn = {
        "geneval": 1.0,
    }
    
    config.prompt_fn = "geneval"

    config.per_prompt_stat_tracking = True
    return config


##### init_same_noise #####
def geneval_sd3_counting_10_step20_init_same_noise():
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/counting")
    config.run_name = "0607_counting_coco80_10_step20_guidance_7_v5_init_same_noise_aigc09"

    # sd3.5 medium
    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.merge_lora_path = '/openseg_blob/zhaoyaqi/workspace/coco80_grpo_counting_sd3_5_medium/output/sd_3_5_medium_base_data_8k_coco80_v5/checkpoint-3000'
    config.sample.num_steps = 20 # 40
    config.sample.eval_num_steps = 40
    # config.sample.guidance_scale = 4.5
    config.sample.guidance_scale = 7.0
    config.sample.init_same_noise = True
    config.resume_from = '/openseg_blob/zhaoyaqi/workspace/coco80_grpo_counting_sd3_5_medium/output/counting_coco80_10_step20_guidance_7_v5_init_same_noise/checkpoints'

    # 8 cards to start LLaVA Server
    config.resolution = 512
    config.sample.train_batch_size = 6
    config.sample.num_image_per_prompt = 12 # device_num = 4
    # config.sample.num_image_per_prompt = 3 # device_num = 1 

    config.sample.num_batches_per_epoch = 24
    config.sample.test_batch_size = 10 # This bs is a special design, the test set has a total of 2212, to make gpu_num*bs*n as close as possible to 2212, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2 # 24/2=12
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    config.train.beta = 0.01
    config.sample.kl_reward = 0
    config.sample.global_std=True
    config.train.ema=True
    config.num_epochs = 100000
    # config.save_freq = 30 # epoch
    # config.eval_freq = 60
    config.save_freq = 10 # epoch
    config.eval_freq = 20
    config.save_dir = '/openseg_blob/zhaoyaqi/workspace/coco80_grpo_counting_sd3_5_medium/output/counting_coco80_10_step20_guidance_7_v5_init_same_noise'
    config.reward_fn = {
        "geneval": 1.0,
    }
    
    config.prompt_fn = "geneval"

    config.per_prompt_stat_tracking = True
    return config

##### timestep_select_strategy #####
def geneval_sd3_counting_10_step20_timestep_select_first_50():
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/counting")
    config.run_name = "0607_counting_coco80_10_step20_guidance_7_v5_timestep_select_first_50_aigc11"
    config.resume_from = '/openseg_blob/zhaoyaqi/workspace/coco80_grpo_counting_sd3_5_medium/output/counting_coco80_10_step20_guidance_7_v5_timestep_select_first_50/checkpoints'

    # sd3.5 medium
    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.merge_lora_path = '/openseg_blob/zhaoyaqi/workspace/coco80_grpo_counting_sd3_5_medium/output/sd_3_5_medium_base_data_8k_coco80_v5/checkpoint-3000'
    config.sample.num_steps = 20 # 40
    config.sample.eval_num_steps = 40
    # config.sample.guidance_scale = 4.5
    config.sample.guidance_scale = 7.0

    # 8 cards to start LLaVA Server
    config.resolution = 512
    config.sample.train_batch_size = 6
    config.sample.num_image_per_prompt = 12 # device_num = 4
    # config.sample.num_image_per_prompt = 3 # device_num = 1 

    config.sample.num_batches_per_epoch = 24
    config.sample.test_batch_size = 10 # This bs is a special design, the test set has a total of 2212, to make gpu_num*bs*n as close as possible to 2212, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2 # 24/2=12
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.5
    config.train.timestep_select_strategy = "first" # custom
    config.train.beta = 0.01
    config.sample.kl_reward = 0
    config.sample.global_std=True
    config.train.ema=True
    config.num_epochs = 100000
    # config.save_freq = 30 # epoch
    # config.eval_freq = 60
    config.save_freq = 10 # epoch
    config.eval_freq = 20
    config.save_dir = '/openseg_blob/zhaoyaqi/workspace/coco80_grpo_counting_sd3_5_medium/output/counting_coco80_10_step20_guidance_7_v5_timestep_select_first_50'
    config.reward_fn = {
        "geneval": 1.0,
    }
    
    config.prompt_fn = "geneval"

    config.per_prompt_stat_tracking = True
    return config

def geneval_sd3_counting_10_step20_timestep_select_random_50():
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/counting")
    config.run_name = "0607_counting_coco80_10_step20_guidance_7_v5_timestep_select_random_50_aigc12"
    config.resume_from = '/openseg_blob/zhaoyaqi/workspace/coco80_grpo_counting_sd3_5_medium/output/counting_coco80_10_step20_guidance_7_v5_timestep_select_random_50/checkpoints'

    # sd3.5 medium
    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.merge_lora_path = '/openseg_blob/zhaoyaqi/workspace/coco80_grpo_counting_sd3_5_medium/output/sd_3_5_medium_base_data_8k_coco80_v5/checkpoint-3000'
    config.sample.num_steps = 20 # 40
    config.sample.eval_num_steps = 40
    # config.sample.guidance_scale = 4.5
    config.sample.guidance_scale = 7.0

    # 8 cards to start LLaVA Server
    config.resolution = 512
    config.sample.train_batch_size = 6
    config.sample.num_image_per_prompt = 12 # device_num = 4
    # config.sample.num_image_per_prompt = 3 # device_num = 1 

    config.sample.num_batches_per_epoch = 24
    config.sample.test_batch_size = 10 # This bs is a special design, the test set has a total of 2212, to make gpu_num*bs*n as close as possible to 2212, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2 # 24/2=12
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.5
    config.train.timestep_select_strategy = "random"
    config.train.beta = 0.01
    config.sample.kl_reward = 0
    config.sample.global_std=True
    config.train.ema=True
    config.num_epochs = 100000
    # config.save_freq = 30 # epoch
    # config.eval_freq = 60
    config.save_freq = 10 # epoch
    config.eval_freq = 20
    config.save_dir = '/openseg_blob/zhaoyaqi/workspace/coco80_grpo_counting_sd3_5_medium/output/counting_coco80_10_step20_guidance_7_v5_timestep_select_random_50'
    config.reward_fn = {
        "geneval_debug": 1.0,
    }
    
    config.prompt_fn = "geneval"

    config.per_prompt_stat_tracking = True
    return config


def geneval_sd3_counting_10_step20_reward_strict():
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/counting")
    config.run_name = "0618_counting_coco80_10_step20_guidance_7_v5_reward_strict_aigc12"

    # sd3.5 medium
    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.merge_lora_path = '/openseg_blob/zhaoyaqi/workspace/coco80_grpo_counting_sd3_5_medium/output/sd_3_5_medium_base_data_8k_coco80_v5/checkpoint-3000'
    config.sample.num_steps = 20 # 40
    config.sample.eval_num_steps = 40
    # config.sample.guidance_scale = 4.5
    config.sample.guidance_scale = 7.0

    # 8 cards to start LLaVA Server
    config.resolution = 512
    config.sample.train_batch_size = 6
    config.sample.num_image_per_prompt = 12 # device_num = 4
    # config.sample.num_image_per_prompt = 3 # device_num = 1 

    config.sample.num_batches_per_epoch = 24
    config.sample.test_batch_size = 10 # This bs is a special design, the test set has a total of 2212, to make gpu_num*bs*n as close as possible to 2212, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2 # 24/2=12
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    config.train.reward_strict = True
    config.train.beta = 0.01
    config.sample.kl_reward = 0
    config.sample.global_std=True
    config.train.ema=True
    config.num_epochs = 100000
    # config.save_freq = 30 # epoch
    # config.eval_freq = 60
    config.save_freq = 10 # epoch
    config.eval_freq = 20
    config.save_dir = '/openseg_blob/zhaoyaqi/workspace/coco80_grpo_counting_sd3_5_medium/output/counting_coco80_10_step20_guidance_7_v5_reward_strict'
    config.reward_fn = {
        "geneval": 1.0,
    }
    
    config.prompt_fn = "geneval"

    config.per_prompt_stat_tracking = True
    return config

def geneval_sd3_counting_5_step20_reward_strict_first_50():
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/counting_5")
    config.run_name = "0717_counting_coco80_5_step20_guidance_7_v5_reward_strict_first_50_aigc10"
    config.resume_from = '/openseg_blob/zhaoyaqi/workspace/coco80_grpo_counting_sd3_5_medium/output/counting_coco80_5_step20_guidance_7_v5_reward_strict_first_50/checkpoints/checkpoint-1536'

    # sd3.5 medium
    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.merge_lora_path = '/openseg_blob/zhaoyaqi/workspace/coco80_grpo_counting_sd3_5_medium/output/sd_3_5_medium_base_data_8k_coco80_v5/checkpoint-3000'
    config.sample.num_steps = 20 # 40
    config.sample.eval_num_steps = 40
    # config.sample.guidance_scale = 4.5
    config.sample.guidance_scale = 7.0

    # 8 cards to start LLaVA Server
    config.resolution = 512
    config.sample.train_batch_size = 6
    config.sample.num_image_per_prompt = 12 # device_num = 4
    # config.sample.num_image_per_prompt = 3 # device_num = 1 

    config.sample.num_batches_per_epoch = 24
    config.sample.test_batch_size = 10 # This bs is a special design, the test set has a total of 2212, to make gpu_num*bs*n as close as possible to 2212, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2 # 24/2=12
    config.train.num_inner_epochs = 1
    # config.train.timestep_fraction = 0.99
    config.train.timestep_select_strategy = "first"
    config.train.timestep_fraction = 0.5
    config.train.reward_strict = True
    config.train.beta = 0.01
    config.sample.kl_reward = 0
    config.sample.global_std=True
    config.train.ema=True
    config.num_epochs = 100000
    # config.save_freq = 30 # epoch
    # config.eval_freq = 60
    config.save_freq = 10 # epoch
    config.eval_freq = 20
    config.save_dir = '/openseg_blob/zhaoyaqi/workspace/coco80_grpo_counting_sd3_5_medium/output/counting_coco80_5_step20_guidance_7_v5_reward_strict_first_50'
    config.reward_fn = {
        "geneval": 1.0,
    }
    
    config.prompt_fn = "geneval"

    config.per_prompt_stat_tracking = True
    return config

def geneval_sd3_counting_10_step20_reward_strict_first_50():
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/counting")
    config.run_name = "0717_counting_coco80_10_step20_guidance_7_v5_reward_strict_first_50_aigc09"
    config.resume_from = '/openseg_blob/zhaoyaqi/workspace/coco80_grpo_counting_sd3_5_medium/output/counting_coco80_10_step20_guidance_7_v5_reward_strict_first_50/checkpoints/checkpoint-3096'

    # sd3.5 medium
    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.merge_lora_path = '/openseg_blob/zhaoyaqi/workspace/coco80_grpo_counting_sd3_5_medium/output/sd_3_5_medium_base_data_8k_coco80_v5/checkpoint-3000'
    config.sample.num_steps = 20 # 40
    config.sample.eval_num_steps = 40
    # config.sample.guidance_scale = 4.5
    config.sample.guidance_scale = 7.0

    # 8 cards to start LLaVA Server
    config.resolution = 512
    config.sample.train_batch_size = 6
    config.sample.num_image_per_prompt = 12 # device_num = 4
    # config.sample.num_image_per_prompt = 3 # device_num = 1 

    config.sample.num_batches_per_epoch = 24
    config.sample.test_batch_size = 10 # This bs is a special design, the test set has a total of 2212, to make gpu_num*bs*n as close as possible to 2212, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2 # 24/2=12
    config.train.num_inner_epochs = 1
    # config.train.timestep_fraction = 0.99
    config.train.timestep_select_strategy = "first"
    config.train.timestep_fraction = 0.5
    config.train.reward_strict = True
    config.train.beta = 0.01
    config.sample.kl_reward = 0
    config.sample.global_std=True
    config.train.ema=True
    config.num_epochs = 100000
    # config.save_freq = 30 # epoch
    # config.eval_freq = 60
    config.save_freq = 10 # epoch
    config.eval_freq = 20
    config.save_dir = '/openseg_blob/zhaoyaqi/workspace/coco80_grpo_counting_sd3_5_medium/output/counting_coco80_10_step20_guidance_7_v5_reward_strict_first_50'
    config.reward_fn = {
        "geneval": 1.0,
    }
    
    config.prompt_fn = "geneval"

    config.per_prompt_stat_tracking = True
    return config

def geneval_sd3_counting_20_step20_reward_strict_first_50():
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/counting_20")
    config.run_name = "0717_counting_coco80_20_step20_guidance_7_v5_reward_strict_first_50_aigc11"
    config.resume_from = '/openseg_blob/zhaoyaqi/workspace/coco80_grpo_counting_sd3_5_medium/output/counting_coco80_20_step20_guidance_7_v5_reward_strict_first_50/checkpoints/checkpoint-1376'

    # sd3.5 medium
    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.merge_lora_path = '/openseg_blob/zhaoyaqi/workspace/coco80_grpo_counting_sd3_5_medium/output/sd_3_5_medium_base_data_8k_coco80_v5/checkpoint-3000'
    config.sample.num_steps = 20 # 40
    config.sample.eval_num_steps = 40
    # config.sample.guidance_scale = 4.5
    config.sample.guidance_scale = 7.0

    # 8 cards to start LLaVA Server
    config.resolution = 512
    config.sample.train_batch_size = 6
    config.sample.num_image_per_prompt = 12 # device_num = 4
    # config.sample.num_image_per_prompt = 3 # device_num = 1 

    config.sample.num_batches_per_epoch = 24
    config.sample.test_batch_size = 10 # This bs is a special design, the test set has a total of 2212, to make gpu_num*bs*n as close as possible to 2212, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2 # 24/2=12
    config.train.num_inner_epochs = 1
    # config.train.timestep_fraction = 0.99
    config.train.timestep_select_strategy = "first"
    config.train.timestep_fraction = 0.5
    config.train.reward_strict = True
    config.train.beta = 0.01
    config.sample.kl_reward = 0
    config.sample.global_std=True
    config.train.ema=True
    config.num_epochs = 100000
    # config.save_freq = 30 # epoch
    # config.eval_freq = 60
    config.save_freq = 10 # epoch
    config.eval_freq = 20
    config.save_dir = '/openseg_blob/zhaoyaqi/workspace/coco80_grpo_counting_sd3_5_medium/output/counting_coco80_20_step20_guidance_7_v5_reward_strict_first_50'
    config.reward_fn = {
        "geneval": 1.0,
    }
    
    config.prompt_fn = "geneval"

    config.per_prompt_stat_tracking = True
    return config

def geneval_sd3_counting_40_step20_reward_strict_first_50():
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/counting_40")
    config.run_name = "0717_counting_coco80_40_step20_guidance_7_v5_reward_strict_first_50_aigc12"
    config.resume_from = '/openseg_blob/zhaoyaqi/workspace/coco80_grpo_counting_sd3_5_medium/output/counting_coco80_40_step20_guidance_7_v5_reward_strict_first_50/checkpoints/checkpoint-1416'

    # sd3.5 medium
    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.merge_lora_path = '/openseg_blob/zhaoyaqi/workspace/coco80_grpo_counting_sd3_5_medium/output/sd_3_5_medium_base_data_8k_coco80_v5/checkpoint-3000'
    config.sample.num_steps = 20 # 40
    config.sample.eval_num_steps = 40
    # config.sample.guidance_scale = 4.5
    config.sample.guidance_scale = 7.0

    # 8 cards to start LLaVA Server
    config.resolution = 512
    config.sample.train_batch_size = 6
    config.sample.num_image_per_prompt = 12 # device_num = 4
    # config.sample.num_image_per_prompt = 3 # device_num = 1 

    config.sample.num_batches_per_epoch = 24
    config.sample.test_batch_size = 10 # This bs is a special design, the test set has a total of 2212, to make gpu_num*bs*n as close as possible to 2212, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2 # 24/2=12
    config.train.num_inner_epochs = 1
    # config.train.timestep_fraction = 0.99
    config.train.timestep_select_strategy = "first"
    config.train.timestep_fraction = 0.5
    config.train.reward_strict = True
    config.train.beta = 0.01
    config.sample.kl_reward = 0
    config.sample.global_std=True
    config.train.ema=True
    config.num_epochs = 100000
    # config.save_freq = 30 # epoch
    # config.eval_freq = 60
    config.save_freq = 10 # epoch
    config.eval_freq = 20
    config.save_dir = '/openseg_blob/zhaoyaqi/workspace/coco80_grpo_counting_sd3_5_medium/output/counting_coco80_40_step20_guidance_7_v5_reward_strict_first_50'
    config.reward_fn = {
        "geneval": 1.0,
    }
    
    config.prompt_fn = "geneval"

    config.per_prompt_stat_tracking = True
    return config

def resume_counting_debug():
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/counting")

    # sd3.5 medium
    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.sample.num_steps = 5 # 40
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale= 7.0

    # 8 cards to start LLaVA Server
    config.resolution = 512
    # config.sample.train_batch_size = 6
    config.sample.train_batch_size = 6 # 12*4=48, 单卡的image
    config.sample.num_image_per_prompt = 12 # 单卡有config.sample.train_batch_size/config.sample.num_image_per_prompt个prompt(2)
    config.sample.num_batches_per_epoch = 4 # 1个epoch有24个batch，每个batch内2个prompt，一个epoch是48个prompt 
    config.sample.test_batch_size = 14 # This bs is a special design, the test set has a total of 2212, to make gpu_num*bs*n as close as possible to 2212, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.
    config.resume_from = '/openseg_blob/zhaoyaqi/workspace/coco80_grpo_counting_sd3_5_medium/output/resume_test_acce/checkpoints/checkpoint-20'
    # config.train.lora_path = '/openseg_blob/zhaoyaqi/workspace/coco80_grpo_counting_sd3_5_medium/output/counting_coco80_10_step20_guidance_7_v5/checkpoints/checkpoint-1056/lora'
    
    
    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    config.train.beta = 0.01
    config.sample.kl_reward = 0
    config.sample.global_std=True
    config.train.ema=True
    config.num_epochs = 100000
    config.save_freq = 3 # epoch
    config.eval_freq = 60
    config.save_dir = '/openseg_blob/zhaoyaqi/workspace/coco80_grpo_counting_sd3_5_medium/output/resume_test_acce'
    config.reward_fn = {
        "geneval": 1.0,
        # "imagereward": 1.0,
        # "unifiedreward": 0.7,
    }
    
    config.prompt_fn = "geneval"

    config.per_prompt_stat_tracking = True
    return config


def get_config(name):
    return globals()[name]()
