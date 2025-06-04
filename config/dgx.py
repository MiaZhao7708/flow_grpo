import ml_collections
import imp
import os

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

def general_ocr_sd3():
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/ocr")

    # sd3.5 medium
    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.sample.num_steps = 10
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale=4.5

    # # 8*A800
    # config.resolution = 512
    # config.sample.train_batch_size = 12
    # config.sample.num_image_per_prompt = 24
    # config.sample.num_batches_per_epoch = 12
    # config.sample.test_batch_size = 16 # 11 is a special design, the test set has a total of 1018, to make 8*16*n as close as possible to 1018, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    # 1 A800 This is just to ensure it runs quickly on a single GPU, though the performance may degrade.
    # If using 8 GPUs, please comment out this section and use the 8-GPU configuration above instead.
    config.resolution = 512
    config.sample.train_batch_size = 12
    config.sample.num_image_per_prompt = 6
    config.sample.num_batches_per_epoch = 12
    config.sample.test_batch_size = 16 # 11 is a special design, the test set has a total of 1018, to make 8*16*n as close as possible to 1018, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2 # 12/2=6
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    # kl loss
    config.train.beta = 0.001
    # kl reward
    config.sample.kl_reward = 0
    config.sample.global_std=True
    config.train.ema=True
    config.num_epochs = 100000
    config.save_freq = 60 # epoch
    config.eval_freq = 60
    config.save_dir = 'logs/ocr/sd3.5-M'
    config.reward_fn = {
        # "geneval": 1.0,
        "ocr": 1.0,
        # "unifiedreward": 0.7,
    }
    
    config.prompt_fn = "general_ocr"

    config.per_prompt_stat_tracking = True
    return config

def geneval_sd3():
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/geneval")

    # sd3.5 medium
    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.sample.num_steps = 10 # 40
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale=4.5

    # 8 cards to start LLaVA Server
    config.resolution = 512
    config.sample.train_batch_size = 6
    # config.sample.train_batch_size = 12 # 
    # config.sample.num_image_per_prompt = 24
    config.sample.num_image_per_prompt = 12
    config.sample.num_batches_per_epoch = 24
    config.sample.test_batch_size = 14 # This bs is a special design, the test set has a total of 2212, to make gpu_num*bs*n as close as possible to 2212, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

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
    config.save_freq = 20 # epoch
    config.eval_freq = 20
    config.save_dir = '/openseg_blob/zhaoyaqi/workspace/flow_grpo/logs/geneval/sd3.5-M'
    config.reward_fn = {
        "geneval": 1.0,
        # "imagereward": 1.0,
        # "unifiedreward": 0.7,
    }
    
    config.prompt_fn = "geneval"

    config.per_prompt_stat_tracking = True
    return config


def geneval_sd3_counting_5_step20():
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/counting_5")

    # sd3.5 medium
    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.merge_lora_path = '/openseg_blob/zhaoyaqi/workspace/coco80_grpo_counting_sd3_5_medium/output/sd_3_5_medium_base_data_8k_coco80_v5/checkpoint-1000'
    config.sample.num_steps = 20 # 40
    config.sample.eval_num_steps = 40
    # config.sample.guidance_scale = 4.5
    config.sample.guidance_scale = 7.0

    # 8 cards to start LLaVA Server
    config.resolution = 512
    config.sample.train_batch_size = 6
    # config.sample.train_batch_size = 12 # 
    # config.sample.num_image_per_prompt = 24
    # care!!!!!
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
    # config.save_dir = '/openseg_blob/zhaoyaqi/workspace/flow_grpo/logs/counting_coco80_5_step10_guidance_7/sd3.5-M'
    config.save_dir = '/openseg_blob/zhaoyaqi/workspace/coco80_grpo_counting_sd3_5_medium/output/counting_coco80_5_step20_guidance_7_v5'
    config.reward_fn = {
        "geneval": 1.0,
        # "imagereward": 1.0,
        # "unifiedreward": 0.7,
    }
    
    config.prompt_fn = "geneval"

    config.per_prompt_stat_tracking = True
    return config

def geneval_sd3_counting_10_step20():
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/counting")

    # sd3.5 medium
    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.merge_lora_path = '/openseg_blob/zhaoyaqi/workspace/coco80_grpo_counting_sd3_5_medium/output/sd_3_5_medium_base_data_8k_coco80_v5/checkpoint-1000'
    config.sample.num_steps = 20 # 40
    config.sample.eval_num_steps = 40
    # config.sample.guidance_scale = 4.5
    config.sample.guidance_scale = 7.0

    # 8 cards to start LLaVA Server
    config.resolution = 512
    config.sample.train_batch_size = 6
    # config.sample.train_batch_size = 12 # 
    # config.sample.num_image_per_prompt = 24
    # care!!!!!
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
    # config.save_dir = '/openseg_blob/zhaoyaqi/workspace/flow_grpo/logs/counting_coco80_10_step20_guidance_7/sd3.5-M'
    config.save_dir = '/openseg_blob/zhaoyaqi/workspace/coco80_grpo_counting_sd3_5_medium/output/counting_coco80_10_step20_guidance_7_v5'
    config.reward_fn = {
        "geneval_step20": 1.0,
        # "imagereward": 1.0,
        # "unifiedreward": 0.7,
    }
    
    config.prompt_fn = "geneval"

    config.per_prompt_stat_tracking = True
    return config



def geneval_sd3_counting_20_step20():
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/counting_20")

    # sd3.5 medium
    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.merge_lora_path = '/openseg_blob/zhaoyaqi/workspace/coco80_grpo_counting_sd3_5_medium/output/sd_3_5_medium_base_data_8k_coco80_v5/checkpoint-1000'
    config.sample.num_steps = 20 # 40
    config.sample.eval_num_steps = 40
    # config.sample.guidance_scale = 4.5
    config.sample.guidance_scale = 7.0

    # 8 cards to start LLaVA Server
    config.resolution = 512
    config.sample.train_batch_size = 6
    # config.sample.train_batch_size = 12 # 
    # config.sample.num_image_per_prompt = 24
    # care!!!!!
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
    # config.save_dir = '/openseg_blob/zhaoyaqi/workspace/flow_grpo/logs/counting_coco80_10_step10_guidance_7/sd3.5-M'
    config.save_dir = '/openseg_blob/zhaoyaqi/workspace/coco80_grpo_counting_sd3_5_medium/output/counting_coco80_20_step20_guidance_7_v5'
    config.reward_fn = {
        "geneval_debug": 1.0,
        # "imagereward": 1.0,
        # "unifiedreward": 0.7,
    }
    
    config.prompt_fn = "geneval"

    config.per_prompt_stat_tracking = True
    return config


def geneval_sd3_counting_15_step20():
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/counting_15")

    # sd3.5 medium
    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.merge_lora_path = '/openseg_blob/zhaoyaqi/workspace/coco80_grpo_counting_sd3_5_medium/output/sd_3_5_medium_base_data_8k_coco80_v5/checkpoint-1000'
    config.sample.num_steps = 20 # 40
    config.sample.eval_num_steps = 40
    # config.sample.guidance_scale = 4.5
    config.sample.guidance_scale = 7.0

    # 8 cards to start LLaVA Server
    config.resolution = 512
    config.sample.train_batch_size = 6
    # config.sample.train_batch_size = 12 # 
    # config.sample.num_image_per_prompt = 24
    # care!!!!!
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
    # config.save_dir = '/openseg_blob/zhaoyaqi/workspace/flow_grpo/logs/counting_coco80_15_step20_guidance_7/sd3.5-M'
    config.save_dir = '/openseg_blob/zhaoyaqi/workspace/coco80_grpo_counting_sd3_5_medium/output/counting_coco80_15_step20_guidance_7_v5'
    config.reward_fn = {
        "geneval_debug": 1.0,
        # "imagereward": 1.0,
        # "unifiedreward": 0.7,
    }
    
    config.prompt_fn = "geneval"

    config.per_prompt_stat_tracking = True
    return config


def geneval_sd3_counting_6_10_step20():
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/counting_6_10")

    # sd3.5 medium
    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.merge_lora_path = '/openseg_blob/zhaoyaqi/workspace/coco80_grpo_counting_sd3_5_medium/output/sd_3_5_medium_base_data_8k_coco80_v5/checkpoint-1000'
    config.sample.num_steps = 20 # 40
    config.sample.eval_num_steps = 40
    # config.sample.guidance_scale = 4.5
    config.sample.guidance_scale = 7.0

    # 8 cards to start LLaVA Server
    config.resolution = 512
    config.sample.train_batch_size = 6
    # config.sample.train_batch_size = 12 # 
    # config.sample.num_image_per_prompt = 24
    # care!!!!!
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
    # config.save_dir = '/openseg_blob/zhaoyaqi/workspace/flow_grpo/logs/counting_coco80_15_step20_guidance_7/sd3.5-M'
    config.save_dir = '/openseg_blob/zhaoyaqi/workspace/coco80_grpo_counting_sd3_5_medium/output/counting_coco80_6_10_step20_guidance_7_v5'
    config.reward_fn = {
        "geneval": 1.0,
        # "imagereward": 1.0,
        # "unifiedreward": 0.7,
    }
    
    config.prompt_fn = "geneval"

    config.per_prompt_stat_tracking = True
    return config

def geneval_sd3_debug():
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/geneval")

    # sd3.5 medium
    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.sample.num_steps = 10 # 40
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale=4.5

    # 8 cards to start LLaVA Server
    config.resolution = 512
    # config.sample.train_batch_size = 6
    config.sample.train_batch_size = 6 # 12*4=48, 单卡的image
    config.sample.num_image_per_prompt = 12 # 单卡有config.sample.train_batch_size/config.sample.num_image_per_prompt个prompt(2)
    config.sample.num_batches_per_epoch = 24 # 1个epoch有24个batch，每个batch内2个prompt，一个epoch是48个prompt 
    config.sample.test_batch_size = 14 # This bs is a special design, the test set has a total of 2212, to make gpu_num*bs*n as close as possible to 2212, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    config.train.beta = 0.01
    config.sample.kl_reward = 0
    config.sample.global_std=True
    config.train.ema=True
    config.num_epochs = 100000
    config.save_freq = 60 # epoch
    config.eval_freq = 60
    config.save_dir = '/openseg_blob/zhaoyaqi/workspace/flow_grpo/logs/geneval/sd3.5-M'
    config.reward_fn = {
        "geneval_debug": 1.0,
        # "imagereward": 1.0,
        # "unifiedreward": 0.7,
    }
    
    config.prompt_fn = "geneval"

    config.per_prompt_stat_tracking = True
    return config

def pickscore_sd3():
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/pickscore")

    # sd3.5 medium
    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.sample.num_steps = 5 # 40
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale=4.5

    # 8*A800
    config.resolution = 512
    config.sample.train_batch_size = 12
    config.sample.num_image_per_prompt = 24
    config.sample.num_batches_per_epoch = 12
    config.sample.test_batch_size = 16 # The test set has a total of 2048

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    config.train.beta = 0    
    config.train.sft=0.0
    config.train.sft_batch_size=3
    config.sample.kl_reward = 0
    config.sample.global_std=True
    config.train.ema=True
    config.num_epochs = 100000
    config.save_freq = 60 # epoch
    config.eval_freq = 60
    config.save_dir = 'logs/pickscore/sd3.5-M'
    config.reward_fn = {
        # "geneval": 1.0,
        "pickscore": 1.0,
        # "unifiedreward": 0.7,
    }
    
    config.prompt_fn = "general_ocr"

    config.per_prompt_stat_tracking = True
    return config


def get_config(name):
    return globals()[name]()
