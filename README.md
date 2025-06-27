# Integrating Flow-GRPO with Numerosity Control in Diffusion Models

## Project Description
This project is based on the open-source [Flow-GRPO](https://github.com/MiaZhao7708/flow_grpo/tree/main) codebase, with our modifications available at: [feature/v01_sd_grpo](https://github.com/MiaZhao7708/flow_grpo/tree/feature/v01_sd_grpo)

**Main improvements:**
1. Cold start training of the SD3.5-M model using a custom dataset
2. Design of a reward function specialized for the counting task
3. Optimization of timestep selection strategy for counting
4. Improved reward design for the counting scenario

## Project Structure

```
flow_grpo/
├── config/                     # Training configurations
│   ├── count_10_aba.py        # Configuration for counting experiments
│   └── dgx.py                 # Hyperparameter configurations
│
├── dataset/                    # Training prompt datasets
│
├── flow_grpo/                 # Core Flow-GRPO implementation
│   ├── diffusers_patch/       # SDE method implementations
│   └── ...                    # Reward score calculation modules
│
├── geneval/                   # GenEval benchmark evaluation code
│   ├── evaluation/            # Evaluation scripts and metrics
│   └── generation/            # Generation utilities
│
├── scripts/                   # Training scripts and configurations
│   ├── accelerate_configs/    # Multi-GPU acceleration configs
│   ├── multi_node/            # Multi-node training scripts
│   ├── single_node/           # Single-node training scripts
│   └── train_sd3.py          # Main SD3 training script
│
├── sde_diff/                  # SDE and ODE inference codebase
│
├── reward-server/             # Reward computation server
│   ├── reward_server/         # Server implementation
│   ├── mmdetection/           # Object detection integration
│   └── test/                  # Testing utilities
│
└── ...                        # Additional utilities and configs
```

**Key Directories:**
- **config/**: Contains training configurations and hyperparameters
- **dataset/**: Houses all training datasets, primarily focused on counting tasks
- **flow_grpo/**: Core implementation including reward score calculation and SDE methods
- **geneval/**: GenEval benchmark evaluation utilities
- **scripts/**: Training scripts and acceleration configurations
- **sde_diff/**: SDE and ODE inference implementations
- **reward-server/**: Online reward computation service

## Environment Setup

### Using Docker (Recommended)
We provide a pre-built Docker image with all required dependencies:

```bash
# Pull the Docker image
docker pull mia189/flow_grpo:v1

# Run the Docker container (full configuration)
docker run --gpus all \
    -it \
    --name flow_grpo \
    -p 6001:6001 \
    -p 6002:6002 \
    -p 6003:6003 \
    -p 6005:6005 \
    -p 6006:6006 \
    -p 8888:8888 \
    -v /path/to/your/workspace:/workspace \
    --workdir /workspace \
    mia189/flow_grpo:v1

# The container includes two environments:
# 1. flow_grpo: for model training
# 2. reward_server: for online reward computation

# Note: Replace /path/to/your/workspace with your actual working directory path.
```

### Manual Setup (Optional)
If you prefer not to use Docker, you can set up the environment manually:

1. Clone the repository and install dependencies:
```bash
git clone https://github.com/yifan123/flow_grpo.git
cd flow_grpo
conda create -n flow_grpo python=3.10.16
pip install -e .
```

2. Set up the GenEval Reward server:
   - Create a new conda environment
   - Follow the instructions in [reward-server](https://github.com/yifan123/reward-server) to install dependencies

## Training

### 1. Start the Reward Server
First, start the reward server by following the instructions in the [reward-server](https://github.com/yifan123/reward-server) repository to launch the GenEval reward service.

### 2. Train the Model
Use the following command to start training:

```bash
# Activate the environment and launch multi-GPU training
conda activate flow_grpo
accelerate launch \
    --config_file scripts/accelerate_configs/multi_gpu.yaml \
    --num_processes=4 \
    --main_process_port 29503 \
    scripts/train_sd3.py \
    --config config/count_10_aba.py:geneval_sd3_counting_10_step20_reward_strict_first_50
```

**Configuration details:**
- Config file: `config/count_10_aba.py`
- Config name: `geneval_sd3_counting_10_step20_reward_strict_first_50`
- Number of GPUs: 4 (adjustable via `--num_processes`)
- Main process port: 29503 (can be changed if needed)

## Important Hyperparameters
You can adjust hyperparameters in `config/dgx.py`. Based on empirical results, we recommend:

- `config.sample.train_batch_size * num_gpu / config.sample.num_image_per_prompt * config.sample.num_batches_per_epoch = 48`
- For example: `group_number=48`, `group_size=24`
- Set `config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch // 2`

These settings have shown good performance in practice.

## Inference

After training, you can use the inference demo to generate images with your trained LoRA weights.

### Usage

```bash
# Basic usage with random test prompt
python inference_demo.py --lora_path /path/to/your/lora/weights

# Using custom prompt
python inference_demo.py \
    --lora_path /path/to/your/lora/weights \
    --custom_prompt "A composition of five cats scattered across a neutral gray background"

# Full parameter specification
python inference_demo.py \
    --lora_path /path/to/your/lora/weights \
    --model_name stabilityai/stable-diffusion-3.5-large \
    --num_images 4 \
    --height 512 \
    --width 512 \
    --num_steps 20 \
    --guidance_scale 7.5 \
    --seed 42 \
    --output_dir ./my_outputs
```

### Parameters

- `--lora_path`: **Required**. Path to LoRA weights (can be a checkpoint directory or direct LoRA weights directory)
- `--model_name`: Base model name (default: `stabilityai/stable-diffusion-3.5-large`)
- `--custom_prompt`: Use a custom prompt instead of randomly selecting from test data
- `--num_images`: Number of images to generate (default: 4)
- `--height`, `--width`: Image dimensions (default: 512x512)
- `--num_steps`: Number of inference steps (default: 20)
- `--guidance_scale`: Classifier-free guidance scale (default: 7.5)
- `--seed`: Random seed for reproducibility (default: 42)
- `--output_dir`: Output directory for generated images (default: `./demo_outputs`)
- `--device_id`: GPU device ID (default: 0)