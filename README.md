# Integrating Flow-GRPO with Numerosity Control in Diffusion Models

## Project Description
This project is based on the open-source [Flow-GRPO](https://github.com/MiaZhao7708/flow_grpo/tree/main) codebase, with our modifications available at: [feature/v01_sd_grpo](https://github.com/MiaZhao7708/flow_grpo/tree/feature/v01_sd_grpo)

**Main improvements:**
1. Cold start training of the SD3.5-M model using a custom dataset
2. Design of a reward function specialized for the counting task
3. Optimization of timestep selection strategy for counting
4. Improved reward design for the counting scenario

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