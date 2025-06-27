from huggingface_hub import HfApi
import os

# 初始化 API 对象
api = HfApi()

# 模型路径和对应的目标文件夹名称
model_paths = {
    "strict_first": "/openseg_blob/zhaoyaqi/workspace/coco80_grpo_counting_sd3_5_medium/output/counting_coco80_10_step20_guidance_7_v5_reward_strict_first_50/checkpoints/checkpoint-1336/lora",
    "relative_first": "/openseg_blob/zhaoyaqi/workspace/coco80_grpo_counting_sd3_5_medium/output/counting_coco80_10_step20_guidance_7_v5_timestep_select_first_50/checkpoints/checkpoint-1336/lora",
    "strict_random": "/openseg_blob/zhaoyaqi/workspace/coco80_grpo_counting_sd3_5_medium/output/counting_coco80_10_step20_guidance_7_v5_reward_strict/checkpoints/checkpoint-1336/lora",
    "relative_random": "/openseg_blob/zhaoyaqi/workspace/coco80_grpo_counting_sd3_5_medium/output/counting_coco80_10_step20_guidance_7_v5_timestep_select_random_50/checkpoints/checkpoint-1336/lora"
}

# 创建新的仓库名称
repo_id = "MiaTiancai/grpo-counting-model"

# 创建仓库（如果不存在）
api.create_repo(repo_id=repo_id, exist_ok=True)

# 上传 README.md
print("Uploading README.md...")
api.upload_file(
    path_or_fileobj="READ_model.md",
    path_in_repo="README.md",
    repo_id=repo_id,
    commit_message="Add model card"
)

# 上传每个模型的文件
for variant_name, model_path in model_paths.items():
    print(f"\nUploading {variant_name} model...")
    for root, dirs, files in os.walk(model_path):
        for file in files:
            local_path = os.path.join(root, file)
            # 将文件放在对应的变体目录下
            relative_path = os.path.join(variant_name, os.path.relpath(local_path, model_path))
            
            print(f"Uploading {relative_path}...")
            api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=relative_path,
                repo_id=repo_id,
                commit_message=f"Upload {relative_path}"
            )

print("Upload completed!") 