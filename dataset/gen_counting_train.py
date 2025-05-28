import json
import random
from pathlib import Path

# 读取object names
def load_object_names(file_path):
    with open(file_path, 'r') as f:
        return [line.strip() for line in f.readlines()]

def get_plural_form(word):
    # 特殊复数规则
    if word == "scissors":  # 已经是复数形式
        return word
    elif word == "mouse":  # computer mouse的特殊情况
        return "mice"
    elif word.endswith(("sh", "ch")):
        return word + "es"
    elif word.endswith("s"):  # glass -> glasses
        return word + "es"
    elif word.endswith("fish"):  # 鱼类保持不变
        return word
    else:
        return word + "s"

# 提示模板
prompt_list = [
    "A composition of {gt_count} identical {animal_str}s scattered across a neutral gray background, each showing detailed textures and characteristic features with clean edges.",
    "A dynamic arrangement of {gt_count} {animal_str}s randomly distributed on a soft gray backdrop, each precisely rendered with authentic details and materials.",
    "An organic display of {gt_count} {animal_str}s dispersed naturally across a neutral gray background, each showcasing realistic textures and dimensional details.",
    "A free-flowing layout of {gt_count} identical {animal_str}s spread throughout the frame against a gray backdrop, each captured with photographic precision and clear details.",
    "A spontaneous composition of {gt_count} {animal_str}s distributed randomly against a gray background, each rendered with realistic materials and textures.",
    "An informal arrangement showing {gt_count} {animal_str}s scattered organically on a neutral gray canvas, each depicted with authentic details and characteristic features.",
    "A natural distribution of {gt_count} {animal_str}s spread across a gray background, each showing clear material textures and design details.",
    "A varied arrangement of {gt_count} identical {animal_str}s randomly positioned against a clean gray backdrop, each rendered with precise details and realistic features.",
    "A casual composition featuring {gt_count} {animal_str}s naturally dispersed on a neutral gray surface, each portrayed with authentic textures and clear design elements.",
    "A free-form presentation of {gt_count} {animal_str}s scattered throughout the space on a soft gray background, each displaying realistic materials and precise details."
]
def generate_metadata(object_names, output_file, samples_per_object=5):
    metadata_list = []
    
    for animal in object_names:
        # 获取正确的复数形式
        animal_plural = get_plural_form(animal)
        
        for number in range(1, 6):
            for prompt_template in prompt_list:
                animal_str = animal_plural if number > 1 else animal
                prompt = prompt_template.format(
                    gt_count=number,
                    animal_str=animal_str
                )
                metadata = {
                    "tag": "counting",
                    "include": [{"class": animal, "count": number}],
                    "exclude": [{"class": animal, "count": number + 1}],
                    "prompt": prompt,
                    "class": animal
                }
            
                metadata_list.append(metadata)
    import pdb; pdb.set_trace()
    # with open(output_file, 'w') as f:
    #     for metadata in metadata_list:
    #         f.write(json.dumps(metadata) + "\\n")

    with open(output_file, "w") as f:
        for metadata in metadata_list:
            json_line = json.dumps(metadata, ensure_ascii=False)
            f.write(json_line + "\n")

def main():
    # 设置路径
    object_names_path = "/openseg_blob/zhaoyaqi/flow_grpo/reward-server/reward_server/object_names.txt"
    output_path = "/openseg_blob/zhaoyaqi/flow_grpo/dataset/counting/metadata_1_5.jsonl"
    
    # 确保输出目录存在
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # 读取object names
    object_names = load_object_names(object_names_path)
    
    # 生成metadata
    generate_metadata(object_names, output_path)
    print(f"Generated metadata saved to {output_path}")

if __name__ == "__main__":
    main()

