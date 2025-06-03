import json
import random
from pathlib import Path

# 读取object names
def load_object_names(file_path):
    with open(file_path, 'r') as f:
        return [line.strip() for line in f.readlines()]

def needs_an(word):
    """判断单词是否需要使用'an'作为不定冠词"""
    # 元音音素开头的词需要'an'
    vowel_sounds = ['a', 'e', 'i', 'o', 'u']
    return word[0].lower() in vowel_sounds or word.lower().startswith(('hour', 'honest'))

def get_plural_form(word):
    """获取单词的复数形式"""
    # 特殊复数规则词典
    irregular_plurals = {
        'person': 'persons',  # 也可以是'people'，根据需求选择
        'mouse': 'mice',
        'knife': 'knives',
        'leaf': 'leaves',
        'sheep': 'sheep',
        'fish': 'fish',
        'computer mouse': 'computer mice',
        'skis': 'skis',
        'scissors': 'scissors',
        'series': 'series'
    }
    
    # 检查是否是特殊复数
    if word in irregular_plurals:
        return irregular_plurals[word]
        
    # 以-s, -ss, -sh, -ch, -x, -o结尾的词加-es
    if word.endswith(('s', 'ss', 'sh', 'ch', 'x', 'o')):
        return word + 'es'
    # 辅音字母+y结尾的词，变y为i加es
    elif word.endswith('y') and word[-2].lower() not in 'aeiou':
        return word[:-1] + 'ies'
    # 以-f或-fe结尾的词，变f为v加es
    elif word.endswith('f'):
        return word[:-1] + 'ves'
    elif word.endswith('fe'):
        return word[:-2] + 'ves'
    # 其他情况加-s
    else:
        return word + 's'

def get_article(word, count):
    """根据单词和数量返回正确的冠词或数量词"""
    if count > 1:
        return number_to_words(count)
    else:
        return 'an' if needs_an(word) else 'a'

def number_to_words(n):
    # 定义阿拉伯数字到字母的映射
    num_to_word = {
        1: "a", 2: "two", 3: "three", 4: "four", 5: "five", 
        6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten", 
        11: "eleven", 12: "twelve", 13: "thirteen", 14: "fourteen", 
        15: "fifteen", 16: "sixteen", 17: "seventeen", 18: "eighteen", 
        19: "nineteen", 20: "twenty", 30: "thirty", 40: "forty", 
        50: "fifty"
    }
    
    if n <= 20:
        # import pdb; pdb.set_trace()
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

prompt_list = [
    "A composition of {article} {object} scattered across a neutral gray background, each showing detailed textures and characteristic features with clean edges.",
    "A dynamic arrangement of {article} {object} randomly distributed on a soft gray backdrop, each precisely rendered with authentic details and materials.",
    "An organic display of {article} {object} dispersed naturally across a neutral gray background, each showcasing realistic textures and dimensional details.",
    "A free-flowing layout of {article} {object} spread throughout the frame against a gray backdrop, each captured with photographic precision and clear details.",
    "A spontaneous composition of {article} {object} distributed randomly against a gray background, each rendered with realistic materials and textures.",
    "An informal arrangement showing {article} {object} scattered organically on a neutral gray canvas, each depicted with authentic details and characteristic features.",
    "A natural distribution of {article} {object} spread across a gray background, each showing clear material textures and design details.",
    "A varied arrangement of {article} {object} randomly positioned against a clean gray backdrop, each rendered with precise details and realistic features.",
    "A casual composition featuring {article} {object} naturally dispersed on a neutral gray surface, each portrayed with authentic textures and clear design elements.",
    "A free-form presentation of {article} {object} scattered throughout the space on a soft gray background, each displaying realistic materials and precise details."
]

prompt_list_single = [
    "A striking {object} positioned centrally on a neutral gray background, highlighting its distinctive shape and form.",
    "A clear view of {article} {object} set against a soft gray backdrop, emphasizing its natural proportions.",
    "An elegant composition featuring {article} {object} placed thoughtfully on a neutral gray canvas.",
    "A minimalist presentation of {article} {object} against a clean gray background, capturing its essential characteristics.",
    "A refined portrayal of {article} {object} set on a subtle gray backdrop, showcasing its unique silhouette.",
    "A focused study of {article} {object} positioned deliberately on a neutral gray surface.",
    "A simple yet effective arrangement showing {article} {object} against a muted gray background.",
    "A carefully composed image of {article} {object} placed on a neutral gray canvas.",
    "A straightforward depiction of {article} {object} set against an understated gray backdrop.",
    "A clean representation of {article} {object} positioned on a neutral gray surface."
]
def generate_metadata(object_names, output_file, max_num):
    metadata_list = []
    
    for animal in object_names:
        # 获取正确的复数形式
        for number in range(1, max_num + 1):
            for index in range(len(prompt_list)):
                # 根据数量决定是否使用复数形式
                object_form = get_plural_form(animal) if number > 1 else animal
                # 获取正确的冠词或数量词
                article = get_article(animal, number)
                
                selected_prompt_list = prompt_list if number > 1 else prompt_list_single
                
                prompt = selected_prompt_list[index].format(
                    article=article,
                    object=object_form
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
    import os
    max_num = 20
    object_names_path = "/openseg_blob/zhaoyaqi/flow_grpo/reward-server/reward_server/object_names.txt"
    output_path = f"/openseg_blob/zhaoyaqi/flow_grpo/dataset/counting_{max_num}/metadata_1_{max_num}.jsonl"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 确保输出目录存在
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # 读取object names
    object_names = load_object_names(object_names_path)
    
    # 生成metadata
    generate_metadata(object_names, output_path, max_num)
    print(f"Generated metadata saved to {output_path}")

if __name__ == "__main__":
    main()

