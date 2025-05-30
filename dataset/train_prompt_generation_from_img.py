import json
import random
import re
import os
import random
from tqdm import tqdm

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

def get_animal_from_id(image_id):
    # 从图片ID中提取动物名称和数量
    # 例如: "42great_white_shark_00064.png" -> ("great white shark", 42)
    # 使用正则表达式匹配开头的数字
    match = re.match(r'^(\d+)', image_id)
    if match:
        gt_count = int(match.group(1))
    else:
        gt_count = 0
        
    # 提取动物名称
    animal_name = re.sub(r'^\d+', '', image_id)
    parts = animal_name.split('_')[:-1]
    animal_name = ' '.join(parts)
    
    return animal_name.lower(), gt_count

def get_animal_from_id_count20(image_id):
    # 从图片ID中提取动物名称
    # 例如: "42great_white_shark_00064.png" -> "great white shark"
    animal_name = image_id.split('_20')[0]
    animal_name = animal_name.replace('_', ' ')
    return animal_name.lower()

# image_folder = '/openseg_blob/zhaoyaqi/Count-FLUX/playground/results/one_animal_random_layout_250animals_50number_40layout_512px_500k'
image_folder = '/openseg_blob/zhaoyaqi/Count-FLUX/playground/results/one_object_random_layout_coco79_grpo_512px_8k_refine_v2'
animals = os.listdir(image_folder)
animals = [animal for animal in animals if animal.endswith('.png')]
output_path = "/openseg_blob/zhaoyaqi/Count-FLUX/playground/data/one_object_random_layout_coco79_grpo_512px_8k_refine_v4_data.json"
animal_names = []
data = []
for animal in tqdm(animals):
    # image_path = os.path.join(image_folder, animal)
    animal_name, gt_count = get_animal_from_id(animal)
    # 根据数量决定是否使用复数形式
    object_form = get_plural_form(animal_name) if gt_count > 1 else animal_name
    # 获取正确的冠词或数量词
    article = get_article(animal_name, gt_count)
    
    if object_form not in animal_names:
        animal_names.append(object_form)
    
    prompt = random.choice(prompt_list).format(
        article=article,
        object=object_form
    )
    
    item = {
        "id": animal,
        "caption": prompt,
        "animal": animal_name,
        "label": gt_count-1
    }
    data.append(item)

count_dict = {}
for item in data:
    if item['label'] not in count_dict:
        count_dict[item['label']] = 0
    count_dict[item['label']] += 1

import pdb; pdb.set_trace()
with open(output_path, 'w') as f:
    json.dump(data, f, indent=4)
