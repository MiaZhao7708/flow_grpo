from flask import Flask, render_template, send_from_directory
import json
import os
import re
import sys
sys.path.append('/openseg_blob/zhaoyaqi/flow_grpo')
from sas_key import add_prefix_suffix

app = Flask(__name__)

# 添加静态文件路径配置
STATIC_PATH = '/openseg_blob/zhaoyaqi/flow_grpo/sde_diff'

def load_metadata():
    prompts = []
    with open('/openseg_blob/zhaoyaqi/flow_grpo/dataset/geneval/test_metadata.jsonl', 'r') as f:
        for line in f:
            data = json.loads(line)
            prompts.append(data['prompt'])
    return prompts

def get_max_sample_index():
    pattern = re.compile(r'sample_(\d+)_image_\d+_(sde|ode)\.png')
    max_index = -1
    for filename in os.listdir(STATIC_PATH):
        match = pattern.match(filename)
        if match:
            sample_index = int(match.group(1))
            max_index = max(max_index, sample_index)
    return max_index + 1

def generate_static_html():
    with app.app_context():
        prompts = load_metadata()
        images_num = get_max_sample_index() - 1
        selected_prompts = prompts[:images_num]
        
        # 创建展示数据
        display_data = []
        for i, prompt in enumerate(selected_prompts):
            data = {
                'prompt': prompt,
                'sample_index': i,
                'sde_images': [add_prefix_suffix(f'/openseg_blob/zhaoyaqi/flow_grpo/sde_diff/sample_{i:02d}_image_{j:02d}_sde.png') for j in range(5)],
                'ode_images': [add_prefix_suffix(f'/openseg_blob/zhaoyaqi/flow_grpo/sde_diff/sample_{i:02d}_image_{j:02d}_ode.png') for j in range(5)]
            }
            display_data.append(data)
        
        # 确保html目录存在
        os.makedirs('./html', exist_ok=True)
        
        # 渲染并保存HTML
        html_content = render_template('index.html', display_data=display_data)
        with open('./html/sde_ode_diff.html', 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print("HTML文件已生成在 ./html/sde_ode_diff.html")

if __name__ == '__main__':
    generate_static_html() 