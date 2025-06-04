# Get results of evaluation

import argparse
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


parser = argparse.ArgumentParser()
parser.add_argument("imagedir", type=str)
args = parser.parse_args()

# Load classnames

with open(os.path.join(os.path.dirname(__file__), "object_names.txt")) as cls_file:
    classnames = [line.strip() for line in cls_file]
    cls_to_idx = {"_".join(cls.split()):idx for idx, cls in enumerate(classnames)}

# Load results
base_dir = "/openseg_blob/zhaoyaqi/workspace/coco80_grpo_counting_sd3_5_medium/summary_scores"
os.makedirs(base_dir, exist_ok=True)
df = pd.read_json(os.path.join(base_dir.replace('summary_scores', 'output_eval'), args.imagedir, "results_the09.jsonl"), orient="records", lines=True)

# Measure overall success

print("Summary")
print("=======")
print(f"Total images: {len(df)}")
print(f"Total prompts: {len(df.groupby('metadata'))}")
print(f"% correct images: {df['correct'].mean():.2%}")
print(f"% correct prompts: {df.groupby('metadata')['correct'].any().mean():.2%}")
print()

# By group

task_scores = []

print("Task breakdown")
print("==============")
for tag, task_df in df.groupby('tag', sort=False):
    task_scores.append(task_df['correct'].mean())
    print(f"{tag:<16} = {task_df['correct'].mean():.2%} ({task_df['correct'].sum()} / {len(task_df)})")
print()

print(f"Overall score (avg. over tasks): {np.mean(task_scores):.5f}")

# 按count统计准确率
count_stats = {}
for _, row in df.iterrows():
    metadata = json.loads(row['metadata'])
    if 'include' in metadata:
        count = metadata['include'][0]['count']
        if count not in count_stats:
            count_stats[count] = {'correct': 0, 'total': 0}
        count_stats[count]['total'] += 1
        if row['correct']:
            count_stats[count]['correct'] += 1

# 计算每个count的准确率
counts = sorted(count_stats.keys())
accuracies = [count_stats[count]['correct'] / count_stats[count]['total'] for count in counts]

# 设置图表样式
plt.style.use('bmh')  # 使用内置的bmh样式
plt.figure(figsize=(12, 6))

# 创建渐变色
colors = ['#4B92DB']  # 使用单一的蓝色

# 绘制柱状图
bars = plt.bar(counts, accuracies, color=colors, alpha=0.8)
plt.grid(axis='y', linestyle='--', alpha=0.7, color='gray')

# 设置坐标轴
plt.xlabel('Required Object Count', fontsize=12, fontweight='bold')
plt.ylabel('Accuracy', fontsize=12, fontweight='bold')
plt.title('Accuracy by Required Object Count', fontsize=14, fontweight='bold', pad=20)

# 设置x轴刻度
plt.xticks(counts, fontsize=10)

# 设置y轴范围和刻度
plt.ylim(0, 1)
plt.yticks(np.arange(0, 1.1, 0.1), [f'{x:.0%}' for x in np.arange(0, 1.1, 0.1)], fontsize=10)

# 在柱子上添加数值标签（显示所有值）
for i, (count, acc) in enumerate(zip(counts, accuracies)):
    # 根据数值大小调整标签位置和字体大小
    if acc >= 0.1:
        y_offset = 0.01
        fontsize = 10
    else:
        y_offset = 0.005  # 较小的偏移量
        fontsize = 8  # 较小的字体
    
    plt.text(count, acc + y_offset, f'{acc:.1%}', 
            ha='center', va='bottom', fontsize=fontsize,
            color='black')

# 设置背景色和边框
plt.gca().set_facecolor('white')
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# 调整布局
plt.tight_layout()

# 保存图表
plot_path = os.path.join(base_dir, args.imagedir, "count_accuracy.png")
os.makedirs(os.path.dirname(plot_path), exist_ok=True)
plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

# Save results
output_path = os.path.join(base_dir, args.imagedir, "summary_scores.txt")
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, "w") as f:
    f.write(f"Total images: {len(df)}\n")
    f.write(f"Total prompts: {len(df.groupby('metadata'))}\n")
    f.write(f"% correct images: {df['correct'].mean():.2%}\n")
    f.write(f"% correct prompts: {df.groupby('metadata')['correct'].any().mean():.2%}\n")
    f.write(f"Task breakdown:\n")
    for tag, task_df in df.groupby('tag', sort=False):
        f.write(f"{tag:<16} = {task_df['correct'].mean():.2%} ({task_df['correct'].sum()} / {len(task_df)})\n")
    f.write(f"Overall score (avg. over tasks): {np.mean(task_scores):.5f}\n")
    f.write("\nAccuracy by Required Count:\n")
    for count in counts:
        stats = count_stats[count]
        f.write(f"Count {count}: {stats['correct']}/{stats['total']} = {stats['correct']/stats['total']:.2%}\n")
    
    # Calculate range accuracies
    def calculate_range_accuracy(count_stats, start, end):
        correct = sum(count_stats[i]['correct'] for i in range(start, end + 1) if i in count_stats)
        total = sum(count_stats[i]['total'] for i in range(start, end + 1) if i in count_stats)
        return correct / total if total > 0 else 0

    # Output range accuracies
    f.write("\nRange Accuracies:\n")
    ranges = [(1, 5), (1, 10), (1, 20)]
    for start, end in ranges:
        acc = calculate_range_accuracy(count_stats, start, end)
        f.write(f"Range {start}-{end}: {acc:.2%}\n")
        print(f"Range {start}-{end} accuracy: {acc:.2%}")

print(f"Results saved to {output_path}")
print(f"Count accuracy plot saved to {plot_path}")