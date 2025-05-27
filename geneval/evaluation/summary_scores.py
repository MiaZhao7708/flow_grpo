# Get results of evaluation

import argparse
import os

import numpy as np
import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument("imagedir", type=str)
args = parser.parse_args()

# Load classnames

with open(os.path.join(os.path.dirname(__file__), "object_names.txt")) as cls_file:
    classnames = [line.strip() for line in cls_file]
    cls_to_idx = {"_".join(cls.split()):idx for idx, cls in enumerate(classnames)}

# Load results
base_dir = "/openseg_blob/zhaoyaqi/workspace/flow_grpo/geneval/summary_scores"
os.makedirs(base_dir, exist_ok=True)
df = pd.read_json(os.path.join(base_dir.replace('summary_scores', 'output_eval'), args.imagedir, "results.jsonl"), orient="records", lines=True)

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

print(f"Results saved to {output_path}")