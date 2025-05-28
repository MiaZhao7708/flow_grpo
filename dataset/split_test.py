import json
import random
from collections import defaultdict
from pathlib import Path

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def save_jsonl(data, file_path):
    with open(file_path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

def split_dataset(input_file, train_output, test_output, test_size=500):
    # 读取所有数据
    all_data = load_jsonl(input_file)
    
    # 按照count分组
    count_groups = defaultdict(list)
    for item in all_data:
        count = item['include'][0]['count']
        if 1 <= count <= 10:  # 只关注1-10的数据
            count_groups[count].append(item)
    
    # 计算每个数字需要多少个样本
    samples_per_count = test_size // 10  # 每个数字50个样本
    
    # 选择测试集数据
    test_data = []
    train_data = all_data.copy()  # 首先复制所有数据
    
    # 对于每个数字1-10
    for count in range(1, 11):
        if count in count_groups:
            # 随机选择指定数量的样本
            count_samples = count_groups[count]
            if len(count_samples) >= samples_per_count:
                selected_samples = random.sample(count_samples, samples_per_count)
                test_data.extend(selected_samples)
                
                # 从训练集中移除选中的测试样本
                for sample in selected_samples:
                    train_data.remove(sample)
            else:
                print(f"Warning: Not enough samples for count {count}. Only {len(count_samples)} available.")
    
    # 保存数据集
    save_jsonl(train_data, train_output)
    save_jsonl(test_data, test_output)
    
    # 打印统计信息
    print(f"Total data: {len(all_data)}")
    print(f"Train set size: {len(train_data)}")
    print(f"Test set size: {len(test_data)}")
    print("\\nTest set distribution:")
    test_distribution = defaultdict(int)
    for item in test_data:
        count = item['include'][0]['count']
        test_distribution[count] += 1
    for count in range(1, 11):
        print(f"Count {count}: {test_distribution[count]} samples")

def main():
    input_file = "/openseg_blob/zhaoyaqi/flow_grpo/dataset/counting/metadata_1_10.jsonl"
    train_output = "/openseg_blob/zhaoyaqi/flow_grpo/dataset/counting/train_metadata.jsonl"
    test_output = "/openseg_blob/zhaoyaqi/flow_grpo/dataset/counting/test_metadata.jsonl"
    
    # 创建输出目录
    Path(train_output).parent.mkdir(parents=True, exist_ok=True)
    
    # 划分数据集
    split_dataset(input_file, train_output, test_output)

if __name__ == "__main__":
    main()
