import os
import json
import numpy as np
import matplotlib.pyplot as plt

def get_folder_stats(folder_path):
    files = sorted([f for f in os.listdir(folder_path) if f.endswith('.json') and f != 'process_record_1.json'])
    stats = {i: {'load_time': [], 'preprocess_time': [], 'embed_time': [], 'prefill_time': [], 'decode_time': [], 'image_time': []} for i in range(5)}
    for file in files:
        with open(os.path.join(folder_path, file), 'r') as f:
            data = json.load(f)
            records = data.get('records', [])
            for i, rec in enumerate(records):
                if not rec or not isinstance(rec, dict):
                    continue
                for key in ['load_time', 'preprocess_time', 'embed_time', 'prefill_time', 'decode_time', 'image_time']:
                    if key in rec:
                        stats[i][key].append(rec[key])
    # 计算平均值
    avg_stats = {}
    for i in range(5):
        avg_stats[i] = {k: np.mean(v) if v else 0 for k, v in stats[i].items()}
    return avg_stats

def collect_all_stats(base_dir):
    all_stats = {}
    for folder in os.listdir(base_dir):
        if folder.startswith('image_size') and folder.endswith('single_engine'):
            image_size = int(folder.split('_')[1][4:])
            # # 只保留batch size在192到384之间的数据
            # if 192 <= image_size <= 384:
            #     avg_stats = get_folder_stats(os.path.join(base_dir, folder))
            #     all_stats[batch_size] = avg_stats
            avg_stats = get_folder_stats(os.path.join(base_dir, folder))
            all_stats[image_size] = avg_stats
    return all_stats

def merge_gpu_cpu_stats(gpu_stats, cpu_stats):
    merged = {}
    for image_size in sorted(set(gpu_stats.keys()) | set(cpu_stats.keys())):
        merged[image_size] = {
            'gpu': gpu_stats.get(image_size, {}),
            'cpu': cpu_stats.get(image_size, {})
        }
    return merged

# 新的画图函数

def plot_bar_per_round_stacked(merged_stats, round_idx, save_path):
    image_sizes = sorted(merged_stats.keys())
    x_labels = []
    image_time_vals = []
    image_time_cpu_vals = []
    prefill_time_vals = []
    prefill_time_cpu_vals = []
    decode_time_vals = []
    decode_time_cpu_vals = []
    # 堆叠部分
    load_vals = []
    preprocess_vals = []
    embed_vals = []
    load_cpu_vals = []
    preprocess_cpu_vals = []
    embed_cpu_vals = []
    for image_size in image_sizes:
        # GPU
        gpu = merged_stats[image_size]['gpu'].get(round_idx, {})
        image_time_vals.append(gpu.get('image_time', 0))
        load_vals.append(gpu.get('load_time', 0))
        preprocess_vals.append(gpu.get('preprocess_time', 0))
        embed_vals.append(gpu.get('embed_time', 0))
        prefill_time_vals.append(gpu.get('prefill_time', 0))
        decode_time_vals.append(gpu.get('decode_time', 0))
        # CPU
        cpu = merged_stats[image_size]['cpu'].get(round_idx, {})
        image_time_cpu_vals.append(cpu.get('image_time', 0))
        load_cpu_vals.append(cpu.get('load_time', 0))
        preprocess_cpu_vals.append(cpu.get('preprocess_time', 0))
        embed_cpu_vals.append(cpu.get('embed_time', 0))
        prefill_time_cpu_vals.append(cpu.get('prefill_time', 0))
        decode_time_cpu_vals.append(cpu.get('decode_time', 0))
    # x轴
    n = len(image_sizes)
    x = np.arange(n * 6)  # 每个batch size 6个柱子
    bar_width = 0.8
    # 组织数据
    all_vals = []
    all_labels = []
    for i, image_size in enumerate(image_sizes):
        # image_time (stacked)
        all_labels.append(f'image_size{image_size}_image_time')
        all_vals.append((load_vals[i], preprocess_vals[i], embed_vals[i], image_time_vals[i]))
        # image_time_cpu (stacked)
        all_labels.append(f'image_size{image_size}_image_time_cpu')
        all_vals.append((load_cpu_vals[i], preprocess_cpu_vals[i], embed_cpu_vals[i], image_time_cpu_vals[i]))
        # prefill_time
        all_labels.append(f'image_size{image_size}_prefill_time')
        all_vals.append(prefill_time_vals[i])
        # prefill_time_cpu
        all_labels.append(f'image_size{image_size}_prefill_time_cpu')
        all_vals.append(prefill_time_cpu_vals[i])
        # decode_time
        all_labels.append(f'image_size{image_size}_decode_time')
        all_vals.append(decode_time_vals[i])
        # decode_time_cpu
        all_labels.append(f'image_size{image_size}_decode_time_cpu')
        all_vals.append(decode_time_cpu_vals[i])
    # 画图
    plt.figure(figsize=(max(12, len(all_labels)*0.4), 6))
    idx = 0
    for i in range(len(image_sizes)):
        # image_time (stacked)
        plt.bar(idx, all_vals[idx][0], bar_width, color='#4C72B0', label='load_time' if i==0 else "")
        plt.bar(idx, all_vals[idx][1], bar_width, bottom=all_vals[idx][0], color='#55A868', label='preprocess_time' if i==0 else "")
        plt.bar(idx, all_vals[idx][2], bar_width, bottom=all_vals[idx][0]+all_vals[idx][1], color='#C44E52', label='embed_time' if i==0 else "")
        # 画一个透明的bar用于显示总高度（image_time）
        plt.bar(idx, all_vals[idx][3], bar_width, fill=False, edgecolor='black', linewidth=1, label='image_time (total)' if i==0 else "")
        # 标注image_time总值
        plt.text(idx, all_vals[idx][3], f'{all_vals[idx][3]:.2f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
        idx += 1
        # image_time_cpu (stacked)
        plt.bar(idx, all_vals[idx][0], bar_width, color='#4C72B0', hatch='//', label='load_time_cpu' if i==0 else "")
        plt.bar(idx, all_vals[idx][1], bar_width, bottom=all_vals[idx][0], color='#55A868', hatch='//', label='preprocess_time_cpu' if i==0 else "")
        plt.bar(idx, all_vals[idx][2], bar_width, bottom=all_vals[idx][0]+all_vals[idx][1], color='#C44E52', hatch='//', label='embed_time_cpu' if i==0 else "")
        plt.bar(idx, all_vals[idx][3], bar_width, fill=False, edgecolor='black', linewidth=1, hatch='//', label='image_time_cpu (total)' if i==0 else "")
        plt.text(idx, all_vals[idx][3], f'{all_vals[idx][3]:.2f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
        idx += 1
        # prefill_time
        plt.bar(idx, all_vals[idx], bar_width, color='#8172B2', label='prefill_time' if i==0 else "")
        plt.text(idx, all_vals[idx], f'{all_vals[idx]:.2f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
        idx += 1
        # prefill_time_cpu
        plt.bar(idx, all_vals[idx], bar_width, color='#8172B2', hatch='//', label='prefill_time_cpu' if i==0 else "")
        plt.text(idx, all_vals[idx], f'{all_vals[idx]:.2f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
        idx += 1
        # decode_time
        plt.bar(idx, all_vals[idx], bar_width, color='#937860', label='decode_time' if i==0 else "")
        plt.text(idx, all_vals[idx], f'{all_vals[idx]:.2f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
        idx += 1
        # decode_time_cpu
        plt.bar(idx, all_vals[idx], bar_width, color='#937860', hatch='//', label='decode_time_cpu' if i==0 else "")
        plt.text(idx, all_vals[idx], f'{all_vals[idx]:.2f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
        idx += 1
    plt.xticks(np.arange(len(all_labels)), all_labels, rotation=45)
    plt.xlabel('Batch Size & Time Type')
    plt.ylabel('Time (s)')
    plt.title(f'Round {round_idx+1} Time Comparison')
    plt.legend(loc='upper right', fontsize=9)
    plt.grid(axis='y', linestyle='--', alpha=0.7)  # 添加y轴网格
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_all_per_round_stacked(merged_stats, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for round_idx in range(5):
        plot_bar_per_round_stacked(merged_stats, round_idx, os.path.join(out_dir, f'round_{round_idx+1}_bar.png'))

if __name__ == '__main__':
    gpu_dir = './output_dir4_gpu'
    cpu_dir = './output_dir4_cpu'
    gpu_stats = collect_all_stats(gpu_dir)
    cpu_stats = collect_all_stats(cpu_dir)
    merged_stats = merge_gpu_cpu_stats(gpu_stats, cpu_stats)
    plot_all_per_round_stacked(merged_stats, './bar_plots_size_gpu_cpu')
    print('Bar charts saved to ./bar_plots_size_gpu_cpu/')