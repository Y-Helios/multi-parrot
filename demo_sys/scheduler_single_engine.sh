#!/bin/bash
#SBATCH --job-name=1_eng
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=512G
#SBATCH --time=0-5:00:00
#SBATCH --cpus-per-task=64                   # 可选：你想给每个任务分配多少 CPU
#SBATCH --output=sbatch_logs/parrot_%j.out           # 输出日志
#SBATCH --error=sbatch_logs/parrot_%j.err            # 错误日志
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jsb5jd@virginia.edu

# 初始化环境
source ~/anaconda3/etc/profile.d/conda.sh

conda activate llava

# 切换到项目目录（可选）
cd /scratch/jsb5jd/LLaVA-NeXT/demo_a_sys

mkdir -p sbatch_logs

echo "Scheduler started at: $(date)"

# 定义batch size序列：从40开始，每次增加40，直到520
batch_sizes=(88 96 104 112 120 128 136 144 152 160 168 176 184 192 200 208 216 224 232 240 248 256 264 272)
# batch_sizes=(4 8)

# 循环执行每个batch size
for bs in "${batch_sizes[@]}"; do
    echo "=========================================="
    echo "Starting batch size: $bs"
    echo "Time: $(date)"
    echo "=========================================="
    
    # 带错误处理的调用
    if ! bash run_single_engine.sh $bs; then
        echo "Batch size $bs failed at $(date), skipping to next."
        continue
    fi
    
    # 等待一段时间再开始下一个batch size（可选）
    # 这样可以避免GPU资源冲突，让系统有时间清理
    echo "Waiting 10 seconds before next batch size..."
    sleep 10
    
    echo "Completed batch size: $bs"
    echo "=========================================="
done

# 记录调度结束时间
SCHEDULER_END_TIME=$(date +%Y%m%d_%H%M%S)
echo "Scheduler completed at: $SCHEDULER_END_TIME"
echo "All batch sizes have been processed!"
echo "Batch sizes tested: ${batch_sizes[*]}" 