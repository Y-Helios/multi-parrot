#!/bin/bash
#SBATCH --job-name=diff_img_size
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=100G
#SBATCH --time=0-5:00:00
#SBATCH --cpus-per-task=16                    # 可选：你想给每个任务分配多少 CPU
#SBATCH --output=sbatch_logs/parrot_%j.out           # 输出日志
#SBATCH --error=sbatch_logs/parrot_%j.err            # 错误日志
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jsb5jd@virginia.edu

# 初始化环境
source ~/anaconda3/etc/profile.d/conda.sh

conda activate llava

# 切换到项目目录（可选）
cd /scratch/jsb5jd/LLaVA-NeXT/demo

mkdir -p sbatch_logs

echo "Scheduler started at: $(date)"

# 定义image size序列
image_sizes=(1120 1152 1184 1216 1248 1280 1312 1344 1376 1408 1440 1472 1504 1536 1568 1600 1632 1664 1696 1728 1760 1792 1824 1856 1888 1920 1952 1984 2016 2048)

# 循环执行每个image size
for size in "${image_sizes[@]}"; do
    echo "=========================================="
    echo "Starting image size: $size"
    echo "Time: $(date)"
    echo "=========================================="
    
    # 带错误处理的调用
    if ! bash monitor_image_size.sh $size; then
        echo "Image size $size failed at $(date), skipping to next."
        continue
    fi
    
    # 等待一段时间再开始下一个batch size（可选）
    # 这样可以避免GPU资源冲突，让系统有时间清理
    echo "Waiting 30 seconds before next batch size..."
    sleep 30
    
    echo "Completed image size: $size"
    echo "=========================================="
done

# 记录调度结束时间
SCHEDULER_END_TIME=$(date +%Y%m%d_%H%M%S)
echo "Scheduler completed at: $SCHEDULER_END_TIME"
echo "All image sizes have been processed!"
echo "Image sizes tested: ${image_sizes[*]}" 