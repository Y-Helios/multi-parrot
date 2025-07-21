#!/bin/bash

# 简单的循环运行脚本
# 图像大小: 300, 600, 900, 1200, 1500
# 每个大小运行3次，总共15次

# 图像大小配置
# IMAGE_SIZES=("300,300" "600,600" "900,900" "1200,1200" "1500,1500")
IMAGE_SIZES=("300,300")
RUNS_PER_SIZE=1

# Wandb监控配置 - 更新为1秒间隔
WANDB_PROJECT="alfred-gpu-monitor"
WANDB_RUN_NAME="alfred_loop_$(date +%Y%m%d_%H%M%S)"
MONITOR_INTERVAL=1  # 改为1秒间隔，与修改后的脚本保持一致

# 设置wandb环境变量
export WANDB_PROJECT="$WANDB_PROJECT"
export WANDB_RUN_NAME="$WANDB_RUN_NAME"

# GPU监控进程ID
WANDB_MONITOR_PID=""

# 停止Wandb监控
stop_wandb_monitoring() {
    if [ ! -z "$WANDB_MONITOR_PID" ]; then
        echo "正在停止Wandb GPU监控..."
        kill $WANDB_MONITOR_PID 2>/dev/null
        wait $WANDB_MONITOR_PID 2>/dev/null
        echo "Wandb GPU监控已停止"
    fi
}

# 设置退出时清理
trap stop_wandb_monitoring EXIT

echo "开始循环运行 ALFRED 系统"
echo "图像大小: ${IMAGE_SIZES[*]}"
echo "每个大小运行次数: $RUNS_PER_SIZE"
echo "总运行次数: $(( ${#IMAGE_SIZES[@]} * RUNS_PER_SIZE ))"
echo "Wandb项目: $WANDB_PROJECT"
echo "Wandb运行名称: $WANDB_RUN_NAME"
echo "监控间隔: ${MONITOR_INTERVAL}秒 (已优化为1秒间隔)"
echo "=========================================="

# 启动Wandb GPU监控（自动事件检测版本）
echo "启动Wandb GPU监控（自动事件检测，1秒间隔）..."
python gpu_monitor_wandb_auto.py \
    --project "$WANDB_PROJECT" \
    --run-name "$WANDB_RUN_NAME" \
    --interval "$MONITOR_INTERVAL" \
    --log-dir "./logs" \
    --stat-dir "./stat" &
WANDB_MONITOR_PID=$!

echo "Wandb监控进程ID: $WANDB_MONITOR_PID"

# 等待3秒让监控启动（增加等待时间确保监控完全启动）
sleep 3

BATCH_SIZE=32  # 批量大小

total_sizes=${#IMAGE_SIZES[@]}
for ((i=0; i<total_sizes; i+=BATCH_SIZE)); do
    batch=()
    for ((j=0; j<BATCH_SIZE && i+j<total_sizes; j++)); do
        batch+=("${IMAGE_SIZES[i+j]}")
    done

    for run_num in $(seq 1 $RUNS_PER_SIZE); do
        echo ""
        echo "运行: batch=${batch[*]}, 第 $run_num/$RUNS_PER_SIZE 次"
        echo "执行: bash run_alfred.sh ${batch[*]}"
        echo "=========================================="
        
        # 批量传递参数
        bash run_alfred.sh "${batch[@]}"
        
        echo "完成: batch=${batch[*]}, 第 $run_num/$RUNS_PER_SIZE 次"
        echo "=========================================="
        
        # 等待10秒再开始下一次
        if [ $run_num -lt $RUNS_PER_SIZE ] || [ $((i+BATCH_SIZE)) -lt $total_sizes ]; then
            echo "等待10秒后继续..."
            sleep 10
        fi
    done
done

echo "所有运行完成！"
echo "Wandb监控数据已保存到: $WANDB_PROJECT/$WANDB_RUN_NAME"
echo "数据提取命令: python extract_wandb_gpu_data.py" 