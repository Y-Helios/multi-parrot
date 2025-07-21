#!/bin/bash
# 检查参数
if [ $# -ne 1 ]; then
    echo "用法: $0 <batch-size>"
    exit 1
fi
size=$1
bs=8
start_time=$(date +%Y%m%d_%H%M%S)
LOG_DIR="image_size_info"
RUN_LOG_DIR="image_size_run_log"
mkdir -p "$LOG_DIR"
mkdir -p "$RUN_LOG_DIR"

# 记录脚本开始运行时间到日志
MONITOR_LOG="$RUN_LOG_DIR/monitor_size${size}_${start_time}.log"
echo "[INFO] Monitor script started at $(python3 -c 'import time; print(time.time())')" > "$MONITOR_LOG"

# 记录batch_image_loader.py启动时间
ENGINE_START_TIME=$(python3 -c 'import time; print(time.time())')
echo "[INFO] batch_image_loader.py started at $ENGINE_START_TIME" >> "$MONITOR_LOG"

# 启动 batch_image_loader.py，日志输出到 run_log/engine_bs${bs}_${start_time}.log，并记录进程号
# nohup python batch_image_loader.py --batch-size=${bs} > "$RUN_LOG_DIR/engine_bs${bs}_${start_time}.log" 2>&1 &
CUDA_VISIBLE_DEVICES=0 python batch_image_loader.py --image-size=${size} --batch-size=${bs} > "$RUN_LOG_DIR/engine_size${size}_${start_time}.log" 2>&1 &
ENGINE_PID=$!

# 获取 GPU 数量
NUM_GPUS=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -n1)

# 初始化每个csv文件表头
for i in $(seq 0 $((NUM_GPUS-1))); do
    CSV_FILE="$LOG_DIR/gpu${i}_size${size}_${start_time}.csv"
    if [ ! -f "$CSV_FILE" ]; then
        echo "time,utilization_gpu_percent,memory_used_MB,memory_total_MB" > "$CSV_FILE"
    fi
done

# trap 捕获 Ctrl+C 或 kill，记录脚本结束时间，并kill batch_image_loader.py
trap 'echo "[INFO] Monitor script ended at $(python3 -c "import time; print(time.time())") (killed by user)" >> "$MONITOR_LOG"; if kill -0 $ENGINE_PID 2>/dev/null; then echo "[INFO] Killing batch_image_loader.py (PID $ENGINE_PID) at $(python3 -c "import time; print(time.time())")" >> "$MONITOR_LOG"; kill $ENGINE_PID; fi; exit 0' SIGINT SIGTERM

# 实时监控，batch_image_loader.py 结束后自动退出
while true; do
    # 检查batch_image_loader.py是否还在
    if ! kill -0 $ENGINE_PID 2>/dev/null; then
        echo "[INFO] batch_image_loader.py (PID $ENGINE_PID) finished at $(python3 -c "import time; print(time.time())")" >> "$MONITOR_LOG"
        echo "[INFO] Monitor script ended at $(python3 -c "import time; print(time.time())")" >> "$MONITOR_LOG"
        exit 0
    fi
    NOW=$(python3 -c "import time; print(time.time())")
    for i in $(seq 0 $((NUM_GPUS-1))); do
        CSV_FILE="$LOG_DIR/gpu${i}_size${size}_${start_time}.csv"
        INFO=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits -i $i)
        UTIL=$(echo $INFO | cut -d',' -f1 | xargs)
        MEM_USED=$(echo $INFO | cut -d',' -f2 | xargs)
        MEM_TOTAL=$(echo $INFO | cut -d',' -f3 | xargs)
        echo "$NOW,$UTIL,$MEM_USED,$MEM_TOTAL" >> "$CSV_FILE"
    done
    sleep 1
done 