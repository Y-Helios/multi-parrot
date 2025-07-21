#!/bin/bash

# ALFRED DAG对话处理系统启动脚本
# 包含：核心服务器 + 文本生成引擎 + 客户端
# 所有输出自动保存到 logs/ 目录下带时间戳的日志文件

set -e

# 配置
MODEL_PATH="/scratch/jsb5jd/LLaVA-NeXT/models/llava-next-interleave-qwen-7b"
CORE_SERVER_PORT=9000
ENGINE1_PORT=9001
ENGINE2_PORT=9002

# 图像大小配置 - 支持batch
IMAGE_SIZES=("$@")  # 支持多个参数
if [ ${#IMAGE_SIZES[@]} -eq 0 ]; then
    IMAGE_SIZES=("300,300")
fi
export IMAGE_SIZES_JSON="$(printf '%s\n' "${IMAGE_SIZES[@]}" | jq -R . | jq -s .)"

# 环境变量设置
export ENGINE1_URL="http://localhost:${ENGINE1_PORT}"
export ENGINE2_URL="http://localhost:${ENGINE2_PORT}"
export CORE_SERVER_PORT=${CORE_SERVER_PORT}
export IMAGE_SIZES_JSON

# NCCL环境变量设置（用于GPU间通信）
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_SOCKET_IFNAME=lo

# 日志设置
LOG_DIR="./logs"
mkdir -p "$LOG_DIR"
mkdir -p "./stat"
TIMESTAMP=$(date +'%Y-%m-%d_%H-%M-%S')
LOG_FILE="$LOG_DIR/alfred_run_${TIMESTAMP}.log"

# 手动清理函数 - 清理可能存在的旧进程
manual_cleanup() {
    echo "正在检查并清理可能存在的旧进程..."
    
    # 清理端口9000-9002上的进程
    for port in 9000 9001 9002; do
        if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
            echo "发现端口 $port 被占用，正在清理..."
            lsof -Pi :$port -sTCP:LISTEN -t | xargs kill -9 2>/dev/null || true
            sleep 1
        fi
    done
    
    # 清理可能的Python进程
    echo "正在清理可能的Python进程..."
    pkill -f "python.*engine.py" 2>/dev/null || true
    pkill -f "python.*core_server.py" 2>/dev/null || true
    pkill -f "python.*client.py" 2>/dev/null || true
    
    sleep 2
    echo "清理完成"
}

# 自动清理函数
cleanup() {
    echo "\n🧹 检测到脚本退出，正在关闭所有服务进程..." | tee -a "$LOG_FILE"
    
    # 杀死进程
    if [ -n "$CORE_SERVER_PID" ]; then 
        echo "正在关闭核心服务器 (PID: $CORE_SERVER_PID)..."
        kill "$CORE_SERVER_PID" 2>/dev/null || true
    fi
    if [ -n "$ENGINE1_PID" ]; then 
        echo "正在关闭文本生成引擎1 (PID: $ENGINE1_PID)..."
        kill "$ENGINE1_PID" 2>/dev/null || true
    fi
    if [ -n "$ENGINE2_PID" ]; then 
        echo "正在关闭文本生成引擎2 (PID: $ENGINE2_PID)..."
        kill "$ENGINE2_PID" 2>/dev/null || true
    fi
    if [ -n "$CLIENT_PID" ]; then 
        echo "正在关闭客户端 (PID: $CLIENT_PID)..."
        kill "$CLIENT_PID" 2>/dev/null || true
    fi
    
    # 等待进程完全退出
    sleep 2
    
    # 强制清理端口
    echo "正在清理端口..."
    for port in $CORE_SERVER_PORT $ENGINE1_PORT $ENGINE2_PORT; do
        if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
            echo "强制关闭端口 $port..."
            lsof -Pi :$port -sTCP:LISTEN -t | xargs kill -9 2>/dev/null || true
        fi
    done
    
    # 再次等待确保端口释放
    sleep 1
    
    echo "所有服务进程和端口已关闭。" | tee -a "$LOG_FILE"
}
trap cleanup EXIT

# 日志重定向
exec &> >(tee -a "$LOG_FILE")

echo "=========================================="
echo "  ALFRED DAG对话处理系统"
echo "=========================================="
echo "模型路径: ${MODEL_PATH}"
echo "核心服务器端口: ${CORE_SERVER_PORT}"
echo "文本生成引擎1端口: ${ENGINE1_PORT}"
echo "文本生成引擎2端口: ${ENGINE2_PORT}"
echo "图像大小: ${IMAGE_SIZES[@]}"
echo "日志目录: ${LOG_DIR}"
echo "时间戳: ${TIMESTAMP}"
echo "=========================================="

# 手动清理可能存在的旧进程
manual_cleanup

# 检查模型路径
if [ ! -d "$MODEL_PATH" ]; then
    echo "错误: 模型路径不存在: $MODEL_PATH"
    exit 1
fi

# 检查端口是否被占用
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null ; then
        echo "错误: 端口 $port 已被占用"
        exit 1
    fi
}

echo "\n检查端口占用..."
check_port $CORE_SERVER_PORT
check_port $ENGINE1_PORT
check_port $ENGINE2_PORT
echo "端口检查完成"

# 健康检查函数
wait_for_service() {
    local port=$1
    local service_name=$2
    local max_attempts=60  # 最多等待60次，每次10秒
    local attempt=0
    
    echo "等待 $service_name 启动 (端口: $port)..."
    while [ $attempt -lt $max_attempts ]; do
        if curl -s "http://localhost:$port/health" > /dev/null 2>&1; then
            echo "✓ $service_name 已就绪 (端口: $port)"
            return 0
        fi
        attempt=$((attempt + 1))
        echo "  尝试 $attempt/$max_attempts - 等待 $service_name 启动..."
        sleep 10
    done
    
    echo "✗ $service_name 启动超时 (端口: $port)"
    return 1
}

# 启动文本生成引擎1
echo "\n1. 启动文本生成引擎1 (端口: ${ENGINE1_PORT}, GPU: cuda:0)..."
ENGINE_ID="Engine1" CUDA_VISIBLE_DEVICES=0 UNIFIED_EVENT_LOG_FILE="$UNIFIED_EVENT_LOG_FILE" DISABLE_ENGINE_LOGS="true" WANDB_PROJECT="$WANDB_PROJECT" WANDB_RUN_NAME="${WANDB_RUN_NAME}_engine1" python engine.py \
    --model-path ${MODEL_PATH} \
    --port ${ENGINE1_PORT} \
    --device cuda:0 \
    --torch-dtype float16 \
    --attn-implementation flash_attention_2 \
    --temperature 0.2 \
    --max-new-tokens 1024 \
    --image-size ${IMAGE_SIZES[@]} \
    --log-file /dev/null 2>/dev/null &
ENGINE1_PID=$!
echo "文本生成引擎1已启动 (PID: $ENGINE1_PID, GPU: cuda:0)"

# 等待引擎1完全启动
if ! wait_for_service $ENGINE1_PORT "文本生成引擎1"; then
    echo "错误: 文本生成引擎1启动失败"
    exit 1
fi

# 启动文本生成引擎2
echo "\n2. 启动文本生成引擎2 (端口: ${ENGINE2_PORT}, GPU: cuda:1)..."
ENGINE_ID="Engine2" CUDA_VISIBLE_DEVICES=1 UNIFIED_EVENT_LOG_FILE="$UNIFIED_EVENT_LOG_FILE" DISABLE_ENGINE_LOGS="true" WANDB_PROJECT="$WANDB_PROJECT" WANDB_RUN_NAME="${WANDB_RUN_NAME}_engine2" python engine.py \
    --model-path ${MODEL_PATH} \
    --port ${ENGINE2_PORT} \
    --device cuda:0 \
    --torch-dtype float16 \
    --attn-implementation flash_attention_2 \
    --temperature 0.2 \
    --max-new-tokens 1024 \
    --image-size ${IMAGE_SIZES[@]} \
    --log-file /dev/null 2>/dev/null &
ENGINE2_PID=$!
echo "文本生成引擎2已启动 (PID: $ENGINE2_PID, GPU: cuda:1)"

# 等待引擎2完全启动
if ! wait_for_service $ENGINE2_PORT "文本生成引擎2"; then
    echo "错误: 文本生成引擎2启动失败"
    exit 1
fi

# 启动核心服务器
echo "\n3. 启动核心服务器 (端口: ${CORE_SERVER_PORT})..."
LOG_FILE="${LOG_DIR}/core_server_${TIMESTAMP}.log" python core_server.py 2>&1 | tee ./stat/pipeline_timing_${IMAGE_SIZES[@]}_${TIMESTAMP}.log &
CORE_SERVER_PID=$!
echo "核心服务器已启动 (PID: $CORE_SERVER_PID)"

# 等待核心服务器完全启动
if ! wait_for_service $CORE_SERVER_PORT "核心服务器"; then
    echo "错误: 核心服务器启动失败"
    exit 1
fi

echo "\n=== 所有服务已启动 ==="
echo "核心服务器: http://localhost:${CORE_SERVER_PORT}"
echo "文本生成引擎1: http://localhost:${ENGINE1_PORT}"
echo "文本生成引擎2: http://localhost:${ENGINE2_PORT}"
echo "图像大小: ${IMAGE_SIZES[@]}"
echo ""
echo "进程ID:"
echo "  核心服务器: $CORE_SERVER_PID"
echo "  文本生成引擎1: $ENGINE1_PID"
echo "  文本生成引擎2: $ENGINE2_PID"
echo ""
echo "日志文件:"
echo "  主日志 (所有组件): ${LOG_DIR}/pipeline_timing_${TIMESTAMP}.log"
echo "  核心服务器 (单独): ./stat/pipeline_timing_${TIMESTAMP}.log"
echo "  文本生成引擎1 (单独): ${LOG_DIR}/engine1_${TIMESTAMP}.log"
echo "  文本生成引擎2 (单独): ${LOG_DIR}/engine2_${TIMESTAMP}.log"
echo "  客户端 (单独): ${LOG_DIR}/client_${TIMESTAMP}.log"
echo ""

echo "\n========================================="
echo "           运行 Client                  "
echo "========================================="

# 启动客户端（后台运行）
python client.py --image-sizes-json="$IMAGE_SIZES_JSON" 2>&1 | tee ${LOG_DIR}/client_${TIMESTAMP}.log &
CLIENT_PID=$!
echo "客户端已启动 (PID: $CLIENT_PID)"
echo "客户端日志: ${LOG_DIR}/client_${TIMESTAMP}.log"

echo "\n=== 系统运行中 ==="
echo "ALFRED DAG对话处理系统已启动并运行。"
echo "使用以下命令查看日志:"
echo "  主日志 (所有组件): tail -f ${LOG_DIR}/pipeline_timing_${TIMESTAMP}.log"
echo "  核心服务器: tail -f ${LOG_DIR}/core_server_${TIMESTAMP}.log"
echo "  文本生成引擎1: tail -f ${LOG_DIR}/engine1_${TIMESTAMP}.log"
echo "  文本生成引擎2: tail -f ${LOG_DIR}/engine2_${TIMESTAMP}.log"
echo "  客户端: tail -f ${LOG_DIR}/client_${TIMESTAMP}.log"
echo ""
echo "使用 Ctrl+C 停止所有服务"

# 等待客户端完成
wait $CLIENT_PID

echo "\nClient 执行完毕。"
exit 0 