#!/bin/bash

# 设置脚本目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ========== 新增：捕获Ctrl+C信号，优雅退出 ==========
cleanup() {
    echo "\n捕获到中断信号，正在清理后台进程..."
    kill $SERVER_PID 2>/dev/null
    kill $ENGINE_PID 2>/dev/null
    exit 1
}
trap cleanup SIGINT SIGTERM EXIT
# ========== 新增结束 ==========

# 检查命令行参数
if [ $# -eq 0 ]; then
    echo "用法: $0 <batch_size>"
    echo "示例: $0 4"
    echo "默认batch_size: 4"
    BATCH_SIZE="4"
elif [ $# -eq 1 ]; then
    # 检查参数是否为数字
    if [[ $1 =~ ^[0-9]+$ ]]; then
        BATCH_SIZE="$1"
        echo "使用指定的batch_size: $BATCH_SIZE"
    else
        echo "错误: batch_size必须是数字"
        echo "用法: $0 <batch_size>"
        echo "示例: $0 4"
        exit 1
    fi
else
    echo "错误: 参数数量不正确"
    echo "用法: $0 <batch_size>"
    echo "示例: $0 4"
    exit 1
fi

# 检查Python是否可用
if ! command -v python &> /dev/null; then
    echo "错误: 未找到python命令"
    exit 1
fi

# 检查server_single_engine.py和client.py是否存在
if [ ! -f "server_single_engine.py" ]; then
    echo "错误: 未找到server_single_engine.py文件"
    exit 1
fi

if [ ! -f "client.py" ]; then
    echo "错误: 未找到client.py文件"
    exit 1
fi

if [ ! -f "engine.py" ]; then
    echo "错误: 未找到engine.py文件"
    exit 1
fi

# 设置默认参数
SERVER_HOST="localhost"
SERVER_PORT="9000"
ENGINE1_HOST="localhost" 
ENGINE1_PORT="9001"
ALFRED_JSON="/scratch/jsb5jd/LLaVA-NeXT/interleave_data/ALFRED_short.json"
MAX_SAMPLES="80"

# 清理可能存在的旧进程
echo "清理可能存在的旧进程..."
pkill -f "python.*server_single_engine.py" 2>/dev/null || true
pkill -f "python.*engine.py" 2>/dev/null || true
# 清理可能占用的端口
for port in 9000 9001; do
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        echo "清理端口 $port..."
        lsof -Pi :$port -sTCP:LISTEN -t | xargs kill -9 2>/dev/null || true
    fi
done
sleep 2

echo "服务器参数:"
echo "  Host: $SERVER_HOST"
echo "  Port: $SERVER_PORT"
echo ""
echo "Engine参数:"
echo "  Host: $ENGINE1_HOST"
echo "  Port: $ENGINE1_PORT"
echo "  GPU: cuda:0"
echo ""
echo "客户端参数:"
echo "  ALFRED JSON: $ALFRED_JSON"
echo "  Batch Size: $BATCH_SIZE"
echo "  Max Samples: $MAX_SAMPLES"
echo ""

# 检查FastAPI和uvicorn是否安装
echo "检查依赖..."
python -c "import fastapi, uvicorn; print('✓ FastAPI和uvicorn已安装')" || {
    echo "错误: 缺少FastAPI或uvicorn依赖"
    exit 1
}

# 启动Engine API服务器（后台运行，GPU 0）
echo "正在启动Engine API服务器..."
echo "执行命令: CUDA_VISIBLE_DEVICES=0 python engine.py --start-api-server --api-host $ENGINE1_HOST --port $ENGINE1_PORT"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0
python engine.py --start-api-server --api-host "$ENGINE1_HOST" --port "$ENGINE1_PORT" \
    --torch-dtype float16 \
    --attn-implementation flash_attention_2 \
    --temperature 0.2 \
    --max-new-tokens 1024 \
    --image-size 64 > engine.log 2>&1 &
ENGINE_PID=$!

echo "Engine PID: $ENGINE_PID (GPU: cuda:0)"

# 等待Engine启动 - 使用健康检查
echo "等待Engine启动..."
for i in {1..120}; do
    sleep 10
    echo "等待Engine中... ($i/120) - 已等待 $((i*10)) 秒"
    
    # 检查进程是否还在运行
    if ! kill -0 $ENGINE_PID 2>/dev/null; then
        echo "错误: Engine进程已退出"
        echo "Engine日志内容:"
        tail -50 engine.log 2>/dev/null || echo "无法读取Engine日志"
        exit 1
    fi
    
    # 使用健康检查接口
    if curl -s "http://$ENGINE1_HOST:$ENGINE1_PORT/health" > /dev/null 2>&1; then
        echo "✓ Engine健康检查通过 (端口: $ENGINE1_PORT)"
        break
    fi
    
    if [ $i -eq 60 ]; then
        echo "错误: Engine启动超时 (等待了10分钟)"
        echo "Engine日志内容:"
        tail -50 engine.log 2>/dev/null || echo "无法读取Engine日志"
        kill $ENGINE_PID 2>/dev/null
        exit 1
    fi
done

echo "Engine已启动 (PID: $ENGINE_PID)"

# 设置环境变量告诉server engine的地址
export ENGINE1_URL="http://$ENGINE1_HOST:$ENGINE1_PORT"
export ENGINE2_URL="http://$ENGINE1_HOST:$ENGINE1_PORT"  # 设置为同一个engine

# 启动服务器（后台运行）
echo "正在启动服务器..."
echo "执行命令: python server_single_engine.py --host $SERVER_HOST --port $SERVER_PORT"

# 将服务器的输出重定向到文件以便调试
python server_single_engine.py --host "$SERVER_HOST" --port "$SERVER_PORT" > server.log 2>&1 &
SERVER_PID=$!

echo "服务器PID: $SERVER_PID"

# 等待Engine启动 - 使用健康检查
echo "等待Server启动..."
for i in {1..60}; do
    sleep 10
    echo "等待Server中... ($i/60) - 已等待 $((i*10)) 秒"
    
    # 检查进程是否还在运行
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "错误: Server进程已退出"
        echo "Server日志内容:"
        tail -50 server.log 2>/dev/null || echo "无法读取Server日志"
        exit 1
    fi
    
    # 使用健康检查接口
    if curl -s "http://$SERVER_HOST:$SERVER_PORT/health" > /dev/null 2>&1; then
        echo "✓ Server健康检查通过 (端口: $SERVER_PORT)"
        break
    fi
    
    if [ $i -eq 60 ]; then
        echo "错误: Server启动超时 (等待了10分钟)"
        echo "Server日志内容:"
        tail -50 server.log 2>/dev/null || echo "无法读取Server日志"
        kill $SERVER_PID 2>/dev/null
        exit 1
    fi
done

echo "服务器已启动 (PID: $SERVER_PID)"

# 启动客户端
echo "正在启动客户端..."
echo "执行命令: python client.py --alfred-json $ALFRED_JSON --batch-size $BATCH_SIZE --max-samples $MAX_SAMPLES --server-host $SERVER_HOST --server-port $SERVER_PORT"

python client.py \
    --alfred-json "$ALFRED_JSON" \
    --batch-size "$BATCH_SIZE" \
    --max-samples "$MAX_SAMPLES" \
    --server-host "$SERVER_HOST" \
    --server-port "$SERVER_PORT" &
CLIENT_PID=$!

wait $CLIENT_PID
CLIENT_EXIT_CODE=$?

# 等待客户端完成
echo "客户端执行完成 (退出码: $CLIENT_EXIT_CODE)"

# 停止服务器和Engine
echo "正在停止服务器和Engine..."
kill $SERVER_PID 2>/dev/null
kill $ENGINE_PID 2>/dev/null
wait $SERVER_PID 2>/dev/null
wait $ENGINE_PID 2>/dev/null

# 显示日志
echo ""
echo "=== 服务器日志 ==="
cat server.log 2>/dev/null || echo "无法读取服务器日志"

echo ""
echo "=== Engine日志 (最后50行) ==="
tail -50 engine.log 2>/dev/null || echo "无法读取Engine日志"

# 保存日志文件（添加时间戳）
TIMESTAMP=$(date +'%Y%m%d_%H%M%S')
mkdir -p logs
if [ -f "server.log" ]; then
    mv server.log "logs/server_${TIMESTAMP}.log"
    echo "服务器日志已保存到: logs/server_${TIMESTAMP}.log"
fi
if [ -f "engine.log" ]; then
    mv engine.log "logs/engine_${TIMESTAMP}.log"
    echo "Engine日志已保存到: logs/engine_${TIMESTAMP}.log"
fi

echo "=== 脚本执行完成 ===" 