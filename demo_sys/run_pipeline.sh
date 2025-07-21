#!/bin/bash

# 设置脚本目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

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

# 检查server_pipeline.py和client.py是否存在
if [ ! -f "server_pipeline.py" ]; then
    echo "错误: 未找到server_pipeline.py文件"
    exit 1
fi

if [ ! -f "client.py" ]; then
    echo "错误: 未找到client.py文件"
    exit 1
fi

if [ ! -f "engine_pipeline.py" ]; then
    echo "错误: 未找到engine_pipeline.py文件"
    exit 1
fi

# 设置默认参数
SERVER_HOST="localhost"
SERVER_PORT="19100"
ENGINE1_HOST="localhost" 
ENGINE1_PORT="19101"
ENGINE2_HOST="localhost"
ENGINE2_PORT="19102"
ALFRED_JSON="/scratch/jsb5jd/LLaVA-NeXT/interleave_data/ALFRED_short.json"
MAX_SAMPLES="8"

# 清理可能存在的旧进程
echo "清理可能存在的旧进程..."
pkill -f "python.*server_pipeline.py" 2>/dev/null || true
pkill -f "python.*engine_pipeline.py" 2>/dev/null || true
# 清理可能占用的端口
for port in 19100 19101 19102; do
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
echo "Engine1参数:"
echo "  Host: $ENGINE1_HOST"
echo "  Port: $ENGINE1_PORT"
echo "  GPU: cuda:0"
echo ""
echo "Engine2参数:"
echo "  Host: $ENGINE2_HOST"
echo "  Port: $ENGINE2_PORT"
echo "  GPU: cuda:1"
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

# 启动Engine1 API服务器（后台运行，GPU 0）
echo "正在启动Engine1 API服务器..."
echo "执行命令: CUDA_VISIBLE_DEVICES=0 python engine_pipeline.py --start-api-server --api-host $ENGINE1_HOST --port $ENGINE1_PORT"

CUDA_VISIBLE_DEVICES=0 python engine_pipeline.py --start-api-server --api-host "$ENGINE1_HOST" --port "$ENGINE1_PORT" \
    --torch-dtype float16 \
    --attn-implementation flash_attention_2 \
    --temperature 0.2 \
    --max-new-tokens 1024 \
    --image-size 384 > engine1.log 2>&1 &
ENGINE1_PID=$!

echo "Engine1 PID: $ENGINE1_PID (GPU: cuda:0)"

# 启动Engine2 API服务器（后台运行，GPU 1）
echo "正在启动Engine2 API服务器..."
echo "执行命令: CUDA_VISIBLE_DEVICES=1 python engine_pipeline.py --start-api-server --api-host $ENGINE2_HOST --port $ENGINE2_PORT"

CUDA_VISIBLE_DEVICES=1 python engine_pipeline.py --start-api-server --api-host "$ENGINE2_HOST" --port "$ENGINE2_PORT" \
    --torch-dtype float16 \
    --attn-implementation flash_attention_2 \
    --temperature 0.2 \
    --max-new-tokens 1024 \
    --image-size 384 > engine2.log 2>&1 &
ENGINE2_PID=$!

echo "Engine2 PID: $ENGINE2_PID (GPU: cuda:1)"

# 等待Engine1启动 - 使用健康检查
echo "等待Engine1启动..."
for i in {1..60}; do
    sleep 10
    echo "等待Engine1中... ($i/60) - 已等待 $((i*10)) 秒"
    
    # 检查进程是否还在运行
    if ! kill -0 $ENGINE1_PID 2>/dev/null; then
        echo "错误: Engine1进程已退出"
        echo "Engine1日志内容:"
        tail -50 engine1.log 2>/dev/null || echo "无法读取Engine1日志"
        exit 1
    fi
    
    # 使用健康检查接口
    if curl -s "http://$ENGINE1_HOST:$ENGINE1_PORT/health" > /dev/null 2>&1; then
        echo "✓ Engine1健康检查通过 (端口: $ENGINE1_PORT)"
        break
    fi
    
    if [ $i -eq 60 ]; then
        echo "错误: Engine1启动超时 (等待了10分钟)"
        echo "Engine1日志内容:"
        tail -50 engine1.log 2>/dev/null || echo "无法读取Engine1日志"
        kill $ENGINE1_PID 2>/dev/null
        kill $ENGINE2_PID 2>/dev/null
        exit 1
    fi
done

echo "Engine1已启动 (PID: $ENGINE1_PID)"

# 等待Engine2启动 - 使用健康检查
echo "等待Engine2启动..."
for i in {1..60}; do
    sleep 10
    echo "等待Engine2中... ($i/60) - 已等待 $((i*10)) 秒"
    
    # 检查进程是否还在运行
    if ! kill -0 $ENGINE2_PID 2>/dev/null; then
        echo "错误: Engine2进程已退出"
        echo "Engine2日志内容:"
        tail -50 engine2.log 2>/dev/null || echo "无法读取Engine2日志"
        kill $ENGINE1_PID 2>/dev/null
        exit 1
    fi
    
    # 使用健康检查接口
    if curl -s "http://$ENGINE2_HOST:$ENGINE2_PORT/health" > /dev/null 2>&1; then
        echo "✓ Engine2健康检查通过 (端口: $ENGINE2_PORT)"
        break
    fi
    
    if [ $i -eq 60 ]; then
        echo "错误: Engine2启动超时 (等待了10分钟)"
        echo "Engine2日志内容:"
        tail -50 engine2.log 2>/dev/null || echo "无法读取Engine2日志"
        kill $ENGINE1_PID 2>/dev/null
        kill $ENGINE2_PID 2>/dev/null
        exit 1
    fi
done

echo "Engine2已启动 (PID: $ENGINE2_PID)"

# 设置环境变量告诉server两个engine的地址
export ENGINE1_URL="http://$ENGINE1_HOST:$ENGINE1_PORT"
export ENGINE2_URL="http://$ENGINE2_HOST:$ENGINE2_PORT"

# 启动服务器（后台运行）
echo "正在启动服务器..."
echo "执行命令: python server_pipeline.py --host $SERVER_HOST --port $SERVER_PORT"

# 将服务器的输出重定向到文件以便调试
python server_pipeline.py --host "$SERVER_HOST" --port "$SERVER_PORT" > server.log 2>&1 &
SERVER_PID=$!

echo "服务器PID: $SERVER_PID"

# 等待服务器启动
echo "等待服务器启动..."
for i in {1..10}; do
    sleep 1
    echo "等待服务器中... ($i/10)"
    
    # 检查进程是否还在运行
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "错误: 服务器进程已退出"
        echo "服务器日志内容:"
        cat server.log 2>/dev/null || echo "无法读取服务器日志"
        exit 1
    fi
    
    # 检查端口是否开放
    if netstat -tuln 2>/dev/null | grep ":$SERVER_PORT " > /dev/null; then
        echo "服务器端口 $SERVER_PORT 已开放"
        break
    fi
    
    if [ $i -eq 10 ]; then
        echo "错误: 服务器启动超时"
        echo "服务器日志内容:"
        cat server.log 2>/dev/null || echo "无法读取服务器日志"
        kill $SERVER_PID 2>/dev/null
        kill $ENGINE_PID 2>/dev/null
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
    --server-port "$SERVER_PORT"

CLIENT_EXIT_CODE=$?

# 等待客户端完成
echo "客户端执行完成 (退出码: $CLIENT_EXIT_CODE)"

# 停止服务器和Engine
echo "正在停止服务器和Engine..."
kill $SERVER_PID 2>/dev/null
kill $ENGINE1_PID 2>/dev/null
kill $ENGINE2_PID 2>/dev/null
wait $SERVER_PID 2>/dev/null
wait $ENGINE1_PID 2>/dev/null
wait $ENGINE2_PID 2>/dev/null

# 显示日志
echo ""
echo "=== 服务器日志 ==="
cat server.log 2>/dev/null || echo "无法读取服务器日志"

echo ""
echo "=== Engine1日志 (最后50行) ==="
tail -50 engine1.log 2>/dev/null || echo "无法读取Engine1日志"

echo ""
echo "=== Engine2日志 (最后50行) ==="
tail -50 engine2.log 2>/dev/null || echo "无法读取Engine2日志"

# 保存日志文件（添加时间戳）
TIMESTAMP=$(date +'%Y%m%d_%H%M%S')
mkdir -p logss
if [ -f "server.log" ]; then
    mv server.log "logss/server_${TIMESTAMP}.log"
    echo "服务器日志已保存到: logss/server_${TIMESTAMP}.log"
fi
if [ -f "engine1.log" ]; then
    mv engine1.log "logss/engine1_${TIMESTAMP}.log"
    echo "Engine1日志已保存到: logss/engine1_${TIMESTAMP}.log"
fi
if [ -f "engine2.log" ]; then
    mv engine2.log "logss/engine2_${TIMESTAMP}.log"
    echo "Engine2日志已保存到: logss/engine2_${TIMESTAMP}.log"
fi

echo "=== 脚本执行完成 ===" 