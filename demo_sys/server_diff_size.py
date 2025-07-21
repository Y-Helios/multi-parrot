import argparse
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import uvicorn
import requests
import os
import time
import sys

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description="LLaVA-NeXT Server")
parser.add_argument("--port", type=int, default=9000, help="Server port")
parser.add_argument("--host", type=str, default="localhost", help="Server host")
parser.add_argument("--image-size", type=int, default=320, help="Image size for processing (width,height).")
args = parser.parse_args()

# --- FastAPI Models ---
class BatchData(BaseModel):
    batch_data: List[Dict[str, Any]]

class BatchResponse(BaseModel):
    status: str
    message: str
    results: List[Dict[str, Any]] = []

# --- Engine Communication ---
# 全局变量用于轮流调度
_current_engine_index = 0
# 全局字典用于记录处理信息
process_record = {
    "request_id": 0,
    "start_process_time": None,
    "records": [],
    "end_process_time": None,
    "total_time": None
}
# 全局请求计数器
request_counter = 0

def get_next_engine_url() -> tuple[str, str]:
    """获取下一个engine的URL和编号，实现轮流调度
    
    Returns:
        tuple: (engine_url, engine_num)
        - engine_url: engine的URL
        - engine_num: engine的编号（"engine1"或"engine2"）
    """
    global _current_engine_index
    
    # 从环境变量获取两个engine的URL
    engine1_url = os.getenv('ENGINE1_URL', 'http://localhost:9001')
    engine2_url = os.getenv('ENGINE2_URL', 'http://localhost:9002')
    
    engines = [engine1_url, engine2_url]
    engine_nums = ["engine1", "engine2"]
    current_url = engines[_current_engine_index]
    current_engine_num = engine_nums[_current_engine_index]
    
    # 轮换到下一个engine
    _current_engine_index = (_current_engine_index + 1) % len(engines)
    
    return current_url, current_engine_num

def round_record(round_num: int, engine_num: str, text_num: int, image_num: int, server_send_time: float, payload_mem: float,
              engine_receive_time: float, server_send_packet_mem: float,
              image_tensor_start: float, image_tensor_end: float,
              image_encode_start: float, image_encode_end: float,
              prefill_start: float, prefill_end: float,
              decode_start: float, decode_end: float,
              engine_send_time: float, tensor_mem: float,
              embed_mem: float, output_mem: float,
              server_receive_time: float, engine_send_packet_mem: float) -> dict:
    """记录一轮对话的所有时间戳和内存信息"""

    tensor_time = image_tensor_end - image_tensor_start
    embed_time = image_encode_end - image_encode_start
    image_time = image_encode_end - image_tensor_start
    prefill_time = prefill_end - prefill_start
    decode_time = decode_end - decode_start
    server_transfer_time = engine_receive_time - server_send_time
    engine_transfer_time = server_receive_time - engine_send_time
    
    return {
        "round_num": round_num,
        "engine_num": engine_num,
        "text_num": text_num,
        "image_num": image_num,
        "server_send_time": server_send_time,
        "payload_mem": payload_mem,
        "engine_receive_time": engine_receive_time,
        "server_send_packet_mem": server_send_packet_mem,
        "image_tensor_start": image_tensor_start,
        "image_tensor_end": image_tensor_end,
        "tensor_time": tensor_time,
        "image_encode_start": image_encode_start,
        "image_encode_end": image_encode_end,
        "embed_time": embed_time,
        "image_time": image_time,
        "prefill_start": prefill_start,
        "prefill_end": prefill_end,
        "prefill_time": prefill_time,
        "decode_start": decode_start,
        "decode_end": decode_end,
        "decode_time": decode_time,
        "engine_send_time": engine_send_time,
        "tensor_mem": tensor_mem,
        "embed_mem": embed_mem,
        "output_mem": output_mem,
        "server_receive_time": server_receive_time,
        "engine_send_packet_mem": engine_send_packet_mem,
        "server_transfer_time": server_transfer_time,
        "engine_transfer_time": engine_transfer_time
    }

def call_engine_generate_text_batch(input_texts: List[str], image_paths: List[List[str]], round_num: int = 0) -> tuple[List[str], dict]:
    """调用engine的generate_text_batch函数
    
    Returns:
        tuple: (output_texts, record_dict)
        - output_texts: 生成的文本列表
        - record_dict: 包含所有时间戳和内存信息的字典
    """
    try:
        # 获取当前轮次应该使用的engine
        engine_url, engine_num = get_next_engine_url()
        
        payload = {
            "round_num": round_num,
            "input_texts": input_texts,
            "image_paths": image_paths
        }
        
        # 记录发送时间和payload大小
        server_send_time = time.time()
        payload_mem = sys.getsizeof(str(payload)) / (1024 * 1024)  # 转换为MB
        
        response = requests.post(
            f"{engine_url}/generate_text_batch", 
            json=payload, 
            timeout=60
        )
        
        # 记录接收时间和response大小
        server_receive_time = time.time()
        # 计算整个response packet的大小
        headers_size = sum(len(key.encode('utf-8')) + len(str(value).encode('utf-8')) for key, value in response.headers.items())
        content_size = len(response.content)
        status_line_size = len(f"{response.status_code} {response.reason}".encode('utf-8'))
        engine_send_packet_mem = (headers_size + content_size + status_line_size) / (1024 * 1024)  # 转换为MB
        
        response.raise_for_status()
        
        result = response.json()
        output_texts = result.get('outputs', [])
        # print(f"engine输出: {output_texts}")
        
        # 从响应中提取时间戳和内存信息
        record_dict = round_record(
            round_num=round_num,
            engine_num=engine_num,
            text_num=len(input_texts),
            image_num=sum(len(sublist) for sublist in image_paths),
            server_send_time=server_send_time,
            payload_mem=payload_mem,
            engine_receive_time=result.get('engine_receive_time', 0.0),
            server_send_packet_mem=result.get('server_send_packet_mem', 0.0),
            image_tensor_start=result.get('image_tensor_start', 0.0),
            image_tensor_end=result.get('image_tensor_end', 0.0),
            image_encode_start=result.get('image_encode_start', 0.0),
            image_encode_end=result.get('image_encode_end', 0.0),
            prefill_start=result.get('prefill_start', 0.0),
            prefill_end=result.get('prefill_end', 0.0),
            decode_start=result.get('decode_start', 0.0),
            decode_end=result.get('decode_end', 0.0),
            engine_send_time=result.get('engine_send_time', 0.0),
            tensor_mem=result.get('tensor_mem', 0.0),
            embed_mem=result.get('embed_mem', 0.0),
            output_mem=result.get('output_mem', 0.0),
            server_receive_time=server_receive_time,
            engine_send_packet_mem=engine_send_packet_mem  # 使用实际计算的大小
        )
        
        return output_texts, record_dict
        
    except Exception as e:
        print(f"调用engine时出错: {e}")
        return [], {}

def process_batch_data(batch_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    global process_record
    batch_texts = []
    batch_image_paths = []
    
    for idx, sample in enumerate(batch_data):
        sample_id = sample.get('sample_id', idx)
        conversations = sample.get('conversations', [])
        image_paths = sample.get('image_paths', [])
        
        batch_texts.append(conversations)
        batch_image_paths.append(image_paths)
    
    batch_size = len(batch_texts)
    
    if batch_size == 0:
        print("empty batch, skip processing")
        return []
    
    max_round = max(len(texts) for texts in batch_texts) if batch_texts else 0
    # print(f"max_round: {max_round}")
    
    # 构建input_texts矩阵
    input_texts = []
    for r in range(max_round):
        input_text = []
        for b in range(batch_size):
            if r < len(batch_texts[b]):
                text = batch_texts[b][r]
                input_text.append(text)
            else:
                text = '<pad>'
                input_text.append(text)
        input_texts.append(input_text)
    # print(f"input_texts: {input_texts}")

    # 计算累计图像数量
    final_results = []
    if input_texts:
        accum = [0] * len(input_texts[0])
        image_nums = []
        for round_texts in input_texts:
            round_counts = []
            for i, text in enumerate(round_texts):
                if text == '<pad>':
                    round_counts.append(0)
                    accum[i] = 0  # pad后累计也归零
                else:
                    count = text.count('<image>')
                    accum[i] += count
                    round_counts.append(accum[i])
            image_nums.append(round_counts)
        # print(f"image_nums: {image_nums}")

        cur_input_text = ["" for _ in range(batch_size)]
        # print(f"cur_input_text: {cur_input_text}")
        
        # 记录每个样本的完成状态和结果
        sample_finished = [False] * batch_size  # 标记哪些样本已完成
        sample_results = [""] * batch_size      # 保存每个样本的最终结果
        sample_rounds = [0] * batch_size        # 记录每个样本完成的轮数

        # 多轮对话处理
        for r in range(max_round):
            image_paths = []
            for i, images in enumerate(batch_image_paths):
                a = image_nums[r][i]
                image_paths.append(images[:a])

            cur_input_text = [x + y for x, y in zip(cur_input_text, input_texts[r])]

            # 检查哪些样本在这一轮结束了（包含<pad>）
            active_indices = []
            for i, text in enumerate(cur_input_text):
                if "<pad>" in text:
                    # 这个样本在这一轮结束了
                    if not sample_finished[i]:
                        # 保存去掉<pad>后的结果
                        clean_text = text.replace("<pad>", "").strip()
                        sample_results[i] = clean_text
                        sample_rounds[i] = r  # 记录完成的轮数
                        sample_finished[i] = True
                        # print(f"  样本 {i} 在第 {r+1} 轮完成: {clean_text}")
                else:
                    # 这个样本还需要继续处理
                    active_indices.append(i)

            # 只处理还没完成的样本
            if active_indices:
                active_cur_input_text = [cur_input_text[i] for i in active_indices]
                active_image_paths = [image_paths[i] for i in active_indices]

                print(f"active_cur_input_text: {active_cur_input_text}")
                print(f"active_image_paths: {active_image_paths}")
                
                # 调用engine的generate_text_batch函数
                output, record_dict = call_engine_generate_text_batch(active_cur_input_text, active_image_paths, r+1)
                process_record["records"].append(record_dict)

                # 将输出结果映射回原始索引
                for j, orig_idx in enumerate(active_indices):
                    if j < len(output):
                        cur_input_text[orig_idx] += output[j]
            else:
                # 没有活跃样本了，全部完成
                # print(f"Round {r+1} - 所有样本都已完成")
                break
        
        # 处理到最后一轮还没完成的样本
        for i in range(batch_size):
            if not sample_finished[i]:
                sample_results[i] = cur_input_text[i]
                sample_rounds[i] = max_round

        # 构建最终结果
        for i, sample in enumerate(batch_data):
            final_results.append({
                "sample_id": sample.get('sample_id', 'unknown'),
                "final_output": sample_results[i] or "",
                "total_rounds": sample_rounds[i]
            })
    
    return final_results, batch_size

# --- FastAPI App ---
app = FastAPI()

@app.post("/process_batch", response_model=BatchResponse)
def process_batch(request: BatchData):
    try:
        # 更新请求计数器
        global process_record, request_counter
        request_counter += 1
        
        # 初始化处理记录
        process_record["start_process_time"] = time.time()
        process_record["records"] = []
        process_record["request_id"] = request_counter
        
        results, batch_size = process_batch_data(request.batch_data)
        
        # 记录结束时间
        process_record["end_process_time"] = time.time()
        process_record["total_time"] = process_record["end_process_time"] - process_record["start_process_time"]

        # 确保输出目录存在
        output_dir = f"/scratch/jsb5jd/LLaVA-NeXT/demo_a_sys/output_dir/image_size{args.image_size}"
        os.makedirs(output_dir, exist_ok=True)

        # 写入JSON文件，文件名包含请求编号
        file_path = os.path.join(output_dir, f"process_record_{request_counter}.json")
        with open(file_path, "w") as f:
            json.dump(process_record, f, indent=4)
        
        return BatchResponse(
            status="success", 
            message=f"200 OK",
            results=results
        )
    except Exception as e:
        print(f"error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "healthy"}

def main():
    print(f"start FastAPI server at {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main() 