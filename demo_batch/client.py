import requests
import time
import os
import json
from typing import Optional, List, Dict
from datetime import datetime
import argparse

# --- Configuration ---
CORE_SERVER_URL = "http://localhost:9000/process_alfred"
ALFRED_DATA_PATH = "/scratch/jsb5jd/LLaVA-NeXT/interleave_data/ALFRED.json"
IMAGE_FOLDER = "/scratch/jsb5jd/LLaVA-NeXT/interleave_data"

# 日志文件设置
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, f"client_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log")

def log_sample(sample_id: str, conversations: List[str], image_paths: List[str], final_output: str):
    """记录整个sample的处理结果"""
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"=== Sample ID: {sample_id} ===\n")
        f.write(f"Conversations: {len(conversations)}\n")
        f.write(f"Image paths: {image_paths}\n")
        f.write("=== Conversations ===\n")
        for i, conv in enumerate(conversations):
            f.write(f"Conversation {i}: {conv}\n")
        f.write("=== Final Output ===\n")
        f.write(final_output + "\n\n")
        f.flush()

# --- Helper Functions ---
def read_alfred_data(data_path: str) -> List[Dict]:
    """读取ALFRED.json数据"""
    print(f"Reading ALFRED data from: {data_path}")
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"Loaded {len(data)} samples from ALFRED.json")
        return data
    except FileNotFoundError:
        print(f"Error: ALFRED data not found at {data_path}")
        return []
    except Exception as e:
        print(f"Error reading ALFRED data: {e}")
        return []

def extract_sample_data(sample: Dict) -> tuple:
    """从sample中提取所有human conversation和image paths"""
    sample_id = sample.get('sample_id', 'unknown')
    
    # 提取所有human conversation
    conversations = sample.get("conversations", [])
    human_conversations = []
    
    for conv in conversations:
        if conv.get("from") == "human":
            human_conversations.append(conv.get("value", ""))
    
    # 提取图像路径
    image_paths = sample.get("image", [])
    
    print(f"Sample {sample_id}: Found {len(human_conversations)} human conversations and {len(image_paths)} images")
    
    return sample_id, human_conversations, image_paths

def extract_sample_data_for_batch(sample: Dict) -> tuple:
    """从sample中提取数据，保持原始conversations格式用于批量处理"""
    sample_id = sample.get('sample_id', 'unknown')
    
    # 保持原始conversations格式（对话对象列表）
    conversations = sample.get("conversations", [])
    
    # 提取图像路径
    image_paths = sample.get("image", [])
    
    print(f"Sample {sample_id}: Found {len(conversations)} conversations and {len(image_paths)} images")
    
    return sample_id, conversations, image_paths

def request_sample_processing(sample_id: str, conversations: List[str], image_paths: List[str]) -> str:
    """一次性提交整个sample的数据给server处理"""
    payload = {
        "sample_id": sample_id,
        "conversations": conversations,  # 所有human conversation
        "image_paths": image_paths,      # 所有image paths
        "conv_mode": "qwen_1_5",
        "temperature": 0.2,
        "max_new_tokens": 1024
    }
    
    # Build URL from the CORE_SERVER_PORT environment variable
    core_server_port = os.getenv('CORE_SERVER_PORT', '9000')
    url = f"http://localhost:{core_server_port}/process_alfred"
    
    try:
        print(f"Sending sample {sample_id} to core server...")
        print(f"  - {len(conversations)} conversations")
        print(f"  - {len(image_paths)} images")
        
        response = requests.post(url, json=payload, timeout=300)  # 增加超时时间
        response.raise_for_status()
        result = response.json()
        
        if not result.get("success", False):
            raise Exception(f"Server error: {result.get('error', 'Unknown error')}")
        
        return result["output_text"]  # 返回最终结果
    except requests.exceptions.RequestException as e:
        print(f"\nError: Could not connect to core server. {e}")
        print("Please ensure the core server and engines are running.")
        exit(1)
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        exit(1)

def process_alfred_sample(sample: Dict) -> str:
    """处理单个ALFRED样本，一次性提交所有数据"""
    # 提取sample数据
    sample_id, conversations, image_paths = extract_sample_data(sample)
    
    print(f"\n=== Processing Sample ID: {sample_id} ===")
    print(f"Human conversations: {len(conversations)}")
    for i, conv in enumerate(conversations):
        print(f"  Conversation {i}: {conv}")
    
    if image_paths:
        print(f"Image paths: {image_paths}")
    
    # 一次性提交给server处理
    final_output = request_sample_processing(sample_id, conversations, image_paths)
    
    # 记录结果
    log_sample(sample_id, conversations, image_paths, final_output)
    
    print(f"\n=== Sample {sample_id} Processing Complete ===")
    print(f"Final output length: {len(final_output)}")
    print(f"Final output: {final_output}")
    
    return final_output

def request_batch_processing(samples, conv_mode="qwen_1_5", temperature=0.2, max_new_tokens=1024):
    """批量提交samples到server"""
    core_server_port = os.getenv('CORE_SERVER_PORT', '9000')
    url = f"http://localhost:{core_server_port}/process_alfred_batch"
    
    # 确保每个sample都有正确的格式
    formatted_samples = []
    for sample in samples:
        sample_id, conversations, image_paths = extract_sample_data_for_batch(sample)
        formatted_sample = {
            "sample_id": sample_id,
            "conversations": conversations,  # 保持原始对话对象格式
            "image": image_paths
        }
        formatted_samples.append(formatted_sample)
    
    payload = {
        "samples": formatted_samples,
        "conv_mode": conv_mode,
        "temperature": temperature,
        "max_new_tokens": max_new_tokens
    }
    try:
        print(f"Sending batch of {len(samples)} samples to core server...")
        response = requests.post(url, json=payload, timeout=600)
        response.raise_for_status()
        result = response.json()
        if not result.get("success", False):
            raise Exception(f"Server error: {result.get('error', 'Unknown error')}")
        return result["results"]  # list of {sample_id, output_text}
    except Exception as e:
        print(f"Error in batch processing: {e}")
        exit(1)

# --- Main Execution ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-sizes-json', type=str, default=None, help='JSON格式的image_size batch列表')
    parser.add_argument('--batch-size', type=int, default=4, help='每个batch的样本数，默认32')
    parser.add_argument('--max-samples', type=int, default=4, help='最多处理多少个样本，0为全部')
    args = parser.parse_args()

    alfred_data = read_alfred_data(ALFRED_DATA_PATH)
    if not alfred_data:
        return

    image_sizes = None
    if args.image_sizes_json and args.image_sizes_json.strip():
        import json as _json
        image_sizes = _json.loads(args.image_sizes_json)
        print(f"Batch image_sizes: {image_sizes}")

    # 支持 max-samples 限制
    if args.max_samples > 0:
        num_samples = min(args.max_samples, len(alfred_data))
    else:
        num_samples = len(alfred_data)
    batch_indices = list(range(num_samples))
    if image_sizes:
        batch_indices = list(range(min(len(image_sizes), num_samples)))

    start_time = time.time()
    batch_size = args.batch_size
    total_batches = (len(batch_indices) + batch_size - 1) // batch_size
    for batch_start in range(0, len(batch_indices), batch_size):
        batch_samples = []
        for idx in batch_indices[batch_start:batch_start+batch_size]:
            sample = alfred_data[idx]
            # 提取所有human conversation，组织二维input_text
            conversations = sample.get("conversations", [])
            input_text = []
            for conv in conversations:
                if conv.get("from") == "human":
                    input_text.append(conv.get("value", ""))
            # pad到最大轮数由server处理
            sample_id = sample.get('sample_id', 'unknown')
            image_paths = sample.get('image', [])
            batch_samples.append({
                'sample_id': sample_id,
                'input_text': input_text,
                'image': image_paths
            })
        print(f"\n--- Processing Batch {batch_start//batch_size+1}/{total_batches} ({len(batch_samples)} samples) ---")
        try:
            # 发送到server，期望server返回每轮output的列表
            core_server_port = os.getenv('CORE_SERVER_PORT', '9000')
            url = f"http://localhost:{core_server_port}/process_alfred_batch"
            payload = {'samples': batch_samples}
            response = requests.post(url, json=payload, timeout=600)
            response.raise_for_status()
            result = response.json()
            if not result.get("success", False):
                raise Exception(f"Server error: {result.get('error', 'Unknown error')}")
            # 期望results为[{sample_id, outputs: [每轮output], success}]
            results = result["results"]
            for res in results:
                sample_id = res.get('sample_id', 'unknown')
                outputs = res.get('outputs', [])
                # 记录每轮output到日志
                with open(LOG_FILE, 'a', encoding='utf-8') as f:
                    f.write(f"=== Sample ID: {sample_id} ===\n")
                    for i, out in enumerate(outputs):
                        f.write(f"Round {i+1} Output: {out}\n")
                    f.write("\n")
                print(f"Sample {sample_id} outputs: {[o[:100] for o in outputs]}")
        except Exception as e:
            print(f"Error processing batch {batch_start//batch_size+1}: {e}")
            continue
    end_time = time.time()
    print("\n" + "="*50)
    print("      ALFRED Data Processing Complete")
    print("="*50)
    print(f"Total processing time: {end_time - start_time:.2f} seconds.")
    print(f"Total samples processed: {len(batch_indices)}")

if __name__ == "__main__":
    main() 