import argparse
import requests
import json
from typing import List, Dict, Any

def read_alfred_json(json_path: str) -> List[Dict[str, Any]]:
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def extract_sample_info(samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    sample_infos = []
    
    for idx, sample in enumerate(samples):
        sample_id = sample.get('sample_id', idx)
        conversations = sample.get("conversations", [])
        human_conversations = []
        for conv in conversations:
            if conv.get("from") == "human":
                human_conversations.append(conv.get("value", ""))
        image_paths = sample.get("image", [])
        sample_info = {
            "sample_id": sample_id,
            "conversations": human_conversations,
            "image_paths": image_paths
        }
        sample_infos.append(sample_info)
    
    return sample_infos

def send_batch_to_server(batch_data: List[Dict[str, Any]], server_host: str, server_port: int) -> List[Dict[str, Any]]:

    try:
        url = f"http://{server_host}:{server_port}/process_batch"
        payload = {"batch_data": batch_data}

        response = requests.post(url, json=payload, timeout=300)  # 增加到 300 秒
        response.raise_for_status()
        
        result = response.json()
        print(f"服务器响应: {result['message']}")
        
        results = result.get('results', [])
        return results
        
    except requests.exceptions.RequestException as e:
        print(f"发送数据到服务器时出错: {e}")
        return []


def main():
    parser = argparse.ArgumentParser(description="LLaVA-NeXT Client")
    parser.add_argument("--alfred-json", type=str, 
                       default="/scratch/jsb5jd/LLaVA-NeXT/interleave_data/ALFRED_short.json", 
                       help="Path to ALFRED.json")
    parser.add_argument("--batch-size", type=int, default=4, 
                       help="Batch size for processing")
    parser.add_argument("--max-samples", type=int, default=0, 
                       help="Max number of samples to process (0=all)")
    parser.add_argument("--server-host", type=str, default="localhost", 
                       help="Server host address")
    parser.add_argument("--server-port", type=int, default=9000, 
                       help="Server port")
    
    args = parser.parse_args()
    
    alfred_samples = read_alfred_json(args.alfred_json)
    
    if args.max_samples > 0:
        alfred_samples = alfred_samples[:args.max_samples]
    
    # 确保只处理20个batch，无论样本总数多少
    target_total_samples = args.batch_size * 20

    print(f"target_total_samples: {target_total_samples}")
    
    # 如果样本数量不够，就循环扩展
    if len(alfred_samples) < target_total_samples:
        from itertools import cycle, islice
        alfred_samples = list(islice(cycle(alfred_samples), target_total_samples))
    else:
        # 如果样本数量超过目标，就截取前target_total_samples个
        alfred_samples = alfred_samples[:target_total_samples]
    
    print(f"alfred_samples: {len(alfred_samples)}")

    
    sample_infos = extract_sample_info(alfred_samples)

    print(f"sample_infos: {len(sample_infos)}")
    
    batch_num = 0
    total_batches = 20  # 固定为20次
    all_results = []
    
    for batch_start in range(0, len(sample_infos), args.batch_size):
        batch_end = min(batch_start + args.batch_size, len(sample_infos))
        batch_data = sample_infos[batch_start:batch_end]
        
        batch_num += 1
        
        # 确保只运行20次
        if batch_num > 20:
            break
            
        batch_results = send_batch_to_server(batch_data, args.server_host, args.server_port)
        all_results.extend(batch_results)
    
    print(f"all_results: {all_results}")
    

if __name__ == "__main__":
    main() 