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

# --- FastAPI Models ---
class BatchData(BaseModel):
    batch_data: List[Dict[str, Any]]

class BatchResponse(BaseModel):
    status: str
    message: str
    results: List[Dict[str, Any]] = []

# --- Engine Communication ---
# Global variable for round-robin scheduling
_current_engine_index = 0
# Global dictionary to record processing information
process_record = {
    "request_id": 0,
    "start_process_time": None,
    "records": [],
    "end_process_time": None,
    "total_time": None
}
# Global request counter
request_counter = 0

def get_next_engine_url() -> tuple[str, str]:
    """Get the URL and number of the next engine for round-robin scheduling.
    
    Returns:
        tuple: (engine_url, engine_num)
        - engine_url: URL of the engine
        - engine_num: Number of the engine (e.g., "engine1" or "engine2")
    """
    global _current_engine_index
    
    # Get the URLs of the two engines from environment variables
    engine1_url = os.getenv('ENGINE1_URL', 'http://localhost:9001')
    engine2_url = os.getenv('ENGINE2_URL', 'http://localhost:9002')
    
    engines = [engine1_url, engine2_url]
    engine_nums = ["engine1", "engine2"]
    current_url = engines[_current_engine_index]
    current_engine_num = engine_nums[_current_engine_index]
    
    # Rotate to the next engine
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
    """Record all timestamps and memory information for a single round of conversation."""

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
    """Call the generate_text_batch function of the engine.
    
    Returns:
        tuple: (output_texts, record_dict)
        - output_texts: List of generated texts
        - record_dict: Dictionary containing all timestamps and memory information
    """
    try:
        # Get the engine to use for the current round
        engine_url, engine_num = get_next_engine_url()
        
        payload = {
            "round_num": round_num,
            "input_texts": input_texts,
            "image_paths": image_paths
        }
        
        # Record send time and payload size
        server_send_time = time.time()
        payload_mem = sys.getsizeof(str(payload)) / (1024 * 1024)  # Convert to MB
        
        response = requests.post(
            f"{engine_url}/generate_text_batch", 
            json=payload, 
            timeout=60
        )
        
        # Record receive time and response size
        server_receive_time = time.time()
        # Calculate the size of the entire response packet
        headers_size = sum(len(key.encode('utf-8')) + len(str(value).encode('utf-8')) for key, value in response.headers.items())
        content_size = len(response.content)
        status_line_size = len(f"{response.status_code} {response.reason}".encode('utf-8'))
        engine_send_packet_mem = (headers_size + content_size + status_line_size) / (1024 * 1024)  # Convert to MB
        
        response.raise_for_status()
        
        result = response.json()
        output_texts = result.get('outputs', [])
        # print(f"engine output: {output_texts}")
        
        # Extract timestamps and memory information from the response
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
            engine_send_packet_mem=engine_send_packet_mem  # Use the actual calculated size
        )
        
        return output_texts, record_dict
        
    except Exception as e:
        print(f"Error calling engine: {e}")
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
    
    # Build input_texts matrix
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

    # Calculate cumulative image count
    final_results = []
    if input_texts:
        accum = [0] * len(input_texts[0])
        image_nums = []
        for round_texts in input_texts:
            round_counts = []
            for i, text in enumerate(round_texts):
                if text == '<pad>':
                    round_counts.append(0)
                    accum[i] = 0  # Reset accumulated count after pad
                else:
                    count = text.count('<image>')
                    accum[i] += count
                    round_counts.append(accum[i])
            image_nums.append(round_counts)
        # print(f"image_nums: {image_nums}")

        cur_input_text = ["" for _ in range(batch_size)]
        # print(f"cur_input_text: {cur_input_text}")
        
        # Record completion status and results for each sample
        sample_finished = [False] * batch_size  # Mark which samples are completed
        sample_results = [""] * batch_size      # Save the final result for each sample
        sample_rounds = [0] * batch_size        # Record the number of rounds completed for each sample

        # Multi-round conversation processing
        for r in range(max_round):
            image_paths = []
            for i, images in enumerate(batch_image_paths):
                a = image_nums[r][i]
                image_paths.append(images[:a])

            cur_input_text = [x + y for x, y in zip(cur_input_text, input_texts[r])]

            # Check which samples finished in this round (including <pad>)
            active_indices = []
            for i, text in enumerate(cur_input_text):
                if "<pad>" in text:
                    # This sample finished in this round
                    if not sample_finished[i]:
                        # Save the result after removing <pad>
                        clean_text = text.replace("<pad>", "").strip()
                        sample_results[i] = clean_text
                        sample_rounds[i] = r  # Record the number of rounds completed
                        sample_finished[i] = True
                        # print(f"   Sample {i} completed in round {r+1}: {clean_text}")
                else:
                    # This sample needs to be processed further
                    active_indices.append(i)

            # Only process samples that are not yet finished
            if active_indices:
                active_cur_input_text = [cur_input_text[i] for i in active_indices]
                active_image_paths = [image_paths[i] for i in active_indices]

                print(f"active_cur_input_text: {active_cur_input_text}")
                print(f"active_image_paths: {active_image_paths}")
                
                # Call the generate_text_batch function of the engine
                output, record_dict = call_engine_generate_text_batch(active_cur_input_text, active_image_paths, r+1)
                process_record["records"].append(record_dict)

                # Map the output results back to the original indices
                for j, orig_idx in enumerate(active_indices):
                    if j < len(output):
                        cur_input_text[orig_idx] += output[j]
            else:
                # No active samples left, all completed
                # print(f"Round {r+1} - All samples completed")
                break
        
        # Process samples that are still not finished in the last round
        for i in range(batch_size):
            if not sample_finished[i]:
                sample_results[i] = cur_input_text[i]
                sample_rounds[i] = max_round

        # Build the final results
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
        # Update request counter
        global process_record, request_counter
        request_counter += 1
        
        # Initialize processing record
        process_record["start_process_time"] = time.time()
        process_record["records"] = []
        process_record["request_id"] = request_counter
        
        results, batch_size = process_batch_data(request.batch_data)
        
        # Record end time
        process_record["end_process_time"] = time.time()
        process_record["total_time"] = process_record["end_process_time"] - process_record["start_process_time"]

        # Ensure output directory exists
        output_dir = f"/scratch/jsb5jd/LLaVA-NeXT/demo_a_sys/output_dir/batch_size{batch_size}"
        os.makedirs(output_dir, exist_ok=True)

        # Write JSON file, filename includes request number
        file_path = os.path.join(output_dir, f"test_process_record_{request_counter}.json")
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
    parser = argparse.ArgumentParser(description="LLaVA-NeXT Server")
    parser.add_argument("--port", type=int, default=9000, help="Server port")
    parser.add_argument("--host", type=str, default="localhost", help="Server host")
    
    args = parser.parse_args()
    
    print(f"start FastAPI server at {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main() 