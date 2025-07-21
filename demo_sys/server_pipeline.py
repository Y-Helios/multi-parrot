import argparse
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Callable
import uvicorn
import asyncio
import os
import time
import sys
from dataclasses import dataclass
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor
import requests

parser = argparse.ArgumentParser(description="LLaVA-NeXT Server")
parser.add_argument("--port", type=int, default=9000, help="Server port")
parser.add_argument("--host", type=str, default="localhost", help="Server host")
parser.add_argument("--image-size", type=int, default=384, help="Image size for processing (width,height).")
args = parser.parse_args()

# --- FastAPI Models ---
class BatchData(BaseModel):
    batch_data: List[Dict[str, Any]]

class BatchResponse(BaseModel):
    status: str
    message: str
    results: List[Dict[str, Any]] = []

# --- Asynchronous communication related data structures ---
class RequestStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class AsyncRequest:
    request_id: str
    round_num: int
    input_texts: Optional[List[str]]
    image_paths: Optional[List[List[str]]]
    engine_url: str
    engine_num: str
    status: RequestStatus
    server_send_time: float
    payload_mem: float
    result: Optional[tuple] = None
    error: Optional[str] = None
    callback: Optional[Callable] = None

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
# Global thread pool
thread_pool = ThreadPoolExecutor(max_workers=10)
# Asynchronous request queue
async_requests: Dict[str, AsyncRequest] = {}
# Background tasks
background_tasks: List[asyncio.Task] = []

def get_next_engine_url() -> tuple[str, str]:
    """Get the URL and number of the next engine for round-robin scheduling
    
    Returns:
        tuple: (engine_url, engine_num)
        - engine_url: URL of the engine
        - engine_num: Number of the engine (e.g., "engine1" or "engine2")
    """
    global _current_engine_index
    
    # Get URLs of the two engines from environment variables
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
    """Record all timestamps and memory information for a round of conversation"""

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

# Modify send_async_request_to_engine to support engine_url/engine_num parameters
async def send_async_request_to_engine(input_texts: Optional[List[str]], image_paths: Optional[List[List[str]]], round_num: int = 0, callback: Optional[Callable] = None, engine_url: Optional[str] = None, engine_num: Optional[str] = None) -> str:
    global thread_pool, request_counter, async_requests
    request_id = f"req_{request_counter}_{round_num}_{int(time.time() * 1000)}"
    request_counter += 1
    # If engine is not specified, default to round-robin
    if engine_url is None or engine_num is None:
        engine_url_, engine_num_ = get_next_engine_url()
        if engine_url is None:
            engine_url = engine_url_
        if engine_num is None:
            engine_num = engine_num_
    payload = {
        "round_num": round_num,
        "input_texts": input_texts,
        "image_paths": image_paths
    }
    
    # Record send time and payload size
    server_send_time = time.time()
    payload_mem = sys.getsizeof(str(payload)) / (1024 * 1024)  # Convert to MB
    
    # Create asynchronous request object
    async_request = AsyncRequest(
        request_id=request_id,
        round_num=round_num,
        input_texts=input_texts,
        image_paths=image_paths,
        engine_url=engine_url,
        engine_num=engine_num,
        status=RequestStatus.PENDING,
        server_send_time=server_send_time,
        payload_mem=payload_mem,
        callback=callback
    )
    
    # Store request
    async_requests[request_id] = async_request
    
    # Start background task to process the request
    task = asyncio.create_task(process_async_request(request_id))
    background_tasks.append(task)
    print(f"Asynchronous request sent: {request_id} -> {engine_url} ({engine_num}) round_num={round_num}")
    return request_id

async def process_async_request(request_id: str):
    """Background processing of asynchronous requests"""
    global thread_pool, async_requests
    
    if request_id not in async_requests:
        return
    
    async_request = async_requests[request_id]
    async_request.status = RequestStatus.PROCESSING
    
    try:
        payload = {
            "round_num": async_request.round_num,
            "input_texts": async_request.input_texts,
            "image_paths": async_request.image_paths
        }
        
        # Execute synchronous HTTP request in the thread pool
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            thread_pool, 
            lambda: send_sync_request_to_engine(
                async_request.engine_url, 
                payload, 
                async_request.server_send_time,
                async_request.payload_mem,
                async_request.engine_num,
                async_request.round_num,
                len(async_request.input_texts) if async_request.input_texts else 0,
                sum(len(sublist) for sublist in async_request.image_paths) if async_request.image_paths else 0
            )
        )
        
        if result:
            output_texts, record_dict = result
            # Update request status
            async_request.status = RequestStatus.COMPLETED
            async_request.result = (output_texts, record_dict)
            
            # If there is a callback function, execute the callback
            if async_request.callback:
                await async_request.callback(output_texts, record_dict)
            
            print(f"Asynchronous request completed: {request_id}")
        else:
            async_request.status = RequestStatus.FAILED
            async_request.error = "Request failed"
            
    except Exception as e:
        print(f"Asynchronous request failed: {request_id}, Error: {e}")
        async_request.status = RequestStatus.FAILED
        async_request.error = str(e)

def send_sync_request_to_engine(engine_url: str, payload: dict, server_send_time: float, payload_mem: float, 
                               engine_num: str, round_num: int, text_num: Optional[int], image_num: Optional[int]) -> Optional[tuple]:
    """Synchronous request to engine (executed in the thread pool)"""
    try:
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
        
        # Extract timestamps and memory information from the response
        record_dict = round_record(
            round_num=round_num,
            engine_num=engine_num,
            text_num=text_num or 0,
            image_num=image_num or 0,
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
            engine_send_packet_mem=engine_send_packet_mem
        )
        
        return output_texts, record_dict
        
    except Exception as e:
        print(f"Synchronous request failed: {e}")
        return None

async def wait_for_request_completion(request_id: str, timeout: float = 60.0) -> Optional[tuple]:
    """Wait for asynchronous request to complete
    
    Returns:
        Optional[tuple]: (output_texts, record_dict) or None (if timeout or failed)
    """
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        if request_id in async_requests:
            async_request = async_requests[request_id]
            
            if async_request.status == RequestStatus.COMPLETED:
                return async_request.result
            elif async_request.status == RequestStatus.FAILED:
                print(f"Request failed: {request_id}, Error: {async_request.error}")
                return None
        
        await asyncio.sleep(0.1)  # Wait for 100ms
    
    print(f"Request timed out: {request_id}")
    return None

async def process_batch_data_async(batch_data: List[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], int]:
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
        return [], 0
    
    max_round = max(len(texts) for texts in batch_texts) if batch_texts else 0
    
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

    # Calculate cumulative image counts
    accum = [0] * batch_size
    image_nums = []
    for round_texts in input_texts:
        round_counts = []
        for i, text in enumerate(round_texts):
            if text == '<pad>':
                round_counts.append(0)
                accum[i] = 0
            else:
                count = text.count('<image>')
                accum[i] += count
                round_counts.append(accum[i])
        image_nums.append(round_counts)
    # Main pipeline scheduling
    cur_input_text = ["" for _ in range(batch_size)]
    sample_results = ["" for _ in range(batch_size)]
    sample_rounds = [0 for _ in range(batch_size)]
    sample_finished = [False for _ in range(batch_size)]
    final_results = []
    # Alternate scheduling between engine1 and engine2
    def get_engine_url_by_index(idx):
        engine1_url = os.getenv('ENGINE1_URL', 'http://localhost:9001')
        engine2_url = os.getenv('ENGINE2_URL', 'http://localhost:9002')
        return [engine1_url, engine2_url][idx]
    def get_engine_num_by_index(idx):
        return ["engine1", "engine2"][idx]
    for r in range(max_round):
        text_input = []
        image_input = []
        for b in range(batch_size):
            if r < len(input_texts) and input_texts[r][b] != '<pad>':
                cur_input_text[b] += input_texts[r][b]
                text_input.append(cur_input_text[b])
            else:
                text_input.append(None)
            images = batch_image_paths[b]
            a = image_nums[r][b] if r < len(image_nums) else 0
            image_input.append(images[:a])
        # Next round of image_paths
        next_image_input = []
        for b in range(batch_size):
            images = batch_image_paths[b]
            bnum = image_nums[r+1][b] if (r+1) < len(image_nums) else 0
            next_image_input.append(images[:bnum])
        # Alternate allocation of engines
        text_engine_idx = r % 2
        embed_engine_idx = 1 - text_engine_idx
        text_engine_url = get_engine_url_by_index(text_engine_idx)
        embed_engine_url = get_engine_url_by_index(embed_engine_idx)
        text_engine_num = get_engine_num_by_index(text_engine_idx)
        embed_engine_num = get_engine_num_by_index(embed_engine_idx)
        print(f"r: {r+1},\n text_input: {text_input},\n image_input: {image_input},\n next_image_input: {next_image_input}")
        # Text generation
        if r == 0:
            text_req_id = await send_async_request_to_engine(
                text_input, image_input, r+1, callback=None, engine_url=text_engine_url, engine_num=text_engine_num
            )
        else:
            text_req_id = await send_async_request_to_engine(
                text_input, None, r+1, callback=None, engine_url=text_engine_url, engine_num=text_engine_num
            )
        # Next round of embedding
        embed_req_id = None
        if r+1 < max_round:
            embed_req_id = await send_async_request_to_engine(
                None, next_image_input, r+2, callback=None, engine_url=embed_engine_url, engine_num=embed_engine_num
            )
        text_result = await wait_for_request_completion(text_req_id)
        if text_result:
            output, record_dict = text_result
            process_record["records"].append(record_dict)
            for i in range(batch_size):
                if text_input[i] is not None and not sample_finished[i]:
                    cur_input_text[i] += output[i] if i < len(output) else ""
                    sample_rounds[i] = r + 1
        else:
            for i in range(batch_size):
                if text_input[i] is not None and not sample_finished[i]:
                    sample_finished[i] = True
        # Check if any sample is completed
        for i in range(batch_size):
            if r == max_round-1 or (input_texts[r][i] == '<pad>' and not sample_finished[i]):
                sample_results[i] = cur_input_text[i]
                sample_finished[i] = True
    # Build final results
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
async def process_batch(request: BatchData):
    try:
        # Update request counter
        global process_record, request_counter
        request_counter += 1
        
        # Initialize processing record
        process_record["start_process_time"] = time.time()
        process_record["records"] = []
        process_record["request_id"] = request_counter
        
        results, batch_size = await process_batch_data_async(request.batch_data)
        
        # Record end time
        process_record["end_process_time"] = time.time()
        process_record["total_time"] = process_record["end_process_time"] - process_record["start_process_time"]

        # Ensure output directory exists
        output_dir = f"/scratch/jsb5jd/LLaVA-NeXT/demo_a_sys/output_dir2/batch_size{batch_size}_size{args.image_size}"
        os.makedirs(output_dir, exist_ok=True)

        # Write JSON file, filename includes request number
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

@app.get("/async_requests")
async def get_async_requests():
    """Get status of all asynchronous requests"""
    return {
        "total_requests": len(async_requests),
        "requests": {
            req_id: {
                "status": req.status.value,
                "round_num": req.round_num,
                "engine_num": req.engine_num,
                "error": req.error
            }
            for req_id, req in async_requests.items()
        }
    }

async def startup_event():
    """Initialize thread pool on application startup"""
    global thread_pool
    print("Thread pool initialized")

async def shutdown_event():
    """Clean up resources on application shutdown"""
    global thread_pool, background_tasks
    
    # Shutdown thread pool
    thread_pool.shutdown(wait=True)
    print("Thread pool shut down")
    
    # Cancel all background tasks
    for task in background_tasks:
        if not task.done():
            task.cancel()
    print("All background tasks cancelled")

# Register startup and shutdown events
app.add_event_handler("startup", startup_event)
app.add_event_handler("shutdown", shutdown_event)

def main():
    print(f"start FastAPI server at {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main() 