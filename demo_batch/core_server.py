import requests
import itertools
import os
import re
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
from typing import Optional, List, Dict
import threading
import time
import json
import httpx
import concurrent.futures
from functools import partial
import sys
from datetime import datetime

# ========== Log Redirection ========== #
# Get the main log file name, prioritize environment variables, otherwise use default
ALFRED_LOG_FILE = os.environ.get('ALFRED_LOG_FILE', './logs/alfred_run.log')

def log_print(*args, **kwargs):
    # Concatenate content
    msg = ' '.join(str(a) for a in args)
    # Timestamp
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    msg = f"[{now}] {msg}"
    # Write to the main log file
    with open(ALFRED_LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(msg + '\n')
        f.flush()
    # Also output to standard output
    print(msg, flush=True)

# --- Configuration ---
# Support any number of engines, prioritize ENGINE_URLS (comma-separated), otherwise compatible with ENGINE1_URL/ENGINE2_URL, otherwise default to 2
engine_urls_env = os.getenv("ENGINE_URLS")
if engine_urls_env:
    ENGINE_URLS = [url.strip() for url in engine_urls_env.split(",") if url.strip()]
else:
    # Compatible with old usage
    ENGINE_URLS = []
    i = 1
    while True:
        url = os.getenv(f"ENGINE{i}_URL")
        if url:
            ENGINE_URLS.append(url)
            i += 1
        else:
            break
    if not ENGINE_URLS:
        # Default to 2
        ENGINE_URLS = ["http://localhost:9001", "http://localhost:9002"]

# Ensure at least two engines
if len(ENGINE_URLS) < 2:
    raise ValueError("At least two engines are required for pipeline processing")

# Global variable to store engine status
engine_status = {}  # {engine_url: {"last_activity": timestamp, "status": "idle|encoding|generating"}}

# Log file configuration
LOG_FILE = os.environ.get('LOG_FILE', 'pipeline_timing.log')

# Global variable to store sample start time and detailed statistics
sample_start_times = {}
sample_timing_stats = {}  # Store detailed timing statistics for each sample
sample_memory_stats = {}  # Store memory statistics for each sample

# Log recording function
def log_event(event_type: str, event_description: str, sample_id: str = "unknown", conversation_round: int = 0, **kwargs):
    """Record event to log file"""
    global sample_timing_stats, sample_start_times, sample_memory_stats
    
    timestamp = time.strftime('%H:%M:%S.%f')[:-3]  # Keep up to milliseconds
    log_line = f"{timestamp}: Server {event_description}"
    
    # If it's a sample start event, record the start time
    if event_type == "server_received_sample":
        sample_start_times[sample_id] = time.time()
        sample_timing_stats[sample_id] = {
            'image_encoding_times': [],
            'text_generation_times': [],
            'data_transfer_times': [],
            'waiting_times': [],
            'conversation_rounds': set()
        }
        sample_memory_stats[sample_id] = {
            'image_embedding_memory': [],
            'output_memory': []
        }
    
    # Record conversation round
    if sample_id not in sample_timing_stats:
        sample_timing_stats[sample_id] = {
            'image_encoding_times': [],
            'text_generation_times': [],
            'data_transfer_times': [],
            'waiting_times': [],
            'conversation_rounds': set()
        }
    sample_timing_stats[sample_id]['conversation_rounds'].add(conversation_round)
    
    # If it's a sample completion event, calculate and record total processing time and detailed statistics
    if event_type == "server_sample_completed":
        if sample_id in sample_start_times:
            total_time = time.time() - sample_start_times[sample_id]
            
            # Generate detailed statistics report
            stats = sample_timing_stats[sample_id]
            memory_stats = sample_memory_stats[sample_id]
            detailed_report = generate_detailed_timing_report(sample_id, total_time, stats, memory_stats)
            log_line += f" (Total processing time: {total_time:.6f}s)\n{detailed_report}"
            
            # Clean up records
            del sample_start_times[sample_id]
            del sample_timing_stats[sample_id]
            del sample_memory_stats[sample_id]
    
    # Write to log file
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(log_line + "\n")
    
    log_print(f"Server: {log_line}")

def generate_detailed_timing_report(sample_id: str, total_time: float, stats: dict, memory_stats: dict) -> str:
    """Generate detailed timing statistics report"""
    report_lines = []
    report_lines.append(f"  === Detailed Timing Report for Sample {sample_id} ===")
    report_lines.append(f"  Total processing time: {total_time:.6f}s")
    report_lines.append(f"  Conversation rounds: {sorted(stats['conversation_rounds'])}")
    
    # Data transfer time statistics
    if stats.get('data_transfer_times'):
        report_lines.append(f"  Data transfer times:")
        for i, (round_num, transfer_type, transfer_time) in enumerate(stats['data_transfer_times'], 1):
            report_lines.append(f"    Round {round_num} ({transfer_type}): {transfer_time:.6f}s")
        
        # Group by type
        transfer_by_type = {}
        for round_num, transfer_type, transfer_time in stats['data_transfer_times']:
            if transfer_type not in transfer_by_type:
                transfer_by_type[transfer_type] = []
            transfer_by_type[transfer_type].append(transfer_time)
        
        for transfer_type, times in transfer_by_type.items():
            avg_transfer = sum(times) / len(times)
            report_lines.append(f"    Average {transfer_type}: {avg_transfer:.6f}s")
        
        # Overall average
        avg_transfer = sum(time for _, _, time in stats['data_transfer_times']) / len(stats['data_transfer_times'])
        report_lines.append(f"    Overall average: {avg_transfer:.6f}s")
    
    # Memory usage statistics
    if memory_stats.get('image_embedding_memory'):
        report_lines.append(f"  Image embedding memory:")
        for round_num, memory_mb, shape_info in memory_stats['image_embedding_memory']:
            shape_str = f" shape: {shape_info}" if shape_info else ""
            report_lines.append(f"    Round {round_num}: {memory_mb:.2f} MB{shape_str}")
        avg_memory = sum(memory for _, memory, _ in memory_stats['image_embedding_memory']) / len(memory_stats['image_embedding_memory'])
        report_lines.append(f"    Average: {avg_memory:.2f} MB")
    
    if memory_stats.get('output_memory'):
        report_lines.append(f"  Output memory:")
        for round_num, memory_mb, shape_info in memory_stats['output_memory']:
            shape_str = f" shape: {shape_info}" if shape_info else ""
            report_lines.append(f"    Round {round_num}: {memory_mb:.4f} MB{shape_str}")
        avg_memory = sum(memory for _, memory, _ in memory_stats['output_memory']) / len(memory_stats['output_memory'])
        report_lines.append(f"    Average: {avg_memory:.4f} MB")
    
    # Data transfer memory statistics
    if memory_stats.get('data_transfer_memory'):
        report_lines.append(f"  Data transfer memory (content only):")
        for round_num, transfer_type, data_size_kb in memory_stats['data_transfer_memory']:
            report_lines.append(f"    Round {round_num} ({transfer_type}): {data_size_kb:.4f} KB")
        
        # Group by type
        transfer_memory_by_type = {}
        for round_num, transfer_type, data_size_kb in memory_stats['data_transfer_memory']:
            if transfer_type not in transfer_memory_by_type:
                transfer_memory_by_type[transfer_type] = []
            transfer_memory_by_type[transfer_type].append(data_size_kb)
        
        for transfer_type, sizes in transfer_memory_by_type.items():
            avg_size = sum(sizes) / len(sizes)
            report_lines.append(f"    Average {transfer_type}: {avg_size:.4f} KB")
    
    # Data transfer packet size statistics
    if memory_stats.get('data_transfer_packet_memory'):
        report_lines.append(f"  Data transfer packet memory (full JSON):")
        for round_num, transfer_type, packet_size_kb in memory_stats['data_transfer_packet_memory']:
            report_lines.append(f"    Round {round_num} ({transfer_type}): {packet_size_kb:.4f} KB")
        
        # Group by type
        transfer_packet_by_type = {}
        for round_num, transfer_type, packet_size_kb in memory_stats['data_transfer_packet_memory']:
            if transfer_type not in transfer_packet_by_type:
                transfer_packet_by_type[transfer_type] = []
            transfer_packet_by_type[transfer_type].append(packet_size_kb)
        
        for transfer_type, sizes in transfer_packet_by_type.items():
            avg_size = sum(sizes) / len(sizes)
            report_lines.append(f"    Average {transfer_type}: {avg_size:.4f} KB")
            
        # Calculate transfer efficiency (KB/s) - based on actual data content size
        report_lines.append(f"  Data transfer efficiency (KB/s) - based on packet size:")
        for round_num, transfer_type, packet_size_kb in memory_stats.get('data_transfer_memory', []):
            # Find the corresponding transfer time
            transfer_time = None
            if stats.get('data_transfer_times'):
                for t_round, t_type, t_time in stats['data_transfer_times']:
                    if t_round == round_num and t_type == transfer_type:
                        transfer_time = t_time
                        break
            
            if transfer_time and transfer_time > 0:
                efficiency = packet_size_kb / transfer_time
                report_lines.append(f"    Round {round_num} ({transfer_type}): {efficiency:.2f} KB/s")
        
        # Calculate average transfer efficiency by type
        for transfer_type, sizes in transfer_memory_by_type.items():
            # Find all transfer times for this type
            type_times = []
            if stats.get('data_transfer_times'):
                for t_round, t_type, t_time in stats['data_transfer_times']:
                    if t_type == transfer_type:
                        type_times.append(t_time)
            
            if type_times and len(type_times) == len(sizes):
                avg_efficiency = sum(size / time for size, time in zip(sizes, type_times) if time > 0) / len(sizes)
                report_lines.append(f"    Average {transfer_type} efficiency: {avg_efficiency:.2f} KB/s")
    
    return "\n".join(report_lines)

def record_image_encoding_time(sample_id: str, conversation_round: int, encoding_time: float):
    """Record image encoding time"""
    if sample_id in sample_timing_stats:
        if 'image_encoding_times' not in sample_timing_stats[sample_id]:
            sample_timing_stats[sample_id]['image_encoding_times'] = []
        sample_timing_stats[sample_id]['image_encoding_times'].append((conversation_round, encoding_time))

def record_text_generation_time(sample_id: str, conversation_round: int, generation_time: float):
    """Record text generation time"""
    if sample_id in sample_timing_stats:
        if 'text_generation_times' not in sample_timing_stats[sample_id]:
            sample_timing_stats[sample_id]['text_generation_times'] = []
        sample_timing_stats[sample_id]['text_generation_times'].append((conversation_round, generation_time))

def record_data_transfer_time(sample_id: str, conversation_round: int, transfer_type: str, transfer_time: float):
    """Record data transfer time"""
    if sample_id in sample_timing_stats:
        if 'data_transfer_times' not in sample_timing_stats[sample_id]:
            sample_timing_stats[sample_id]['data_transfer_times'] = []
        sample_timing_stats[sample_id]['data_transfer_times'].append((conversation_round, transfer_type, transfer_time))

def record_waiting_time(sample_id: str, conversation_round: int, waiting_time: float):
    """Record waiting time"""
    if sample_id in sample_timing_stats:
        if 'waiting_times' not in sample_timing_stats[sample_id]:
            sample_timing_stats[sample_id]['waiting_times'] = []
        sample_timing_stats[sample_id]['waiting_times'].append((conversation_round, waiting_time))

def record_image_embedding_memory(sample_id: str, conversation_round: int, memory_mb: float, shape_info: tuple = None):
    """Record image embedding memory usage"""
    if sample_id in sample_memory_stats:
        sample_memory_stats[sample_id]['image_embedding_memory'].append((conversation_round, memory_mb, shape_info))

def record_output_memory(sample_id: str, conversation_round: int, memory_mb: float, shape_info: tuple = None):
    """Record output memory usage"""
    if sample_id in sample_memory_stats:
        sample_memory_stats[sample_id]['output_memory'].append((conversation_round, memory_mb, shape_info))

def record_data_transfer_memory(sample_id: str, conversation_round: int, transfer_type: str, data_size_kb: float):
    """Record data transfer memory usage - only record the actual data content size"""
    if sample_id in sample_memory_stats:
        if 'data_transfer_memory' not in sample_memory_stats[sample_id]:
            sample_memory_stats[sample_id]['data_transfer_memory'] = []
        sample_memory_stats[sample_id]['data_transfer_memory'].append((conversation_round, transfer_type, data_size_kb))

def record_data_transfer_packet_memory(sample_id: str, conversation_round: int, transfer_type: str, packet_size_kb: float):
    """Record data transfer packet size - record the full JSON packet size"""
    if sample_id in sample_memory_stats:
        if 'data_transfer_packet_memory' not in sample_memory_stats[sample_id]:
            sample_memory_stats[sample_id]['data_transfer_packet_memory'] = []
        sample_memory_stats[sample_id]['data_transfer_packet_memory'].append((conversation_round, transfer_type, packet_size_kb))

# --- FastAPI App ---
app = FastAPI()

class AlfredRequest(BaseModel):
    sample_id: str  # sample ID
    conversations: List[str]  # all human conversations
    image_paths: Optional[List[str]] = None  # all image paths
    conv_mode: Optional[str] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    num_beams: Optional[int] = None
    max_new_tokens: Optional[int] = None

class AlfredResponse(BaseModel):
    output_text: str  # final text returned to client
    success: bool
    error: Optional[str] = None

class AlfredBatchRequest(BaseModel):
    samples: List[Dict]
    conv_mode: Optional[str] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    num_beams: Optional[int] = None
    max_new_tokens: Optional[int] = None

class AlfredBatchResponse(BaseModel):
    results: List[Dict]
    success: bool
    error: Optional[str] = None

def count_image_tokens(text: str) -> int:
    """Count the number of <image> placeholders in text"""
    return len(re.findall(r'<image>', text))

def get_image_paths_for_conversation(conversation: str, all_image_paths: List[str]) -> List[str]:
    """Return the corresponding image paths based on the number of <image> placeholders in the conversation"""
    image_count = count_image_tokens(conversation)
    return all_image_paths[:image_count] if image_count > 0 else []

def update_engine_status(engine_url: str, status: str, message: str = ""):
    """Update engine status"""
    engine_status[engine_url] = {
        "last_activity": time.time(),
        "status": status,
        "message": message
    }
    log_print(f"Server: Engine {engine_url} status updated - {status}: {message}")

def check_engine_status(engine_url: str, expected_status: str, timeout: int = 60) -> bool:
    """Check engine status, wait until expected status is reached or timeout"""
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        if engine_url in engine_status:
            current_status = engine_status[engine_url]["status"]
            if current_status == expected_status:
                log_print(f"Server: Engine {engine_url} reached expected status: {expected_status}")
                return True
            elif current_status == "error":
                error_msg = engine_status[engine_url].get("message", "Unknown error")
                raise Exception(f"Engine {engine_url} encountered error: {error_msg}")
        
        time.sleep(0.5)  # Check every 0.5 seconds
    
    raise Exception(f"Engine {engine_url} did not reach expected status '{expected_status}' within {timeout} seconds")

def encode_images_on_engine(engine_url: str, image_paths: List[str], conversation_round: int, sample_id: str, **kwargs) -> bool:
    """Encode images on a specified engine and monitor status"""
    log_event(
        "server_start_image_encoding", 
        f"Server starting image encoding for conversation {conversation_round} on engine {engine_url}", 
        sample_id, conversation_round,
        engine_url=engine_url,
        image_count=len(image_paths)
    )
    
    update_engine_status(engine_url, "encoding", f"Encoding {len(image_paths)} images for conversation {conversation_round}")
    
    try:
        # Send image encoding request
        encode_request = {
            "image_paths": image_paths,
            "conv_mode": kwargs.get("conv_mode"),
            "temperature": kwargs.get("temperature"),
            "top_p": kwargs.get("top_p"),
            "num_beams": kwargs.get("num_beams"),
            "max_new_tokens": kwargs.get("max_new_tokens"),
            "sample_id": sample_id,
            "conversation_round": conversation_round
        }
        
        # Record server send start time
        send_start = time.time()
        
        # Record actual start time of sending image encoding request
        log_event(
            "server_actual_image_send_start", 
            f"Server actual image send start time recorded: {send_start:.6f}", 
            sample_id, conversation_round,
            engine_url=engine_url,
            send_start=send_start
        )
        
        log_event(
            "server_send_image_request", 
            f"Server sending image encoding request to engine {engine_url}", 
            sample_id, conversation_round,
            engine_url=engine_url,
            image_count=len(image_paths)
        )
        
        response = requests.post(
            f"{engine_url}/encode_images",
            json=encode_request,
            timeout=60
        )
        
        response.raise_for_status()
        
        result = response.json()
        
        # Calculate accurate server_to_engine transfer time
        if result.get("status_info", {}).get("receive_timestamp"):
            engine_receive_time = result["status_info"]["receive_timestamp"]
            server_to_engine_transfer_time = engine_receive_time - send_start
            log_event(
                "server_accurate_image_transfer_time", 
                f"Server accurate image transfer time calculation - Send start: {send_start:.6f}, Engine receive: {engine_receive_time:.6f}, Transfer time: {server_to_engine_transfer_time:.6f}s", 
                sample_id, conversation_round,
                engine_url=engine_url,
                send_start=send_start,
                engine_receive_time=engine_receive_time,
                transfer_time=server_to_engine_transfer_time
            )
        else:
            # If no receive timestamp, use estimation method
            total_transfer_time = time.time() - send_start
            estimated_send_time = total_transfer_time * 0.1
            server_to_engine_transfer_time = estimated_send_time
            log_event(
                "server_image_transfer_time_estimate", 
                f"Server image transfer time estimate - Total: {total_transfer_time:.6f}s, Estimated send: {estimated_send_time:.6f}s", 
                sample_id, conversation_round,
                engine_url=engine_url,
                total_time=total_transfer_time,
                estimated_send_time=estimated_send_time
            )
        
        # Record data transfer time
        record_data_transfer_time(sample_id, conversation_round, "server_to_engine_image_paths", server_to_engine_transfer_time)
        
        # Calculate and record full JSON request and response size
        encode_request_size = len(json.dumps(encode_request).encode('utf-8'))
        encode_request_size_kb = encode_request_size / 1024
        
        result_size = len(json.dumps(result).encode('utf-8'))
        result_size_kb = result_size / 1024
        
        # Calculate actual image path data size (excluding JSON wrapper)
        image_paths_data = "\n".join(image_paths)
        image_paths_size_kb = len(image_paths_data.encode('utf-8')) / 1024
        
        # Record data transfer memory size - record actual data content size
        record_data_transfer_memory(sample_id, conversation_round, "server_to_engine_image_paths", image_paths_size_kb)
        
        # Record data transfer packet size - record full JSON packet size
        record_data_transfer_packet_memory(sample_id, conversation_round, "server_to_engine_image_paths", encode_request_size_kb)
        
        # Record detailed data transfer information
        log_event(
            "server_image_data_transfer_details", 
            f"Server image data transfer details - Request content: {image_paths_size_kb:.4f} KB, Request packet: {encode_request_size_kb:.4f} KB, Response packet: {result_size_kb:.4f} KB", 
            sample_id, conversation_round,
            engine_url=engine_url,
            request_content_kb=image_paths_size_kb,
            request_packet_kb=encode_request_size_kb,
            response_packet_kb=result_size_kb
        )
        
        if result.get("success", False):
            # Check status info
            status_info = result.get("status_info", {})
            if status_info:
                engine_port = status_info.get("engine_port", "unknown")
                status_message = status_info.get("message", "")
                log_print(f"Server: {status_message}")
            
            update_engine_status(engine_url, "idle", f"Image encoding completed for conversation {conversation_round}")
            log_event(
                "server_image_encoding_success", 
                f"Server successfully encoded {len(image_paths)} images on engine {engine_url} (transfer time: {server_to_engine_transfer_time:.6f}s)", 
                sample_id, conversation_round,
                engine_url=engine_url,
                image_count=len(image_paths),
                transfer_time=server_to_engine_transfer_time
            )
            return True
        else:
            error_msg = result.get('error', 'Unknown error')
            update_engine_status(engine_url, "error", f"Image encoding failed: {error_msg}")
            log_event(
                "server_image_encoding_failed", 
                f"Server failed to encode images on engine {engine_url}: {error_msg}", 
                sample_id, conversation_round,
                engine_url=engine_url,
                error=error_msg
            )
            return False
            
    except Exception as e:
        update_engine_status(engine_url, "error", f"Image encoding error: {str(e)}")
        log_event(
            "server_image_encoding_error", 
            f"Server error encoding images on engine {engine_url}: {e}", 
            sample_id, conversation_round,
            engine_url=engine_url,
            error=str(e)
        )
        return False

def generate_text_on_engine(engine_url: str, input_text: str, conversation_round: int, sample_id: str, image_paths: Optional[List[str]] = None, **kwargs) -> tuple[str, float]:
    """Generate text on a specified engine and monitor status, return (output_text, server_to_engine_transfer_time)"""
    log_event(
        "server_start_text_generation", 
        f"Server starting text generation for conversation {conversation_round} on engine {engine_url}", 
        sample_id, conversation_round,
        engine_url=engine_url,
        text_length=len(input_text)
    )
    
    update_engine_status(engine_url, "generating", f"Generating text for conversation {conversation_round}")
    
    try:
        # Build text generation request
        text_request = {
            "input_text": input_text,
            "conv_mode": kwargs.get("conv_mode"),
            "temperature": kwargs.get("temperature"),
            "top_p": kwargs.get("top_p"),
            "num_beams": kwargs.get("num_beams"),
            "max_new_tokens": kwargs.get("max_new_tokens"),
            "sample_id": sample_id,
            "conversation_round": conversation_round
        }
        
        # If there are image paths, add them to the request
        if image_paths:
            text_request["image_paths"] = image_paths
            log_event(
                "server_send_text_with_images", 
                f"Server sending text generation request with {len(image_paths)} image paths to engine {engine_url}", 
                sample_id, conversation_round,
                engine_url=engine_url,
                image_count=len(image_paths)
            )
        else:
            log_event(
                "server_send_text_only", 
                f"Server sending text generation request (no images) to engine {engine_url}", 
                sample_id, conversation_round,
                engine_url=engine_url
            )
        
        # Record start time of sending request
        send_start = time.time()
        
        # Record actual start time of sending request
        log_event(
            "server_actual_send_start", 
            f"Server actual send start time recorded: {send_start:.6f}", 
            sample_id, conversation_round,
            engine_url=engine_url,
            send_start=send_start
        )
        
        # Calculate size of data to send
        request_data_size = len(json.dumps(text_request).encode('utf-8'))
        request_data_size_kb = request_data_size / 1024
        
        # Call engine
        response = requests.post(
            f"{engine_url}/generate",
            json=text_request,
            timeout=120
        )
        
        response.raise_for_status()
        
        # Record time of receiving result
        receive_timestamp = time.time()
        
        # Record actual completion time of receiving
        log_event(
            "server_actual_receive_complete", 
            f"Server actual receive complete time recorded: {receive_timestamp:.6f}", 
            sample_id, conversation_round,
            engine_url=engine_url,
            receive_timestamp=receive_timestamp
        )
        
        engine_response = response.json()
        
        # Calculate accurate server_to_engine transfer time
        if "receive_timestamp" in engine_response:
            engine_receive_time = engine_response["receive_timestamp"]
            server_to_engine_transfer_time = engine_receive_time - send_start
            log_event(
                "server_accurate_transfer_time", 
                f"Server accurate transfer time calculation - Send start: {send_start:.6f}, Engine receive: {engine_receive_time:.6f}, Transfer time: {server_to_engine_transfer_time:.6f}s", 
                sample_id, conversation_round,
                engine_url=engine_url,
                send_start=send_start,
                engine_receive_time=engine_receive_time,
                transfer_time=server_to_engine_transfer_time
            )
        else:
            # If no receive timestamp, use estimation method
            total_transfer_time = receive_timestamp - send_start
            estimated_send_time = total_transfer_time * 0.1
            server_to_engine_transfer_time = estimated_send_time
            log_event(
                "server_transfer_time_estimate", 
                f"Server transfer time estimate - Total: {total_transfer_time:.6f}s, Estimated send: {estimated_send_time:.6f}s", 
                sample_id, conversation_round,
                engine_url=engine_url,
                total_time=total_transfer_time,
                estimated_send_time=estimated_send_time
            )
        
        # Calculate engine to server transfer time
        if "return_timestamp" in engine_response:
            engine_return_time = engine_response["return_timestamp"]
            engine_to_server_transfer_time = receive_timestamp - engine_return_time
            record_data_transfer_time(sample_id, conversation_round, "engine_to_server_output", engine_to_server_transfer_time)
            log_event(
                "server_received_engine_output", 
                f"Server received output from engine {engine_url} (transfer time: {engine_to_server_transfer_time:.6f}s)", 
                sample_id, conversation_round,
                engine_url=engine_url,
                transfer_time=engine_to_server_transfer_time
            )
        
        # Calculate output data size
        output_text = engine_response.get("output_text", "")
        output_data_size = len(output_text.encode('utf-8'))
        output_data_size_kb = output_data_size / 1024
        
        # Calculate full JSON response size (including all fields)
        full_response_size = len(json.dumps(engine_response).encode('utf-8'))
        full_response_size_kb = full_response_size / 1024
        
        # Calculate size of sent request data
        request_data_size = len(json.dumps(text_request).encode('utf-8'))
        request_data_size_kb = request_data_size / 1024
        
        # Calculate actual data content size (excluding JSON wrapper)
        input_text_size = len(input_text.encode('utf-8'))
        input_text_size_kb = input_text_size / 1024
        
        # Record data transfer memory size - record actual data content size
        record_data_transfer_memory(sample_id, conversation_round, "server_to_engine_text", input_text_size_kb)
        record_data_transfer_memory(sample_id, conversation_round, "engine_to_server_output", output_data_size_kb)
        
        # Record data transfer packet size - record full JSON packet size
        record_data_transfer_packet_memory(sample_id, conversation_round, "server_to_engine_text", request_data_size_kb)
        record_data_transfer_packet_memory(sample_id, conversation_round, "engine_to_server_output", full_response_size_kb)
        
        # Record detailed data transfer information
        log_event(
            "server_data_transfer_details", 
            f"Server data transfer details - Request content: {input_text_size_kb:.4f} KB, Request packet: {request_data_size_kb:.4f} KB, Response content: {output_data_size_kb:.4f} KB, Response packet: {full_response_size_kb:.4f} KB", 
            sample_id, conversation_round,
            engine_url=engine_url,
            request_content_kb=input_text_size_kb,
            request_packet_kb=request_data_size_kb,
            response_content_kb=output_data_size_kb,
            response_packet_kb=full_response_size_kb
        )
        
        if not engine_response.get("success", False):
            error_msg = engine_response.get('error', 'Unknown error')
            update_engine_status(engine_url, "error", f"Text generation failed: {error_msg}")
            log_event(
                "server_text_generation_failed", 
                f"Server text generation failed on engine {engine_url}: {error_msg}", 
                sample_id, conversation_round,
                engine_url=engine_url,
                error=error_msg
            )
            raise Exception(f"Engine error: {error_msg}")
        
        if not output_text:
            update_engine_status(engine_url, "error", "Engine returned empty output_text")
            log_event(
                "server_empty_output", 
                f"Server received empty output from engine {engine_url}", 
                sample_id, conversation_round,
                engine_url=engine_url
            )
            raise Exception("Engine returned empty output_text")
        
        # Record engine return time information
        if "timing" in engine_response:
            timing = engine_response["timing"]
            if "image_encoding_time" in timing and timing["image_encoding_time"] > 0:
                record_image_encoding_time(sample_id, conversation_round, timing["image_encoding_time"])
                log_event(
                    "server_recorded_image_encoding_time", 
                    f"Server recorded image encoding time: {timing['image_encoding_time']:.6f}s", 
                    sample_id, conversation_round,
                    engine_url=engine_url,
                    encoding_time=timing["image_encoding_time"]
                )
            
            if "text_generation_time" in timing:
                record_text_generation_time(sample_id, conversation_round, timing["text_generation_time"])
                log_event(
                    "server_recorded_text_generation_time", 
                    f"Server recorded text generation time: {timing['text_generation_time']:.6f}s", 
                    sample_id, conversation_round,
                    engine_url=engine_url,
                    generation_time=timing["text_generation_time"]
                )
        
        # Record engine return memory information
        if "memory_stats" in engine_response:
            memory_stats = engine_response["memory_stats"]
            if "image_embedding_memory_mb" in memory_stats:
                record_image_embedding_memory(sample_id, conversation_round, memory_stats["image_embedding_memory_mb"])
                log_event(
                    "server_recorded_image_embedding_memory", 
                    f"Server recorded image embedding memory: {memory_stats['image_embedding_memory_mb']:.2f} MB", 
                    sample_id, conversation_round,
                    engine_url=engine_url,
                    memory_mb=memory_stats["image_embedding_memory_mb"]
                )
            
            if "output_memory_mb" in memory_stats:
                record_output_memory(sample_id, conversation_round, memory_stats["output_memory_mb"])
                log_event(
                    "server_recorded_output_memory", 
                    f"Server recorded output memory: {memory_stats['output_memory_mb']:.4f} MB", 
                    sample_id, conversation_round,
                    engine_url=engine_url,
                    memory_mb=memory_stats["output_memory_mb"]
                )
        
        update_engine_status(engine_url, "idle", f"Text generation completed for conversation {conversation_round}")
        log_event(
            "server_text_generation_success", 
            f"Server successfully generated text on engine {engine_url}, output length: {len(output_text)}", 
            sample_id, conversation_round,
            engine_url=engine_url,
            output_length=len(output_text)
        )
        
        # Return server_to_engine transfer time for caller to use
        return output_text, server_to_engine_transfer_time
        
    except Exception as e:
        update_engine_status(engine_url, "error", f"Text generation error: {str(e)}")
        log_event(
            "server_text_generation_error", 
            f"Server error generating text on engine {engine_url}: {e}", 
            sample_id, conversation_round,
            engine_url=engine_url,
            error=str(e)
        )
        raise e

def pipeline_process_sample(sample_id: str, conversations: List[str], image_paths: List[str], **kwargs) -> str:
    """Process the entire sample using a pipeline approach, handling conversation dependencies"""
    # Check if in batch processing mode
    is_batch_mode = kwargs.pop('is_batch_mode', False)
    
    log_event(
        "server_start_pipeline", 
        f"Server starting pipeline processing for sample {sample_id} (batch_mode: {is_batch_mode})", 
        sample_id, 0,
        conversation_count=len(conversations),
        image_count=len(image_paths)
    )
    
    # Initialize engine status
    for engine_url in ENGINE_URLS:
        update_engine_status(engine_url, "idle", "Initialized")
    
    # Use two engines for pipeline processing
    engine1_url = ENGINE_URLS[0]
    engine2_url = ENGINE_URLS[1]
    
    accumulated_conversation = ""
    
    # Only encode images in non-batch mode (images are pre-encoded in batch mode)
    if not is_batch_mode:
        # Batch pre-encode all images (if multiple rounds of conversations need images)
        all_encoding_requests = []
        for i, conversation in enumerate(conversations):
            # Build input text for the current round
            if i == 0:
                current_input = conversation
            else:
                current_input = f"{accumulated_conversation}\n{conversation}"
            
            # Get image paths needed for the current round
            current_image_count = count_image_tokens(current_input)
            current_image_paths = image_paths[:current_image_count] if current_image_count > 0 else []
            
            if current_image_paths:
                # Determine which engine to use for the current round
                current_engine_url = engine1_url if i % 2 == 0 else engine2_url
                
                # Add to batch encoding requests
                all_encoding_requests.append({
                    'engine_url': current_engine_url,
                    'image_paths': current_image_paths,
                    'conversation_round': i + 1,
                    'sample_id': sample_id,
                    'kwargs': kwargs
                })
        
        # Batch encode all images
        if all_encoding_requests:
            log_print(f"Server: Batch encoding {len(all_encoding_requests)} image sets for sample {sample_id}")
            
            # Group batch requests by engine
            engine_requests = {}
            for req in all_encoding_requests:
                engine_url = req['engine_url']
                if engine_url not in engine_requests:
                    engine_requests[engine_url] = []
                engine_requests[engine_url].append(req)
            
            # Execute batch encoding for each engine
            for engine_url, engine_requests_list in engine_requests.items():
                try:
                    # Build batch request
                    batch_requests = []
                    for req in engine_requests_list:
                        batch_requests.append({
                            "image_paths": req['image_paths'],
                            "conv_mode": req['kwargs'].get("conv_mode"),
                            "temperature": req['kwargs'].get("temperature"),
                            "top_p": req['kwargs'].get("top_p"),
                            "num_beams": req['kwargs'].get("num_beams"),
                            "max_new_tokens": req['kwargs'].get("max_new_tokens"),
                            "sample_id": req['sample_id'],
                            "conversation_round": req['conversation_round']
                        })
                    
                    # Send batch encoding request
                    batch_request = {"requests": batch_requests}
                    response = requests.post(
                        f"{engine_url}/encode_images_batch",
                        json=batch_request,
                        timeout=120
                    )
                    response.raise_for_status()
                    result = response.json()
                    
                    if result.get("success", False):
                        log_print(f"Server: Batch image encoding completed on {engine_url} for {len(engine_requests_list)} requests")
                    else:
                        log_print(f"Server: Batch image encoding failed on {engine_url}: {result.get('error', 'Unknown error')}")
                        
                except Exception as e:
                    log_print(f"Server: Error in batch image encoding on {engine_url}: {e}")
    else:
        log_print(f"Server: Skipping image encoding for sample {sample_id} (batch mode)")
    
    # Process each conversation round
    for i, conversation in enumerate(conversations):
        log_event(
            "server_start_conversation", 
            f"Server starting conversation {i+1}/{len(conversations)}", 
            sample_id, i+1
        )
        
        # Determine which engine to use for the current round
        current_engine_url = engine1_url if i % 2 == 0 else engine2_url
        
        # Build input text for the current round
        if i == 0:
            current_input = conversation
        else:
            current_input = f"{accumulated_conversation}\n{conversation}"
        
        # Get image paths needed for the current round
        current_image_count = count_image_tokens(current_input)
        current_image_paths = image_paths[:current_image_count] if current_image_count > 0 else []
        
        log_event(
            "server_conversation_info", 
            f"Server processing conversation {i+1} on engine {current_engine_url}", 
            sample_id, i+1,
            engine_url=current_engine_url,
            text_length=len(current_input),
            image_count=current_image_count
        )
        
        # Pass image_paths only in the first round, subsequent rounds do not pass (use pre-encoded embedding)
        image_paths_to_send = current_image_paths if i == 0 else None
        
        # Send text generation request to the current engine
        log_event(
            "server_send_text_generation", 
            f"Server sending text generation request to engine {current_engine_url}", 
            sample_id, i+1,
            engine_url=current_engine_url,
            text_length=len(current_input),
            image_paths_sent=len(image_paths_to_send) if image_paths_to_send else 0,
            is_first_round=(i == 0)
        )
        
        try:
            response, server_to_engine_transfer_time = generate_text_on_engine(
                current_engine_url, 
                current_input, 
                i + 1, 
                sample_id,
                image_paths_to_send,  # Only pass image_paths in the first round
                conv_mode=kwargs.get("conv_mode"),
                temperature=kwargs.get("temperature"),
                top_p=kwargs.get("top_p"),
                num_beams=kwargs.get("num_beams"),
                max_new_tokens=kwargs.get("max_new_tokens")
            )
            
            # Record correct data transfer time (only network transfer, not engine processing time)
            # Pass image_paths in the first round, subsequent rounds pass text (using pre-encoded embedding)
            transfer_type = "image_paths" if i == 0 else "text"
            record_data_transfer_time(sample_id, i+1, f"server_to_engine_{transfer_type}", server_to_engine_transfer_time)
            
            log_event(
                "server_text_generation_completed", 
                f"Server text generation completed on engine {current_engine_url} (transfer time: {server_to_engine_transfer_time:.6f}s, type: {transfer_type})", 
                sample_id, i+1,
                engine_url=current_engine_url,
                output_length=len(response),
                transfer_time=server_to_engine_transfer_time,
                transfer_type=transfer_type
            )
            accumulated_conversation = f"{current_input}\n{response}"
        except Exception as e:
            log_event(
                "server_text_generation_exception", 
                f"Server exception during text generation on engine {current_engine_url}: {e}", 
                sample_id, i+1,
                engine_url=current_engine_url,
                error=str(e)
            )
            raise e
    
    log_event(
        "server_pipeline_completed", 
        f"Server pipeline processing completed for sample {sample_id}", 
        sample_id, len(conversations),
        final_output_length=len(accumulated_conversation)
    )
    
    return accumulated_conversation

# --- API Endpoints ---
@app.post("/process_alfred", response_model=AlfredResponse)
def process_alfred(request: AlfredRequest):
    """
    Process ALFRED data, implementing a pipeline-style conversation dependency
    Alternates between two engines to process different conversation rounds, and pre-encodes images in advance
    """
    try:
        # Record start time of processing sample
        log_event(
            "server_received_sample", 
            f"Server received sample {request.sample_id}", 
            request.sample_id, 0,
            conversation_count=len(request.conversations),
            image_count=len(request.image_paths) if request.image_paths else 0
        )
        
        log_print(f"Server: Processing sample {request.sample_id} with pipeline approach")
        log_print(f"Server: {len(request.conversations)} conversations, {len(request.image_paths or []) if request.image_paths else 'no'} images")
        log_print(f"Server: Available engines: {ENGINE_URLS}")
        
        # Process the entire sample using the pipeline approach
        final_output = pipeline_process_sample(
            sample_id=request.sample_id,
            conversations=request.conversations,
            image_paths=request.image_paths or [],
            conv_mode=request.conv_mode,
            temperature=request.temperature,
            top_p=request.top_p,
            num_beams=request.num_beams,
            max_new_tokens=request.max_new_tokens
        )
        
        log_print(f"Server: Successfully processed sample {request.sample_id}")
        log_print(f"Server: Final output length: {len(final_output)}")
        
        # Get engine timing summary
        try:
            for engine_url in ENGINE_URLS:
                try:
                    response = requests.get(f"{engine_url}/timing_summary/{request.sample_id}", timeout=5)
                    if response.status_code == 200:
                        summary_data = response.json()
                        summary = summary_data.get("summary", "")
                        if summary:
                            log_print(f"Server: Engine timing summary for sample {request.sample_id}:")
                            log_print(summary)
                except Exception as e:
                    log_print(f"Server: Warning - Failed to get timing summary from {engine_url}: {e}")
        except Exception as e:
            log_print(f"Server: Warning - Failed to get engine timing summaries: {e}")
        
        # Record time when sample processing is completed
        log_event(
            "server_sample_completed", 
            f"Server completed processing sample {request.sample_id}", 
            request.sample_id, len(request.conversations),
            output_length=len(final_output)
        )
        
        return AlfredResponse(output_text=final_output, success=True)
        
    except requests.exceptions.RequestException as e:
        log_print(f"Server: Error connecting to engine. {e}")
        raise HTTPException(status_code=503, detail=f"Engine is unavailable: {e}")
    except Exception as e:
        log_print(f"Server: An unexpected error occurred. {e}")
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {e}")

@app.post("/process_alfred_batch")
def process_alfred_batch(request: AlfredBatchRequest):
    """
    Batch pipeline inference (truly pipeline pre-encoding):
    - In each round, server sends the current input_text batch and image paths batch to the current engine (inference),
      while sending the next image paths batch to the next engine (pre-encoding).
    - After waiting for the current output to return, concatenate the output to the next input_text, and then give it to the next engine for inference.
    - Alternate scheduling engines to achieve pipeline-style batch inference and image pre-encoding.
    - Refer to the pipeline scheduling logic in process_alfred.
    """
    results = []
    batch_samples = request.samples
    batch_size = len(batch_samples)
    log_print(f"Server: Starting batch pipeline processing for {batch_size} samples")

    # 1. Organize two-dimensional input_text and image path
    all_input_text = []  # [ [str, str, ...], ... ]
    all_image_paths = []
    max_round = 0
    for sample in batch_samples:
        input_text = sample.get('input_text', [])
        log_print(f"cccccccccc input_text: {input_text}")
        all_input_text.append(input_text)
        all_image_paths.append(sample.get('image', []))
        if len(input_text) > max_round:
            max_round = len(input_text)
    # Pad input_text to max_round
    for i in range(batch_size):
        while len(all_input_text[i]) < max_round:
            all_input_text[i].append('<pad>')

    log_print(f"===================Server: all_input_text: ")
    log_print(f"batch_size: {len(all_input_text)}, max_round: {max_round}")
    log_print(f"{all_input_text}")
    log_print(f"all_image_paths: {all_image_paths}")
    log_print(f"=" * 100)

    # 2. Calculate cumulative image count matrix image_num[n][r]
    image_num = []
    for n in range(batch_size):
        row = []
        acc = 0
        for r in range(max_round):
            if all_input_text[n][r] == '<pad>':
                row.append(acc)
            else:
                acc += count_image_tokens(all_input_text[n][r])
                row.append(acc)
        image_num.append(row)

    log_print(f"===================Server: image_num: ")
    log_print(f"batch_size: {len(image_num)}, max_round: {len(image_num[0]) if image_num else 0}")
    log_print(f"{image_num}")
    log_print(f"=" * 100)

    # Pad image_num to max_round (using the cumulative count of the last round)
    for n in range(batch_size):
        last = image_num[n][-1] if image_num[n] else 0
        while len(image_num[n]) < max_round:
            image_num[n].append(last)

    log_print(f"===================Server: after pad image_num: ")
    log_print(f"batch_size: {len(image_num)}, max_round: {len(image_num[0]) if image_num else 0}")
    log_print(f"{image_num}")
    log_print(f"=" * 100)

    # 3. Main inference loop (pipeline scheduling)
    outputs = [[] for _ in range(batch_size)]  # Output for each round of each sample
    cur_input = [all_input_text[n][0] if all_input_text[n][0] != '<pad>' else '<pad>' for n in range(batch_size)]
    num_engines = len(ENGINE_URLS)
    # Pre-prepare image path batch for each round
    round_image_paths = []
    for r in range(max_round):
        round_paths = []
        for n in range(batch_size):
            if all_input_text[n][r] == '<pad>':
                # Pad samples do not need images
                round_paths.append([])
            else:
                # Use cumulative image count: take the first image_num[n][r] images
                cumulative_image_count = image_num[n][r]
                if cumulative_image_count > 0:
                    paths = all_image_paths[n][:cumulative_image_count]
                else:
                    paths = []
                round_paths.append(paths)
        round_image_paths.append(round_paths)

    log_print(f"===================Server: round_image_paths: ")
    log_print(f"max_round: {len(round_image_paths)}, batch_size: {len(round_image_paths[0]) if round_image_paths else 0}")
    log_print(f"{round_image_paths}")
    log_print(f"=" * 100)

    # Main pipeline loop
    for r in range(max_round):
        log_print(f"\nServer: [Pipeline] Starting Round {r+1}")
        cur_engine_idx = r % num_engines
        next_engine_idx = (r+1) % num_engines
        cur_engine_url = ENGINE_URLS[cur_engine_idx]
        next_engine_url = ENGINE_URLS[next_engine_idx]
        # 1. Current round inference batch
        round_inputs = []
        for n in range(batch_size):
            is_pad = (all_input_text[n][r] == '<pad>' or cur_input[n] == '<pad>')
            input_str = cur_input[n] if not is_pad else '<pad>'
            round_inputs.append({
                'sample_id': batch_samples[n]['sample_id'],
                'input_text': input_str,
                'is_pad': is_pad,
                'conversation_round': r+1
            })
        # 2. Current round image batch
        batch_image_requests = []
        for n in range(batch_size):
            paths = round_image_paths[r][n]
            batch_image_requests.append({
                'image_paths': paths,
                'sample_id': batch_samples[n]['sample_id'],
                'conversation_round': r+1
            })
        # 3. Next round image batch (for pre-encoding)
        if r+1 < max_round:
            preencode_image_requests = []
            for n in range(batch_size):
                if all_input_text[n][r+1] == '<pad>':
                    # Pad samples do not need pre-encoded images
                    preencode_image_requests.append({
                        'image_paths': [],
                        'sample_id': batch_samples[n]['sample_id'],
                        'conversation_round': r+2
                    })
                else:
                    # Use cumulative image count for the next round: take the first image_num[n][r+1] images
                    next_round_cumulative_count = image_num[n][r+1]
                    if next_round_cumulative_count > 0:
                        paths = all_image_paths[n][:next_round_cumulative_count]
                    else:
                        paths = []
                    preencode_image_requests.append({
                        'image_paths': paths,
                        'sample_id': batch_samples[n]['sample_id'],
                        'conversation_round': r+2
                    })
            preencode_batch_request = {'requests': preencode_image_requests}
            # Asynchronously send pre-encoding requests to next_engine
            def send_preencode_request():
                try:
                    log_print(f"Server: [Pipeline] Round {r+2} Pre-encoding image batch sent to engine {next_engine_url}")
                    response = requests.post(f"{next_engine_url}/encode_images_batch", json=preencode_batch_request, timeout=300)
                    if response.status_code == 200:
                        log_print(f"Server: [Pipeline] Round {r+2} Pre-encoding image batch sent successfully")
                    else:
                        log_print(f"Server: [Pipeline] Round {r+2} Pre-encoding image batch failed: {response.status_code}")
                except Exception as e:
                    log_print(f"Server: [Pipeline] Round {r+2} Pre-encoding image batch asynchronous send exception: {e}")
            
            # Start asynchronous thread
            threading.Thread(target=send_preencode_request, daemon=True).start()
        # 4. Current round image encoding (synchronous, must wait)
        batch_image_request = {'requests': batch_image_requests}
        try:
            response = requests.post(f"{cur_engine_url}/encode_images_batch", json=batch_image_request, timeout=300)
            response.raise_for_status()
            result = response.json()
            if result.get('success', False):
                log_print(f"Server: [Pipeline] Round {r+1} Batch image encoding completed")
            else:
                log_print(f"Server: [Pipeline] Round {r+1} Batch image encoding failed: {result.get('error', 'Unknown error')}")
        except Exception as e:
            log_print(f"Server: [Pipeline] Round {r+1} Batch image encoding exception: {e}")
        # 5. Current round text generation (synchronous, must wait)
        batch_text_requests = []
        for n in range(batch_size):
            batch_text_requests.append({
                'input_text': round_inputs[n]['input_text'],
                'sample_id': round_inputs[n]['sample_id'],
                'conversation_round': r+1,
                'is_pad': round_inputs[n]['is_pad'],
                'image_paths': round_image_paths[r][n] if r == 0 else None
            })
        batch_text_request = {'requests': batch_text_requests}
        try:
            response = requests.post(f"{cur_engine_url}/generate_batch", json=batch_text_request, timeout=300)
            response.raise_for_status()
            result = response.json()
            if result.get('success', False):
                round_outputs = result.get('results', [])
                log_print(f"Server: [Pipeline] Round {r+1} Batch text generation completed, obtained {len(round_outputs)} outputs")
                for n, output_info in enumerate(round_outputs):
                    output_text = output_info.get('output_text', '')
                    if round_inputs[n]['is_pad']:
                        output_text = ''
                    outputs[n].append(output_text)
                    # Update cur_input
                    if r+1 < max_round:
                        if all_input_text[n][r+1] == '<pad>' or cur_input[n] == '<pad>':
                            cur_input[n] = '<pad>'
                        else:
                            cur_input[n] = cur_input[n] + output_text + all_input_text[n][r+1]
                    log_print(f"Server: Sample {batch_samples[n]['sample_id']} Round {r+1} Output: {output_text}")
            else:
                log_print(f"Server: [Pipeline] Round {r+1} Batch text generation failed: {result.get('error', 'Unknown error')}")
        except Exception as e:
            log_print(f"Server: [Pipeline] Round {r+1} Batch text generation exception: {e}")
            for n in range(batch_size):
                outputs[n].append('ERROR')
                if r+1 < max_round:
                    cur_input[n] = '<pad>'
    # 4. Return the outputs list for each sample
    for n in range(batch_size):
        results.append({
            'sample_id': batch_samples[n]['sample_id'],
            'outputs': outputs[n],
            'success': True
        })
    log_print(f"Server: [Pipeline] Batch inference completed, returning outputs for all rounds of each sample")

    log_print(f"=" * 100)
    log_print(f"Server: results: {results}")
    
    return {'results': results, 'success': True}

@app.get("/health")
def health_check():
    return {"status": "ok", "engines": ENGINE_URLS, "engine_status": engine_status}

# --- Main Execution ---
if __name__ == "__main__":
    import uvicorn
    # Get port from environment variable, default to 9000.
    port = int(os.getenv("CORE_SERVER_PORT", "9000"))
    uvicorn.run(app, host="0.0.0.0", port=port) 