from itertools import cycle
import os
import datetime
import pynvml


# Log file folder and file
# os.makedirs("engine_logs", exist_ok=True)
# log_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
# bs = 64
# log_file_path = f"engine_logs/engine_bs{bs}_{log_time}.log"
# log_file = open(log_file_path, "a", encoding="utf-8", buffering=1)  # Line buffering

# Tee class and related redirection deleted

import sys

# Delete sys.stdout = Tee(sys.stdout, log_file)
# Delete sys.stderr = Tee(sys.stderr, log_file)

import argparse
import torch
import json
import time
from fastapi import FastAPI, Body
from pydantic import BaseModel
import base64
from io import BytesIO
from PIL import Image
from typing import List, Optional, Dict, Sequence, Union
from llava.constants import IMAGE_TOKEN_INDEX
import sys
import concurrent.futures
import asyncio
from typing import Tuple



# Import LLaVA related modules
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from llava.constants import IGNORE_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_TOKEN_INDEX
import transformers

import re
import math
from llava.model.llava_arch import LlavaMetaForCausalLM
from torch.cuda.amp import autocast # Import autocast


parser = argparse.ArgumentParser(description="Batch Image Loader")
parser.add_argument("--image-folder", type=str, 
                    default="/scratch/jsb5jd/LLaVA-NeXT/interleave_data/Split1/ALFRED/full/images/",
                    help="Path to the image folder")
parser.add_argument("--batch-size", type=int, default=8, help="Batch size for processing")
parser.add_argument("--run-time", type=int, default=360, help="Run time in seconds (default: 600 = 10 minutes)")
parser.add_argument("--model-path", type=str, default="/scratch/jsb5jd/LLaVA-NeXT/models/llava-next-interleave-qwen-7b", help="Path to the model")
parser.add_argument("--model-base", type=str, default=None, help="Base model name")
parser.add_argument("--torch-dtype", type=str, default="float16", help="Data type for model")
parser.add_argument("--attn-implementation", type=str, default="flash_attention_2", help="Attention implementation")
parser.add_argument("--image-size", type=int, default=120, help="Image size")

args = parser.parse_args()

# --- Model Loading ---
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.utils import disable_torch_init

disable_torch_init()
model_path = os.path.expanduser(args.model_path)
model_name = get_model_name_from_path(model_path)
overwrite_config = {
    "image_size": args.image_size,
}
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=args.model_base,
    model_name=model_name,
    # device_map='auto',  # Let transformers automatically split the model across multiple GPUs
    torch_dtype=args.torch_dtype,
    attn_implementation=args.attn_implementation,
    overwrite_config=overwrite_config
)

# Delete model.to(device=args.device)
model.resize_token_embeddings(len(tokenizer))

print(f"special_tokens_map: {tokenizer.special_tokens_map}")
print(f"additional_special_tokens: {tokenizer.additional_special_tokens}")

# print(f"model: {type(model)}")

def ensure_model_dtype_consistency():
    """Ensure that all parts of the model use the same data type"""
    if torch.cuda.is_available():
        # Convert string to torch.dtype object
        if args.torch_dtype == "float16":
            target_dtype = torch.float16
        elif args.torch_dtype == "float32":
            target_dtype = torch.float32
        elif args.torch_dtype == "bfloat16":
            target_dtype = torch.bfloat16
        else:
            target_dtype = torch.float16  # Default to float16
            print(f"Warning - Unknown dtype {args.torch_dtype}, using float16")
        
        print(f"Ensuring model dtype consistency to {target_dtype}")
        
        # Convert model to target data type
        model.to(dtype=target_dtype)
        
        # Check the main components of the model
        if hasattr(model, 'model') and hasattr(model.model, 'vision_tower'):
            model.model.vision_tower.to(dtype=target_dtype)
            # print(f"Vision tower dtype: {model.model.vision_tower.dtype}")
        
        if hasattr(model, 'model') and hasattr(model.model, 'language_model'):
            model.model.language_model.to(dtype=target_dtype)
            # print(f"Language model dtype: {model.model.language_model.dtype}")
        
        print(f"Model dtype consistency ensured")

ensure_model_dtype_consistency()

# Check model device allocation
print(f"Model device map: {model.hf_device_map if hasattr(model, 'hf_device_map') else 'No device map'}")
print(f"Model parameters device: {next(model.parameters()).device}")
print(f"Available GPUs: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    print(f"GPU {i} memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")


# Move model parts to specified device
def move_model_parts_to_device(model, device_map):
    for name, module in model.named_modules():
        if name in device_map:
            module.to(device_map[name])

# Define device map
new_device_map = {
    'vision_tower': 'cuda:0',      # Keep on GPU 1
    'vision_resampler': 'cuda:0',  # Move to GPU 0
    'mm_projector': 'cuda:0',      # Move to GPU 0
    'lm_head': 'cuda:0',            # Move to GPU 0
    'model.layers.15': 'cuda:0',
    'model.layers.16': 'cuda:0',
    'model.layers.17': 'cuda:0',
    'model.layers.18': 'cuda:0',
    'model.layers.19': 'cuda:0',
    'model.layers.20': 'cuda:0',
    'model.layers.21': 'cuda:0',
    'model.layers.22': 'cuda:0',
    'model.layers.23': 'cuda:0',
    'model.layers.24': 'cuda:0',
    'model.layers.25': 'cuda:0',
    'model.layers.26': 'cuda:0',
    'model.layers.27': 'cuda:0',
    'model.layers.28': 'cuda:0',
    'model.layers.29': 'cuda:0',
    'model.layers.30': 'cuda:0',
    'model.layers.31': 'cuda:0',
    'model.norm': 'cuda:0'
}


# Apply device map
if torch.cuda.device_count() > 1:
    move_model_parts_to_device(model, new_device_map)

print(f"Model device map: {model.hf_device_map if hasattr(model, 'hf_device_map') else 'No device map'}")



def load_and_process_images(all_image_paths, vision_tower_device):

    load_start_time = time.time()
    images = [Image.open(args.image_folder + p).convert("RGB") for p in all_image_paths]
    load_end_time = time.time()

    preprocess_start_time = time.time()
    image_tensor = image_processor.preprocess(images=images, return_tensors='pt', size=args.image_size, device=vision_tower_device)['pixel_values']
    preprocess_end_time = time.time()

    return image_tensor, load_start_time, load_end_time, preprocess_start_time, preprocess_end_time

def get_image_files_from_folder(folder_path: str) -> List[str]:
    """Get all image file paths from a specified folder"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.gif', '.webp'}
    image_paths = []
    
    if not os.path.exists(folder_path):
        print(f"Error: Folder {folder_path} does not exist!")
        return image_paths
    
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_ext = os.path.splitext(file)[1].lower()
            if file_ext in image_extensions:
                rel_path = os.path.relpath(os.path.join(root, file), folder_path)
                image_paths.append(rel_path)
    
    print(f"Found {len(image_paths)} image files in {folder_path}")
    return image_paths

batch_data_list = []

def record_batch_data(batch_count, batch_size, image_tensors, image_embeds, 
                     embed_start_time, embed_end_time,tensor_start_time, tensor_end_time, 
                     load_start_time, load_end_time, preprocess_start_time, preprocess_end_time):
    """
    Record data for a single batch into a dictionary
    """
    embed_time = embed_end_time - embed_start_time
    tensor_time = tensor_end_time - tensor_start_time
    image_time = embed_end_time - tensor_start_time
    load_time = load_end_time - load_start_time
    preprocess_time = preprocess_end_time - preprocess_start_time
    
    # Calculate memory usage
    image_tensor_memory = image_tensors.element_size() * image_tensors.nelement() / 1024 / 1024  # MB
    image_embed_memory = image_embeds.element_size() * image_embeds.nelement() / 1024 / 1024    # MB
    
    # Create data dictionary
    batch_data = {
        "batch": batch_count,
        "batch_size": batch_size,
        "load_start_time": load_start_time,
        "load_end_time": load_end_time,
        "load_time": load_time,
        "preprocess_start_time": preprocess_start_time,
        "preprocess_end_time": preprocess_end_time,
        "preprocess_time": preprocess_time,
        "image_tensor_start": tensor_start_time,
        "image_tensor_end": tensor_end_time,
        "image_tensor_time": tensor_time,
        "image_tensor_memory": round(image_tensor_memory, 2),
        "image_embed_start": embed_start_time,
        "image_embed_end": embed_end_time,
        "image_embed_time": embed_time,
        "image_embed_memory": round(image_embed_memory, 2),
        "image_total_time": image_time,
        "average_time_per_image": round(image_time / batch_size, 6),
        "average_embed_mem_per_image": round(image_embed_memory / batch_size, 2)
    }
    
    # Add to list
    batch_data_list.append(batch_data)
    
    # return batch_data

def main():
    
    all_image_paths = get_image_files_from_folder(args.image_folder)
    
    # Set batches length to 30*batch_size
    target_length = args.batch_size
    if len(all_image_paths) >= target_length:
        batches = all_image_paths[:target_length]
    else:
        # Use itertools.cycle to extend samples
        from itertools import cycle
        batches = list(next(cycle(all_image_paths)) for _ in range(target_length))

    
    # 3. Set run time
    time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    start_time = time.time()
    end_time = start_time + args.run_time
    batch_count = 0
    
    # 4. Continuously process until time ends
    while time.time() < end_time:

        # Check for timeout
        if time.time() >= end_time:
            # print(f"Time limit reached ({args.run_time} seconds), stopping...")
            break
        
        batch_count += 1
        batch_size = len(batches)

        vision_tower_device = model.get_model().get_vision_tower().device
        image_tensor_start = time.time()
        image_tensors, load_start_time, load_end_time, preprocess_start_time, preprocess_end_time = load_and_process_images(batches, vision_tower_device)
        image_tensors = image_tensors.to(dtype=torch.float16)
        image_tensor_end = time.time()

        # Auto-detect vision_tower on which GPU and put image tensor on corresponding device
        # vision_tower_device = model.get_model().get_vision_tower().device
        # print(f"Vision tower device: {vision_tower_device}")
        # image_tensors = image_tensors.to(device=vision_tower_device)

        image_embed_start = time.time()
        image_embeds = model.get_image_features(images=image_tensors)
        image_embed_end = time.time()

        record_batch_data(batch_count, batch_size, image_tensors, image_embeds, image_embed_start, image_embed_end, image_tensor_start, image_tensor_end, load_start_time, load_end_time, preprocess_start_time, preprocess_end_time)

        del image_tensors
        del image_embeds
        torch.cuda.empty_cache()

    total_time = time.time() - start_time
    print(f"Finished after {total_time:.1f} seconds, processed {batch_count} batches")

    # Create a directory with write permissions (e.g., your scratch space)
    log_dir = "/scratch/jsb5jd/LLaVA-NeXT/demo/data_logs"
    os.makedirs(log_dir, exist_ok=True)
    # Build save path
    file_path = os.path.join(log_dir, f"img_bs_encode_bs{args.batch_size}_size{args.image_size}_{time_str}.json")
    with open(file_path, "w") as f:
        json.dump(batch_data_list, f, indent=4)

if __name__ == "__main__":
    main() 