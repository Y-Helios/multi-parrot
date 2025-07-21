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

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description="LLaVA-NeXT Engine Local Batch Runner")
parser.add_argument("--model-path", type=str, default="/scratch/jsb5jd/LLaVA-NeXT/models/llava-next-interleave-qwen-7b", help="Path to the LLaVA model.")
parser.add_argument("--model-base", type=str, default=None, help="Base model path (optional).")
parser.add_argument("--device", type=str, default="cuda:0", help="Device to run the model on (e.g., 'cuda:0', 'cpu').")
parser.add_argument("--torch-dtype", type=str, default="float16", help="Torch dtype for model loading.")
parser.add_argument("--attn-implementation", type=str, default="flash_attention_2", help="Attention implementation.")
parser.add_argument("--conv-mode", type=str, default="qwen_1_5", help="Conversation mode.")
parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for generation.")
parser.add_argument("--num-beams", type=int, default=5, help="Number of beams for generation.")
parser.add_argument("--image-folder", type=str, default="/scratch/jsb5jd/LLaVA-NeXT/interleave_data/", help="Base folder for images.")
parser.add_argument("--image-size", type=str, default="300,300", help="Image size for processing (width,height).")
parser.add_argument("--alfred-json", type=str, default="/scratch/jsb5jd/LLaVA-NeXT/interleave_data/ALFRED_380.json", help="Path to ALFRED.json.")
parser.add_argument("--batch-size", type=int, default=128, help="Batch size for local inference.")
parser.add_argument("--max-samples", type=int, default=1024, help="Max number of samples to process (0=all).")
parser.add_argument("--max-new-tokens", type=int, default=1024, help="Maximum new tokens to generate.")
args = parser.parse_args()

# --- Model Loading ---
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.utils import disable_torch_init

disable_torch_init()
model_path = os.path.expanduser(args.model_path)
model_name = get_model_name_from_path(model_path)
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=args.model_base,
    model_name=model_name,
    device_map='auto',  # Let transformers automatically split the model across multiple GPUs
    torch_dtype=args.torch_dtype,
    attn_implementation=args.attn_implementation
)

# Delete model.to(device=args.device)
model.resize_token_embeddings(len(tokenizer))

# Delete DataParallel related code
# print(f"tokenizer.padding_side: {tokenizer.padding_side}")

print(f"model: {type(model)}")

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

# GPU device check
def check_gpu_device():
    """Check the GPU device of the current process"""
    if torch.cuda.is_available():
        current_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_device)
        device_memory = torch.cuda.get_device_properties(current_device).total_memory / 1024**3
        cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')
        
        print(f"Using GPU {current_device}: {device_name} ({device_memory:.1f} GB)")
        print(f"CUDA_VISIBLE_DEVICES: {cuda_visible_devices}")
        
        # Display all visible GPUs
        gpu_count = torch.cuda.device_count()
        print(f"Total visible GPUs: {gpu_count}")
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
        
        # Display physical GPU ID (if possible)
        if cuda_visible_devices != 'Not set':
            physical_gpu_id = cuda_visible_devices.split(',')[current_device]
            print(f"Physical GPU ID: {physical_gpu_id}")
    else:
        print(f"No CUDA available, using CPU")

check_gpu_device()


# --- ALFRED data batch reading and processing tool ---
def read_alfred_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def extract_batch_input(samples):
    batch_input_text = []
    batch_image_paths = []
    for sample in samples:
        conversations = sample.get("conversations", [])
        input_text = [conv["value"] for conv in conversations if conv.get("from") == "human"]
        image_paths = sample.get("image", [])
        batch_input_text.append(input_text)
        batch_image_paths.append(image_paths)
    return batch_input_text, batch_image_paths

def count_image_tokens(text):
    return len(re.findall(r'<image>', text))

def get_round_image_paths(input_texts, image_paths):
    round_image_paths = []
    acc = 0
    for text in input_texts:
        acc += count_image_tokens(text)
        round_image_paths.append(image_paths[:acc])
    return round_image_paths

def preprocess_qwen(sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False, max_len=2048, system_message: str = "You are a helpful assistant.") -> torch.Tensor:
    """Same preprocess_qwen function as model_vqa.py"""
    roles = {"human": "<|im_start|>user", "gpt": "<|im_start|>assistant"}

    im_start, im_end = tokenizer.additional_special_tokens_ids
    nl_tokens = tokenizer("\n").input_ids
    _system = tokenizer("system").input_ids + nl_tokens
    _user = tokenizer("user").input_ids + nl_tokens
    _assistant = tokenizer("assistant").input_ids + nl_tokens

    # Apply prompt templates
    input_ids, targets = [], []

    source = sources
    if roles[source[0]["from"]] != roles["human"]:
        source = source[1:]

    input_id, target = [], []
    system = [im_start] + _system + tokenizer(system_message).input_ids + [im_end] + nl_tokens
    input_id += system
    target += [im_start] + [IGNORE_INDEX] * (len(system) - 3) + [im_end] + nl_tokens
    assert len(input_id) == len(target)
    for j, sentence in enumerate(source):
        role = roles[sentence["from"]]
        if has_image and sentence["value"] is not None and "<image>" in sentence["value"]:
            num_image = len(re.findall(DEFAULT_IMAGE_TOKEN, sentence["value"]))
            texts = sentence["value"].split('<image>')
            _input_id = tokenizer(role).input_ids + nl_tokens 
            for i,text in enumerate(texts):
                _input_id += tokenizer(text).input_ids 
                if i<len(texts)-1:
                    _input_id += [IMAGE_TOKEN_INDEX] + nl_tokens
            _input_id += [im_end] + nl_tokens
            assert sum([i==IMAGE_TOKEN_INDEX for i in _input_id])==num_image
        else:
            if sentence["value"] is None:
                _input_id = tokenizer(role).input_ids + nl_tokens
            else:
                _input_id = tokenizer(role).input_ids + nl_tokens + tokenizer(sentence["value"]).input_ids + [im_end] + nl_tokens
        input_id += _input_id
        if role == "<|im_start|>user":
            _target = [im_start] + [IGNORE_INDEX] * (len(_input_id) - 3) + [im_end] + nl_tokens
        elif role == "<|im_start|>assistant":
            _target = [im_start] + [IGNORE_INDEX] * len(tokenizer(role).input_ids) + _input_id[len(tokenizer(role).input_ids) + 1 : -2] + [im_end] + nl_tokens
        else:
            raise NotImplementedError
        target += _target

    input_ids.append(input_id)
    targets.append(target)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)
    
    # Ensure tensor is on the correct GPU
    if torch.cuda.is_available():
        # Get the GPU device of the current process
        current_device = torch.cuda.current_device()
        input_ids = input_ids.to(device=f"cuda:{current_device}")
        print(f"Input IDs tensor moved to device: cuda:{current_device}")
    
    return input_ids

def preprocess_qwen_infer(
    sources: List[Dict[str, str]],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    system_message: str = "You are a helpful assistant."
) -> Tuple[torch.Tensor, str]:
    """
    Construct ChatML format prompt for inference.
    Returns: (input_ids: torch.LongTensor, raw_text: str)
    """
    roles = {"human": "<|im_start|>user", "gpt": "<|im_start|>assistant"}

    # Special token IDs
    im_start, im_end = tokenizer.additional_special_tokens_ids
    nl_tokens = tokenizer("\n").input_ids
    _system = tokenizer("system").input_ids + nl_tokens
    _user = tokenizer("user").input_ids + nl_tokens
    _assistant = tokenizer("assistant").input_ids + nl_tokens

    input_id = []
    raw_text = ""

    # system prompt
    system_tokens = [im_start] + _system + tokenizer(system_message).input_ids + [im_end] + nl_tokens
    input_id += system_tokens
    raw_text += f"<|im_start|>system\n{system_message}<|im_end|>\n"

    # If the first speaker is not human, skip it (not compliant)
    source = sources
    if roles[source[0]["from"]] != roles["human"]:
        source = source[1:]

    # Previous history of conversations
    for sentence in source[:-1]:
        role = roles[sentence["from"]]
        value = sentence["value"]

        if has_image and value is not None and "<image>" in value:
            # Process image content
            num_image = len(re.findall(DEFAULT_IMAGE_TOKEN, value))
            texts = value.split(DEFAULT_IMAGE_TOKEN)
            _input_id = tokenizer(role).input_ids + nl_tokens
            text_raw = f"<|im_start|>{role.split('|>')[-1]}\n"
            for i, text in enumerate(texts):
                _input_id += tokenizer(text).input_ids
                text_raw += text
                if i < len(texts) - 1:
                    _input_id += [IMAGE_TOKEN_INDEX] + nl_tokens
                    text_raw += "<image>\n"
            _input_id += [im_end] + nl_tokens
            text_raw += "<|im_end|>\n"
        else:
            # Normal text
            text_raw = f"<|im_start|>{role.split('|>')[-1]}\n{value}<|im_end|>\n"
            _input_id = tokenizer(role).input_ids + nl_tokens + tokenizer(value).input_ids + [im_end] + nl_tokens

        input_id += _input_id
        raw_text += text_raw

    # The last round is the user query, followed by assistant to wait for model generation
    last_user = source[-1]
    if roles[last_user["from"]] != "<|im_start|>user":
        raise ValueError("The last round must be a user query to trigger generation")

    query_value = last_user["value"]
    query_text = f"<|im_start|>user\n{query_value}<|im_end|>\n<|im_start|>assistant\n"
    raw_text += query_text

    _input_id = tokenizer("<|im_start|>user\n").input_ids \
        + tokenizer(query_value).input_ids \
        + tokenizer("\n").input_ids \
        + tokenizer("<|im_end|>\n<|im_start|>assistant\n").input_ids

    input_id += _input_id

    # Convert to tensor
    input_ids = torch.tensor([input_id])

    if torch.cuda.is_available():
        current_device = torch.cuda.current_device()
        input_ids = input_ids.to(device=f"cuda:{current_device}")
        # print(f"[INFO] input_ids moved to cuda:{current_device}")

    return input_ids, raw_text

# Parse image size parameter
width, height = map(int, args.image_size.split(','))

def load_and_process_images(all_image_paths):

    # print(f"all_image_paths: {all_image_paths}")
    images = [Image.open(args.image_folder + p).convert("RGB").resize((width, height)) for sublist in all_image_paths for p in sublist]

    # print(f"images: {len(images)}")
    image_tensor = image_processor.preprocess(images, return_tensors='pt')['pixel_values']
    # Previously set dtype = torch.long, but it consumes too much memory, causing errors
    image_tensor = image_tensor.to(dtype=torch.float32)

    return image_tensor


def count_leading_token_1d(tensor_1d, token_id=151646):
    count = 0
    for t in tensor_1d:
        if t == token_id:
            count += 1
        else:
            break
    return count


def generate_text_batch(input_texts, image_paths):
    """
    Real batch text generation: process all requests at once, supporting pad processing
    """        

    all_input_texts = input_texts
    all_image_paths = image_paths

    image_encode_start = time.time()
    print(f"Engine: image_encode_start: {image_encode_start}")
    global global_image_embeds
    global_image_embeds = None

    if all_input_texts is not None:
        all_has_images = ["<image>" in s for s in all_input_texts]

        # Ensure on the correct device
        device = next(model.parameters()).device

        if any(all_has_images):
            # text has <image>
            if any(len(x) > 0 for x in all_image_paths):
                # image paths passed
                print(f"obtain image paths")
                image_encode_start1 = time.time()
                image_tensors = load_and_process_images(all_image_paths)
                image_tensors = image_tensors.to(dtype=torch.float16)
                image_embeds = model.get_image_features(images=image_tensors)

                image_encode_end1 = time.time()
                print(f"Engine: image_encode_end: {image_encode_end1}")
                print(f"Engine: image_encode_time_s: {image_encode_end1 - image_encode_start1}")
                # log_gpu_status("image_encode") # Delete log_gpu_status

                # print(f"image_embeds: {image_embeds.shape}")
                # print(f"image_tensors: {image_tensors.shape}")
                device = next(model.parameters()).device
                image_embeds = image_embeds.to(device=device)
                image_tensors = image_tensors.to(device=device)
                print(f"image_embeds memory: {image_embeds.element_size() * image_embeds.nelement() / 1024 / 1024:.2f} MB")
                
                # Calculate number of images and average memory size
                total_images = sum(len(img_list) for img_list in all_image_paths)
                avg_memory_per_image = (image_embeds.element_size() * image_embeds.nelement() / 1024 / 1024) / total_images if total_images > 0 else 0
                print(f"total images embedded: {total_images}")
                print(f"average memory per image: {avg_memory_per_image:.2f} MB")
                print(f"average time per image: {(image_encode_end1 - image_encode_start1) / total_images:.6f} s")
            else:
                # No image path passed, possibly pre-coded
                print(f"loading pre-encoded embeds")
                if global_image_embeds is not None:
                    # print(f"global_image_embeds shape: {global_image_embeds.shape}")
                    device = next(model.parameters()).device
                    image_embeds = global_image_embeds.to(device=device)
                    image_tensors = None
                    global_image_embeds = None  # Avoid subsequent mixing
                else:
                    print(f"did not pre-encode images, and did not pass image paths")
    
    return "finished"


# --- Main batch inference process ---
def main():
    # 1. Read ALFRED data
    alfred_samples = read_alfred_json(args.alfred_json)
    # print(f"alfred_samples: {alfred_samples}")
    if args.max_samples > 0:
        alfred_samples = alfred_samples[:args.max_samples]

    print(f"{len(alfred_samples)} samples in total.")
    batch_input_text, batch_image_paths = extract_batch_input(alfred_samples)
    # print(f"batch_input_text: {batch_input_text}")
    # print(f"batch_image_paths: {batch_image_paths}")
    num_samples = len(batch_input_text)
    # max_round = max(len(x) for x in batch_input_text)
    # max_round = 1
    batch_size = args.batch_size

    # Set run time to 10 minutes
    start_time = time.time()
    run_duration = 10 * 60  # 10 minutes = 600 seconds
    batch_num = 0
    
    print(f"Starting continuous batch processing for {run_duration} seconds...")
    
    # 2. Continuously process batches until time ends
    while time.time() - start_time < run_duration:
        # Loop through all samples
        for batch_start in range(0, num_samples, batch_size):
            # Check for timeout
            if time.time() - start_time >= run_duration:
                print(f"Time limit reached ({run_duration} seconds), stopping...")
                break
                
            batch_end = min(batch_start + batch_size, num_samples)
            batch_texts = batch_input_text[batch_start:batch_end]
            batch_images = batch_image_paths[batch_start:batch_end]
            # print(f"batch_texts: {batch_texts}")
            # print(f"batch_images: {batch_images}")
            max_round = max(len(x) for x in batch_texts)
            batch_ids = list(range(batch_start, batch_end))
            
            input_texts = []
            image_nums = []
            for r in range(max_round):
                input_text = []
                for b in range(len(batch_ids)):
                    if r < len(batch_texts[b]):
                        text = batch_texts[b][r]
                        input_text.append(text)
                    else:
                        text = '<pad>'
                        input_text.append(text)
                input_texts.append(input_text)
            # print(f"input_texts: {input_texts}")

            accum = [0] * len(input_texts[0])
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

            batch_num += 1
            elapsed_time = time.time() - start_time
            print(f"Processing batch {batch_num} (elapsed: {elapsed_time:.1f}s)")

            for r in range(max_round):
            # for r in range(1):
                # Check for timeout
                if time.time() - start_time >= run_duration:
                    break
                    
                image_paths = []
                for i, images in enumerate(batch_images):
                    a = image_nums[r][i]
                    image_paths.append(images[:a])
                # print(f"image_paths{r+1}: {image_paths}")

                cur_input_text = [x + y for x, y in zip(cur_input_text, input_texts[r])]

                filtered = [
                    (text, img_list)
                    for text, img_list in zip(cur_input_text, image_paths)
                    if "<pad>" not in text
                ]

                if filtered:
                    new_cur_input_text, new_image_paths = zip(*filtered)
                    new_cur_input_text = list(new_cur_input_text)
                    new_image_paths = list(new_image_paths)
                else:
                    # All have <pad>, indicating this batch has reached the last round
                    break

                # Optimized to input text even if rounds are missing
                # print(f"new_cur_input_text: {new_cur_input_text}")
                # print(f"new_image_paths: {new_image_paths}")

                print(f"BATCH: {batch_num}, Round: {r+1}")
                output = generate_text_batch(new_cur_input_text, new_image_paths)
                # print(f"output: {output}")

                cur_input_text = [x + y for x, y in zip(cur_input_text, output)]

    total_time = time.time() - start_time
    print(f"Finished after {total_time:.1f} seconds, processed {batch_num} batches")



if __name__ == "__main__":
    # print(f"transformers.__file__: {transformers.__file__}")
    # from transformers import Qwen2Model
    # import inspect
    # print(inspect.getfile(Qwen2Model))
    main() 