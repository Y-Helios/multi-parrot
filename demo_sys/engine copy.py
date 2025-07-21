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
parser.add_argument("--image-size", type=int, default="384", help="Image size for processing (width,height).")
parser.add_argument("--alfred-json", type=str, default="/scratch/jsb5jd/LLaVA-NeXT/interleave_data/ALFRED_short.json", help="Path to ALFRED.json.")
parser.add_argument("--batch-size", type=int, default=4, help="Batch size for local inference.")
parser.add_argument("--max-samples", type=int, default=12, help="Max number of samples to process (0=all).")
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
    image_size = args.image_size,
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
    
    # Ensure tensor is on the correct device
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

    # System prompt
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

def load_and_process_images(all_image_paths, vision_tower_device):

    images = [Image.open(args.image_folder + p).convert("RGB") for p in all_image_paths]
    image_tensor = image_processor.preprocess(images=images, return_tensors='pt', size=args.image_size, device=vision_tower_device)['pixel_values']
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
    batch
    """        

    all_input_texts = input_texts
    all_image_paths = image_paths

    image_encode_start = time.time()
    print(f"Engine: image_encode_start: {image_encode_start}")
    global global_image_embeds
    global_image_embeds = None

    if all_input_texts is not None:
        all_has_images = ["<image>" in s for s in all_input_texts]
        batch_input_texts = []
        batch_input_ids = []

        for idx, q in enumerate(all_input_texts):
            input_ids, raw_text = preprocess_qwen_infer(
                # [{"from": "human", "value": q}, {"from": "gpt", "value": None}],
                [{"from": "human", "value": q}],
                tokenizer,
                has_image=all_has_images[idx]
            )
            batch_input_ids.append(input_ids)
            batch_input_texts.append(raw_text)

        batch_input_ids = tokenizer(batch_input_texts, padding="longest")

        if torch.cuda.is_available():
                current_device = torch.cuda.current_device()
                batch_input_ids = torch.tensor(batch_input_ids["input_ids"], dtype=torch.long).to(device=f"cuda:{current_device}")

        attention_mask = torch.zeros((batch_input_ids.shape[0], batch_input_ids.shape[1]), dtype=torch.long)
        # attention_mask = torch.ones((len(input_ids_list), max_len), dtype=torch.long)
        for i, seq in enumerate(batch_input_ids):
            pad_count = count_leading_token_1d(seq)
            attention_mask[i, pad_count:] = 1

        # Ensure on the correct device
        device = next(model.parameters()).device
        batch_input_ids = batch_input_ids.to(device=device)
        attention_mask = attention_mask.to(device=device)

        # Set stopping conditions
        conv = conv_templates[args.conv_mode].copy()

        if any(all_has_images):
            # text has <image>
            if any(len(x) > 0 for x in all_image_paths):
                # Image paths passed
                print(f"obtain image paths")
                vision_tower_device = model.get_model().get_vision_tower().device
                image_encode_start1 = time.time()
                image_tensors = load_and_process_images(all_image_paths, vision_tower_device)
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

            # Start prefill
            prefill_start = time.time()
            print(f"Engine: prefill_start: {prefill_start}")
            with torch.inference_mode():
            # Prepare input
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    outputs = model.forward_with_image_embeds(
                        input_ids=batch_input_ids,
                        attention_mask=attention_mask,
                        image_embeds=image_embeds,
                        image_tensors=image_tensors,
                        # pad_counts=pad_counts,
                        use_cache=True,
                        return_dict=True
                    )
                    past_key_values = outputs.past_key_values
        else:
            print(f"no image in text")
            # text does not have <image>, pure text
            prefill_start = time.time()
            print(f"Engine: prefill_start: {prefill_start}")
            with torch.inference_mode():
                # Pure text input
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    outputs = model(
                        input_ids=batch_input_ids,
                        attention_mask=attention_mask,
                        use_cache=True,
                        return_dict=True
                    )
                    past_key_values = outputs.past_key_values
        prefill_end = time.time()
        print(f"Engine: prefill_end: {prefill_end}")
        print(f"Engine: prefill_time_s: {prefill_end - prefill_start}")
        # log_gpu_status("prefill") # Delete log_gpu_status

    else:
        if all_image_paths is not None and any(len(x) > 0 for x in all_image_paths):
            # No text passed, but image paths exist, meaning pre-encoding images is needed
            print(f"no text but image paths")
            image_encode_start2 = time.time()
            vision_tower_device = model.get_model().get_vision_tower().device
            image_tensors = load_and_process_images(all_image_paths, vision_tower_device)
            image_tensors = image_tensors.to(dtype=torch.float16)
            image_embeds = model.get_image_features(images=image_tensors)
            image_encode_end2 = time.time()
            print(f"Engine: image_encode_end: {image_encode_end2}")
            print(f"Engine: image_encode_time_s: {image_encode_end2 - image_encode_start2}")
            # log_gpu_status("image_encode") # Delete log_gpu_status

            device = next(model.parameters()).device
            image_embeds = image_embeds.to(device=device)
            image_tensors = image_tensors.to(device=device)
            global_image_embeds = image_embeds
        else:
            # No text passed, and no image path passed, meaning no image
            print(f"Error: no text neither image path")
            return None


    # Batch decode - batch processing
    batch_size = len(batch_input_ids)
    generated_ids = [[] for _ in range(batch_size)]

    # Ensure all tensors are on the same device
    device = next(model.parameters()).device
    current_input_ids = batch_input_ids.clone().to(device)
    attention_mask = torch.ones_like(current_input_ids, dtype=torch.float16, device=device)
    prefill_len = current_input_ids.shape[1]
    cur_past_key_values = past_key_values
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
    eos_token_id = torch.tensor(tokenizer.eos_token_id, device=device)

    decode_start = time.time()
    print(f"Engine: decode_start: {decode_start}")
    for step in range(args.max_new_tokens):
    # for step in range(20):
        with torch.inference_mode():
            if step == 0:
                next_token_logits = outputs.logits[:, -1, :]
                attention_mask_step = torch.ones((batch_size, 1), dtype=torch.long, device=device)
            else:
                input_ids_step = current_input_ids[:, -1:]
                position_ids = torch.full((batch_size, 1), prefill_len + step - 1, dtype=torch.long, device=device)

                # attention_mask_step = (~finished).long().unsqueeze(1)

                attention_mask_step = torch.ones_like(input_ids_step, dtype=torch.long, device=device)

                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    outputs = model.forward_with_image_embeds(
                        input_ids=input_ids_step,
                        attention_mask=attention_mask_step,
                        position_ids=position_ids,
                        past_key_values=cur_past_key_values,
                        use_cache=True,
                        return_dict=True
                    )
                    next_token_logits = outputs.logits[:, -1, :]
                    cur_past_key_values = outputs.past_key_values

            # Mask logits for finished samples
            next_token_logits[finished] = float('-inf')
            next_token_logits[finished, eos_token_id.item()] = 0.0

            # 采样
            if args.temperature > 0:
                logits = next_token_logits / args.temperature
                top_k = 50
                top_k_logits, _ = torch.topk(logits, k=top_k, dim=-1)
                min_top_k = top_k_logits[:, -1].unsqueeze(-1)
                logits[logits < min_top_k] = float('-inf')
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                _, next_token = torch.topk(next_token_logits, k=1, dim=-1)
                next_token = next_token.squeeze(1)

            # 确保next_token在正确的设备上
            next_token = next_token.to(device=device)

            # 对于已完成的 sample，不更新 token
            # next_token = next_token.squeeze(1)
            # next_token[finished] = eos_token_id

            next_token = torch.where(finished, eos_token_id, next_token)

            # 更新生成序列
            for i in range(batch_size):
                if not finished[i]:
                    generated_ids[i].append(next_token[i].item())
                    if next_token[i].item() == eos_token_id.item():
                        finished[i] = 1
                    elif next_token[i].item() == tokenizer.encode("<|im_end|>")[0]:
                        finished[i] = 1
                    # Optional: Custom stopping condition
                    gen_text = tokenizer.decode(generated_ids[i], skip_special_tokens=True)
                    if conv.sep and gen_text.endswith(conv.sep):
                        finished[i] = True
                    #     gen_text = tokenizer.decode(generated_ids[i], skip_special_tokens=True)
                    #     if conv.sep and gen_text.endswith(conv.sep):
                    #         finished[i] = True
                    # 其他自定义停止条件可加在这里

            # 更新current_input_ids
            current_input_ids = torch.cat([current_input_ids, next_token.unsqueeze(1)], dim=1)
            attention_mask = torch.cat([
                attention_mask,
                attention_mask_step
            ], dim=1)

            # 如果全部完成，提前break
            if all(finished):
                print(f"All samples finished generating")
                break
            # if finished.all():
            #     break

    decode_end = time.time()
    print(f"Engine: decode_end: {decode_end}")
    print(f"Engine: decode_time_s: {decode_end - decode_start}")
    # log_gpu_status("decode") # 删除 log_gpu_status

    special_token_ids = set(tokenizer.all_special_ids)
    all_generated_texts = []
    for i in range(batch_size):
        ids = generated_ids[i]
        # 只保留eos前内容
        if eos_token_id.item() in ids:
            ids = ids[:ids.index(eos_token_id.item())]
        # 过滤所有特殊token
        ids = [tid for tid in ids if tid not in special_token_ids]
        gen_text = tokenizer.decode(ids, skip_special_tokens=False)
        # 去掉自定义分隔符
        if conv.sep and gen_text.endswith(conv.sep):
            gen_text = gen_text[:-len(conv.sep)]
        all_generated_texts.append(gen_text.strip())

    return all_generated_texts


# --- 主批量推理流程 ---
def main():
    # 1. 读取ALFRED数据
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

    # 2. 按batch分批处理
    batch_num = 0
    for batch_start in range(0, num_samples, batch_size):
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
                    accum[i] = 0  # pad后累计也归零
                else:
                    count = text.count('<image>')
                    accum[i] += count
                    round_counts.append(accum[i])
            image_nums.append(round_counts)
        # print(f"image_nums: {image_nums}")

        cur_input_text = ["" for _ in range(batch_size)]

        batch_num += 1

        for r in range(max_round):
        # for r in range(1):
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
                #全都有<pad>，说明这个batch到最后一轮结束了
                break

            # 轮数较少的在没有的轮次依然会输入text，已优化
            # print(f"new_cur_input_text: {new_cur_input_text}")
            # print(f"new_image_paths: {new_image_paths}")

            print(f"BATCH: {batch_num}, Round: {r+1}")
            output = generate_text_batch(new_cur_input_text, new_image_paths)
            print(f"output: {output}")

            cur_input_text = [x + y for x, y in zip(cur_input_text, output)]

    print(f"Finished")



if __name__ == "__main__":
    # print(f"transformers.__file__: {transformers.__file__}")
    # from transformers import Qwen2Model
    # import inspect
    # print(inspect.getfile(Qwen2Model))
    main() 