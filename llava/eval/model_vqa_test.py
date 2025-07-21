import argparse
import torch
import os
import json
import time
import math
import pynvml
from tqdm import tqdm
from PIL import Image
import transformers
import re

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import get_model_name_from_path

def print_gpu_util(note=""):
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
    print(f"[GPU {note}] Mem: {meminfo.used / 1024 ** 2:.1f} MB | Util: {util.gpu}%")
    print(f"[CUDA Allocated] {torch.cuda.memory_allocated() / 1024 ** 2:.1f} MB")

def preprocess_qwen(sources, tokenizer, has_image=False, max_len=2048, system_message="You are a helpful assistant."):
    roles = {"human": "<|im_start|>user", "gpt": "<|im_start|>assistant"}
    im_start, im_end = tokenizer.additional_special_tokens_ids
    nl_tokens = tokenizer("\n").input_ids
    _system = tokenizer("system").input_ids + nl_tokens

    input_ids, targets = [], []
    source = sources if roles[sources[0]["from"]] == roles["human"] else sources[1:]
    input_id = [im_start] + _system + tokenizer(system_message).input_ids + [im_end] + nl_tokens
    target = [im_start] + [IGNORE_INDEX] * (len(input_id) - 3) + [im_end] + nl_tokens

    for sentence in source:
        role = roles[sentence["from"]]
        if has_image and sentence["value"] and "<image>" in sentence["value"]:
            texts = sentence["value"].split('<image>')
            _input_id = tokenizer(role).input_ids + nl_tokens
            for i, text in enumerate(texts):
                _input_id += tokenizer(text).input_ids
                if i < len(texts) - 1:
                    _input_id += [IMAGE_TOKEN_INDEX] + nl_tokens
            _input_id += [im_end] + nl_tokens
        else:
            _input_id = tokenizer(role).input_ids + nl_tokens + tokenizer(sentence["value"]).input_ids + [im_end] + nl_tokens
        input_id += _input_id
    input_ids.append(input_id)
    return torch.tensor(input_ids, dtype=torch.long)

def profile_image_encode(image_paths, image_processor, model):
    images = [Image.open(p) for p in image_paths]
    start = time.time()
    image_tensors = [image_processor(img, return_tensors='pt')['pixel_values'].half().cuda() for img in images]
    image_tensors = torch.cat(image_tensors, dim=0)
    torch.cuda.synchronize()
    mid = time.time()
    image_embeds = model.encode_images(image_tensors)
    torch.cuda.synchronize()
    end = time.time()
    print(f"[Image Encode] Pre+Enc: {end - start:.2f}s | Encode: {end - mid:.2f}s")
    print_gpu_util("Image Encode")
    return image_embeds

def profile_prefill(input_ids, image_embeds, model):
    attention_mask = torch.ones_like(input_ids)
    torch.cuda.synchronize()
    start = time.time()
    output = model(input_ids=input_ids, images=image_embeds, attention_mask=attention_mask, use_cache=True)
    torch.cuda.synchronize()
    end = time.time()
    print(f"[Prefill] Time: {end - start:.2f}s")
    print_gpu_util("Prefill")
    return output

def profile_decode(input_ids, model, tokenizer, image_embeds=None, max_new_tokens=20):
    generated = input_ids
    past_key_values = None
    total_decode_time = 0.0

    for step in range(max_new_tokens):
        torch.cuda.synchronize()
        start = time.time()
        outputs = model(input_ids=generated[:, -1:], images=image_embeds if step == 0 else None,
                        use_cache=True, past_key_values=past_key_values)
        torch.cuda.synchronize()
        end = time.time()
        step_time = end - start
        total_decode_time += step_time

        logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(logits, dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)
        past_key_values = outputs.past_key_values

        print(f"[Decode Token {step}] Time: {step_time:.3f}s")
        if next_token.item() == tokenizer.eos_token_id:
            break

    print(f"[Decode Total] Time: {total_decode_time:.2f}s")
    print_gpu_util("Decode")
    return generated

def main(args):
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    with open(os.path.expanduser(args.question_file)) as f:
        questions = json.load(f)
    questions = questions[:args.num_samples]

    total_times = {"encode": [], "prefill": [], "decode": []}

    for line in tqdm(questions):
        qs = line["conversations"][0]["value"]
        input_ids = preprocess_qwen([line["conversations"][0], {"from": "gpt", "value": None}], tokenizer, has_image=True).cuda()
        image_paths = [os.path.join(args.image_folder, p) for p in line["image"]]

        t0 = time.time()
        image_embeds = profile_image_encode(image_paths, image_processor, model)
        total_times["encode"].append(time.time() - t0)

        t1 = time.time()
        prefill_output = profile_prefill(input_ids, image_embeds, model)
        total_times["prefill"].append(time.time() - t1)

        t2 = time.time()
        output_ids = profile_decode(input_ids, model, tokenizer, image_embeds=image_embeds, max_new_tokens=args.max_tokens)
        total_times["decode"].append(time.time() - t2)

        print("Pred:", tokenizer.decode(output_ids[0], skip_special_tokens=True))

    print("\n=== Average Times ===")
    for k in total_times:
        avg_time = sum(total_times[k]) / len(total_times[k])
        print(f"[{k.upper()}] Avg Time: {avg_time:.2f}s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, required=True)
    parser.add_argument("--question-file", type=str, required=True)
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=20)
    args = parser.parse_args()
    main(args)
