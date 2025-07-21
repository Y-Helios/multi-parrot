# README

## Preparation

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Download the model from [LLaVA-NeXT](https://huggingface.co/lmms-lab/llava-next-interleave-qwen-7b/tree/main) and place it under the directory:
   ```
   /LLaVA-NeXT/models/
   ```

3. Download the dataset by following the instructions in the official LLaVA-NeXT repository. The dataset used is **ALFRED**.

4. Modify the image encoding file:

   Run the following Python code in the terminal to locate the path of `siglip_encoder.py`, then replace that file with `LLaVA-NeXT/siglip_encoder.py`:
   ```python
   import inspect
   from llava.model.multimodal_encoder.siglip_encoder import SigLipImageProcessor
   print(inspect.getfile(SigLipImageProcessor))
   ```

---

## Introduction

1. In `siglip_encoder.py`, the functions `preprocess()` and `preprocess_gpu()` are used to convert image data into image tensors.  
   - `preprocess()` uses CPU (original implementation)  
   - `preprocess_gpu()` uses GPU (modified version)

   If you want to use GPU preprocessing, modify the `load_and_process_image()` function in the engine to call `image_processor.preprocess_gpu()` instead.

2. `demo_image` is used to independently test image embedding.

   Test different batch sizes (e.g., batch size = 4):
   ```bash
   bash monitor_image_encode.sh 4
   ```

   Test different image sizes (e.g., image size = 300):
   ```bash
   bash monitor_image_size.sh 300
   ```

3. `demo_sys` builds a complete end-to-end workflow and implements initial support for image pre-encoding.

   Basic version using 2 engines:

   Test different batch sizes:
   ```bash
   bash run.sh 4
   ```

   Test different image sizes:
   ```bash
   bash run_img_size.sh 300
   ```

   Single-engine versions:
   ```bash
   bash run_single_engine.sh 4
   bash run_single_diff.sh 300
   ```

   To try running with image pre-encoding:
   ```bash
   bash run_pipeline.sh 4
   ```

   ⚠️ Note: Pre-encoding is only partially implemented and may have issues. The JSON output files are also not finalized yet.

4. `demo_batch` contains the batch processing implementation.  
   If `demo_sys` does not work, you can fall back to this version and continue modifying it (not recommended).

---

## Others

For more details, please refer to:

- [LLaVA-NeXT Interleave Documentation](https://github.com/LLaVA-VL/LLaVA-NeXT/blob/main/docs/LLaVA-NeXT-Interleave.md)
- [Parrot (ParrotServe by Microsoft)](https://github.com/microsoft/ParrotServe)
