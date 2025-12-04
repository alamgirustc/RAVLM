# RAVLM: Region-Aware Vision–Language Model for Image Captioning

This repository contains the official implementation of:

> **Region-Aware Vision–Language Model for Image Captioning (RAVLM)**  
> Mohammad Alamgir Hossain, Md. Ibrahim Abdullah, ZhongFu Ye,  
> Md Bipul Hossen, Md. Atiqur Rahman, Md Shohidul Islam

RAVLM augments a strong patch-based vision–language backbone with **detector-based region features** for image captioning.  
The model is implemented on top of the **mPLUG** architecture and uses **bottom-up Faster R-CNN region features** generated following the official **Bottom-Up and Top-Down Attention** implementation:

- **mPLUG (official repo):** <https://github.com/X-PLUG/mPLUG>  
- **Bottom-Up Attention (official repo):** <https://github.com/peteanderson80/bottom-up-attention>  

> 🔗 RAVLM code: <https://github.com/alamgirustc/RAVLM>


## Overview

Recent vision–language models for image captioning often rely purely on dense ViT patch embeddings, leaving explicit object-region information unused.  
RAVLM revisits **region-aware modeling** and shows that lightweight detector features remain complementary to patch-based encoders.

**High-level idea:**

- Use **mPLUG** as the vision–language backbone (ViT + BERT + cross-modal skip-connections + PrefixLM decoder).
- Follow **bottom-up-attention** to obtain **Faster R-CNN-based region features** (ROI features, bounding boxes, and backbone region descriptors).
- Project region descriptors into the same embedding space as ViT patch tokens.
- Concatenate **patch tokens + region tokens**, and feed the combined sequence into the original mPLUG fusion network and decoder.
- Train with **cross-entropy (XE)** only (no CIDEr-based RL).

On MS COCO (Karpathy split), the best RAVLM configuration (patch + Faster R-CNN backbone tokens) achieves:

- **BLEU-4:** 45.4  
- **CIDEr:** 151.7  

using the same 14M-image pre-training as mPLUG and identical fine-tuning settings to the patch-only mPLUG captioning baseline.


## News

- **2025.xx.xx** – Initial public release of RAVLM.
- **2025.xx.xx** – Added configs for:
  - Full ROI (features + boxes),
  - Geometry-only ROI,
  - Patch + Faster R-CNN backbone features (recommended RAVLM setting).


## Model Variants

RAVLM is implemented as a small extension to the original **[mPLUG](https://github.com/X-PLUG/mPLUG)** captioning framework.

### Visual Configurations

We consider the following visual input variants:

1. **Patch-only (mPLUG baseline)**  
   - Only ViT patch tokens (no region features).  
   - This corresponds to the original mPLUG captioning setup.

2. **Full ROI (features + boxes)**  
   - For each region, concatenate ROI appearance features (from Faster R-CNN) with normalized bounding-box geometry.  
   - Project the concatenated vector to the model hidden size and append as region tokens.

3. **Geometry-only ROI**  
   - Use only normalized bounding-box geometry + simple geometric attributes (width, height, area).  
   - Project geometry into the hidden space and append as region tokens.

4. **RAVLM (Patch + Faster R-CNN backbone features)**  
   - Use ViT patch tokens + region tokens derived from Faster R-CNN backbone features (following the **[bottom-up-attention](https://github.com/peteanderson80/bottom-up-attention)** detector).  
   - No explicit concatenation of ROI appearance + geometry—just clean region descriptors as additional tokens.  
   - This is the **best-performing** configuration and the default RAVLM model.

### COCO Captioning Results (Karpathy test, XE only)

All models are trained with **cross-entropy only** (no CIDEr RL) on the COCO Karpathy split, starting from the same mPLUG base checkpoint.

| Visual Configuration                      | B@4 | METEOR | CIDEr | SPICE |
|------------------------------------------|:---:|:------:|:-----:|:-----:|
| Patch-only (mPLUG baseline)              | 43.1 | 31.4 | 141.0 | 24.2 |
| Full ROI (features + boxes)              | 43.8 | 31.5 | 147.9 | 24.1 |
| Geometry-only ROI                        | 44.5 | 31.5 | 149.7 | 24.1 |
| **RAVLM (Patch + Faster R-CNN feat.)**   | **45.4** | **31.8** | **151.7** | **24.4** |


## Requirements

- Linux
- Python ≥ 3.8
- [PyTorch](https://pytorch.org/) ≥ 1.11.0
- CUDA-compatible GPUs (experiments in the paper use 8× Tesla V100 16GB)

Install Python dependencies:

```bash
pip install -r requirements.txt
