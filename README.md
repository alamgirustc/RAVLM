# RAVLM: Region-Aware Vision–Language Model for Image Captioning

This repository contains the official implementation of:

> **Region-Aware Vision–Language Model for Image Captioning (RAVLM)**  
> Mohammad Alamgir Hossain, Md. Ibrahim Abdullah, ZhongFu Ye, Md Bipul Hossen,  
> Md. Atiqur Rahman, Md Shohidul Islam

RAVLM augments a strong patch-based vision–language backbone with **detector-based region features** for image captioning.  
Building on top of the mPLUG architecture, RAVLM injects Faster R-CNN region tokens into the visual stream during fine-tuning, improving caption quality **without** any CIDEr-based reinforcement learning.

> 📎 Code: https://github.com/alamgirustc/RAVLM


## Overview

Recent vision–language models for image captioning often rely solely on dense patch tokens, implicitly encoding objects and regions. RAVLM revisits explicit **region-aware modeling** and shows that lightweight detector features still provide complementary benefits over ViT patches.

**Key ideas:**

- Start from a **patch-only** mPLUG captioning baseline.
- Extract **Faster R-CNN** region features (appearance + geometry / backbone tokens).
- Project region descriptors into the same embedding space as ViT patches.
- Concatenate patch tokens and region tokens, and feed them through the original mPLUG cross-modal skip-connected fusion network + PrefixLM decoder.
- Train **only with cross-entropy** (XE); no RL / CIDEr optimization.

On the MS COCO Karpathy split, RAVLM improves over the XE-trained mPLUG baseline:

- BLEU-4: **45.4**  
- CIDEr: **151.7**

using the same 14M-image pre-training as mPLUG and identical fine-tuning settings.


## News

- **2025.xx.xx**: Initial release of RAVLM code and COCO captioning scripts.
- **2025.xx.xx**: Added configuration files for different region-aware variants (full ROI, geometry-only, Faster R-CNN tokens).


## Model and Variants

RAVLM is implemented as a small modification on top of the public mPLUG captioning codebase.

We consider the following visual configurations:

1. **Patch-only (baseline)**  
   - Original mPLUG captioning model with ViT patch tokens only.

2. **Full ROI (features + boxes)**  
   - Concatenate Faster R-CNN ROI appearance features and normalized box coordinates, project to hidden size, append as tokens.

3. **Geometry-only ROI**  
   - Only normalized bounding-box geometry tokens (no ROI appearance).

4. **RAVLM (Patch + Faster R-CNN features)**  
   - ViT patch tokens + projected Faster R-CNN backbone region tokens.  
   - This is our **best-performing** and recommended configuration.

### COCO Captioning Results (Karpathy test, XE only)

| Visual Configuration                  | B@4 | METEOR | CIDEr | SPICE |
|--------------------------------------|:---:|:------:|:-----:|:-----:|
| Patch-only (mPLUG baseline)          | 43.1 | 31.4 | 141.0 | 24.2 |
| Full ROI (features + boxes)          | 43.8 | 31.5 | 147.9 | 24.1 |
| Geometry-only ROI                    | 44.5 | 31.5 | 149.7 | 24.1 |
| **RAVLM (Patch + Faster R-CNN feat.)** | **45.4** | **31.8** | **151.7** | **24.4** |

All models are:

- Fine-tuned with **cross-entropy (PrefixLM)** only  
- Trained on the **MS COCO Karpathy split**  
- Initialized from the same **mPLUG base** checkpoint (14M image–text pairs)


## Requirements

- Linux + Python 3.8+  
- [PyTorch](https://pytorch.org/) **>= 1.11.0**
- CUDA-compatible GPUs (experiments in the paper use 8× V100 16GB)

Install Python dependencies:

```bash
pip install -r requirements.txt
