# RAVLM: Region-Aware Vision–Language Model for Image Captioning

Official PyTorch implementation of:

> **Region-Aware Vision–Language Model for Image Captioning (RAVLM)**  
> Mohammad Alamgir Hossain\*, Md. Ibrahim Abdullah\*, ZhongFu Ye,  
> Md Bipul Hossen, Md. Atiqur Rahman, Md Shohidul Islam  
> \*Equal contribution

RAVLM augments a strong **patch-based** vision–language backbone with **detector-based region features** for image captioning.  
It is built on top of **mPLUG** and uses **Faster R-CNN region features** generated following the official **Bottom-Up and Top-Down Attention** implementation:

- **mPLUG (official repo):** <https://github.com/X-PLUG/mPLUG>  
- **Bottom-Up Attention (official repo):** <https://github.com/peteanderson80/bottom-up-attention>  
- **RAVLM code:** <https://github.com/alamgirustc/RAVLM>

On **MS COCO (Karpathy split)**, our best configuration (ViT patches + Faster R-CNN backbone tokens) achieves:

- **BLEU-4:** 45.4  
- **CIDEr:** 151.7  

using **only cross-entropy training** (no CIDEr-based reinforcement learning), starting from the same **14M-image pre-training** as mPLUG.

---

## Table of Contents

1. [Overview](#overview)  
2. [Method](#method)  
3. [News](#news)  
4. [Results](#results)  
5. [Installation](#installation)  
6. [Data & Features](#data--features)  
7. [Training](#training)  
8. [Evaluation & Inference](#evaluation--inference)  
9. [Pretrained Models & Outputs](#pretrained-models--outputs)  
10. [Acknowledgements](#acknowledgements)  
11. [Citation](#citation)  
12. [Contact](#contact)

---

## Overview

Recent vision–language models for image captioning mostly operate on **dense ViT patch tokens** and often discard explicit **object-region information**. This can make it harder to describe salient entities and relationships in complex scenes.

**RAVLM** revisits region-aware modeling on top of a modern backbone and shows that **lightweight detector features are still complementary** to powerful patch-based encoders:

- Use **mPLUG** (CLIP-ViT + BERT + cross-modal skip-connections + PrefixLM decoder) as the base vision–language model.  
- Extract **region features** using a **Faster R-CNN** detector following the **Bottom-Up and Top-Down Attention** (bottom-up attention) pipeline.  
- Project region descriptors into the same space as ViT patch embeddings and **concatenate region tokens with patch tokens**.  
- Keep the original **mPLUG pre-training, fusion network, and decoder unchanged** – only the visual tokens are modified.

We systematically study several ways to inject region information and find that **Faster R-CNN backbone tokens** yield the strongest gains over a patch-only baseline, while remaining efficient and easy to plug into existing pipelines.

---

## Method

### Base Architecture: mPLUG Captioning

RAVLM starts from the official **mPLUG** captioning setup:

- **Visual encoder:** CLIP-style ViT-B/16, producing a sequence of patch tokens  
  `V = {v_cls, v_1, ..., v_M}`  
- **Text encoder:** BERT-base, producing token representations  
  `L = {l_cls, l_1, ..., l_N}`  
- **Fusion:** cross-modal skip-connected Transformer network  
- **Decoder:** Transformer with **Prefix Language Modeling (PrefixLM)** objective for caption generation  

We **do not modify** these components; we only augment the **visual input**.

### Region Feature Extraction

For each image, we run a **Faster R-CNN** detector (as in **Bottom-Up and Top-Down Attention**):

- Up to **K** detected regions per image  
- For each region *i*:
  - ROI **appearance feature** `f_i ∈ R^{d_det}`
  - Bounding box `(x_min, y_min, x_max, y_max)` used to derive normalized **geometry** `g_i` (coordinates, width, height, area)

In some variants we also use **Faster R-CNN backbone features** `\tilde{f}_i` as additional region descriptors.

### Region Tokens & Visual Sequence

We project region descriptors into the **ViT hidden dimension** and turn them into **region tokens**:

- **Appearance-only (Patch + ROI features)**  
  `r_i^feat = W_feat f_i + b_feat`
- **Geometry-only (Patch + ROI boxes)**  
  `r_i^box = W_box g_i + b_box`
- **Full ROI (features + boxes)**  
  `r_i^full = W_full [f_i ; g_i] + b_full`
- **Faster R-CNN backbone tokens (RAVLM)**  
  `r_i^rcnn = W_rcnn \tilde{f}_i + b_rcnn`

These tokens are **concatenated** with ViT patch tokens:

`V_hat = {v_cls, v_1, ..., v_M, r_1, ..., r_K}`

The fused sequence `V_hat` is fed into the **unchanged mPLUG fusion network and PrefixLM decoder**.

Training uses **cross-entropy (XE)** only:

`L_CE = - Σ_t log p(y_t | y_<t, I)`

### Visual Configurations

We evaluate four visual configurations:

1. **Patch-only (baseline)**  
   - Original mPLUG captioning setup (ViT patches only).

2. **Full ROI (features + boxes)**  
   - For each region, concatenate ROI appearance features and normalized geometry, project, and append as region tokens.

3. **Geometry-only ROI**  
   - Use only normalized box geometry (coordinates, width, height, area), project, and append as region tokens.

4. **RAVLM (Patch + Faster R-CNN features)**  
   - ViT patch tokens + region tokens derived from Faster R-CNN backbone features (no explicit concatenation of appearance + geometry).  
   - This is the **default and best-performing** configuration.

---

## News

- **2025-XX-XX** – Initial public release of RAVLM.  
- **2025-XX-XX** – Release of COCO/Karpathy pretrained checkpoint and generated-caption files.

---

## Results

### COCO Captioning (Karpathy Test, XE Only)

All models are trained on the **COCO Karpathy split**, with **cross-entropy only** (no CIDEr RL), using the same **mPLUG 14M-image pre-training**.

**Ablation over visual configurations:**

| Visual configuration                    | B@4 | METEOR | CIDEr | SPICE |
|----------------------------------------|:---:|:------:|:-----:|:-----:|
| Patch-only (mPLUG baseline)            | 43.1 | 31.4 | 141.0 | 24.2 |
| Full ROI (features + boxes)            | 43.8 | 31.5 | 147.9 | 24.1 |
| Geometry-only ROI                      | 44.5 | 31.5 | 149.7 | 24.1 |
| **RAVLM (Patch + Faster R-CNN feat.)** | **45.4** | **31.8** | **151.7** | **24.4** |

RAVLM consistently improves over the strong patch-only baseline **without any RL / CIDEr optimization**.

---

## Installation

### 1. Environment

We recommend:

- Python ≥ 3.8  
- PyTorch ≥ 1.11 with CUDA  
- Linux + CUDA-compatible GPUs (experiments used **8× Tesla V100 16GB**)

```bash
git clone https://github.com/alamgirustc/RAVLM.git
cd RAVLM

# (Optional but recommended)
conda create -n ravlm python=3.10 -y
conda activate ravlm

# Install Python dependencies
pip install -r requirements.txt
