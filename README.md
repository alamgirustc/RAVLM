# RAVLM: Region-Aware Vision–Language Model for Image Captioning

Official PyTorch implementation of:

> **Region-Aware Vision–Language Model for Image Captioning (RAVLM)**  
> Mohammad Alamgir Hossain\*, Md. Ibrahim Abdullah\*, ZhongFu Ye,  
> Md Bipul Hossen, Md. Atiqur Rahman, Md Shohidul Islam  
> \*Equal contribution

RAVLM augments a strong **patch-based vision–language backbone** with **detector-based region features** for image captioning. It is implemented on top of **mPLUG** and uses **Faster R-CNN region features** generated following **Bottom-Up and Top-Down Attention**:

- **mPLUG (official repo):** <https://github.com/X-PLUG/mPLUG>  
- **Bottom-Up Attention (official repo):** <https://github.com/peteanderson80/bottom-up-attention>  
- **RAVLM code:** <https://github.com/alamgirustc/RAVLM>

On **MS COCO (Karpathy split)**, our best configuration (ViT patches + Faster R-CNN backbone tokens) achieves:

- **BLEU-4:** 45.4  
- **CIDEr:** 151.7  

using **cross-entropy training only** (no CIDEr-based RL), starting from the same **14M-image pre-training** as mPLUG.

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
9. [Model Checkpoint & Outputs](#model-checkpoint--outputs)  
10. [Acknowledgements](#acknowledgements)  
11. [Citation](#citation)  
12. [Contact](#contact)

---

## Overview

Recent image captioning models often rely only on **ViT patch tokens**, ignoring explicit **object-region information**. This can make it harder to describe salient entities and their relationships in complex scenes.

**RAVLM** revisits region-aware modeling on a modern backbone and shows that **lightweight detector features** are still complementary to patch-based encoders:

- Base model: **mPLUG** (CLIP-ViT + BERT + cross-modal skip-connections + PrefixLM decoder).  
- Extra input: **Faster R-CNN** region descriptors as additional visual tokens.  
- Architecture and pre-training of mPLUG are unchanged; only the **visual tokens** are modified.

---

## Method

### Base Architecture

We start from the official **mPLUG** captioning setup:

- Visual encoder: CLIP-style ViT-B/16 → patch tokens  
- Text encoder: BERT-base  
- Fusion: cross-modal skip-connected Transformer  
- Decoder: Transformer with **Prefix Language Modeling (PrefixLM)** for caption generation  

### Region Features and Tokens

We follow **Bottom-Up Attention** to extract **Faster R-CNN** region features:

- Up to *K* regions per image  
- For each region:
  - ROI appearance feature  
  - Bounding box (for normalized geometry)  
  - Optionally, backbone region descriptors  

Region descriptors are projected to the **ViT hidden size** and used as **region tokens**, which are concatenated with patch tokens and fed into the existing fusion network and PrefixLM decoder. Training uses **cross-entropy (XE)** only.

### Visual Configurations

We evaluate four visual configurations:

1. **Patch-only (baseline)** – original mPLUG captioning (ViT patches only)  
2. **Full ROI (features + boxes)** – concatenated appearance + geometry per region  
3. **Geometry-only ROI** – normalized boxes and simple geometric attributes only  
4. **RAVLM (Patch + Faster R-CNN features)** – ViT patches + Faster R-CNN backbone tokens  
   - This is the **default and best-performing** variant.

---

## News

- **2025-XX-XX** – Initial public release of RAVLM.  
- **2025-XX-XX** – Released:
  - COCO Karpathy **RAVLM checkpoint (XE-only)**  
  - **Generated captions** on COCO Karpathy test  
  (see [Model Checkpoint & Outputs](#model-checkpoint--outputs))

---

## Results

### COCO Captioning (Karpathy Test, XE Only)

All models:

- Dataset: **MS COCO**, **Karpathy split**  
- Training: **XE-only** (no CIDEr RL)  
- Pre-training: same **14M-image mPLUG** base

| Visual configuration                    | B@4 | METEOR | CIDEr | SPICE |
|----------------------------------------|:---:|:------:|:-----:|:-----:|
| Patch-only (mPLUG baseline)            | 43.1 | 31.4 | 141.0 | 24.2 |
| Full ROI (features + boxes)            | 43.8 | 31.5 | 147.9 | 24.1 |
| Geometry-only ROI                      | 44.5 | 31.5 | 149.7 | 24.1 |
| **RAVLM (Patch + Faster R-CNN feat.)** | **45.4** | **31.8** | **151.7** | **24.4** |

RAVLM consistently improves over the patch-only mPLUG baseline without RL.

---

## Installation

### Environment

Recommended:

- Linux  
- Python ≥ 3.8  
- PyTorch ≥ 1.11 with CUDA  
- CUDA GPUs (paper experiments: **8× Tesla V100 16GB**)

```bash
git clone https://github.com/alamgirustc/RAVLM.git
cd RAVLM

# (optional)
conda create -n ravlm python=3.10 -y
conda activate ravlm

pip install -r requirements.txt
