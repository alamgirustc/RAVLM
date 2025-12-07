# RAVLM: Region-Aware Vision–Language Model for Image Captioning

Official PyTorch implementation of:

> **Region-Aware Vision–Language Model for Image Captioning (RAVLM)**  
> Mohammad Alamgir Hossain\*, Md. Ibrahim Abdullah\*, ZhongFu Ye,  
> Md Bipul Hossen, Md. Atiqur Rahman, Md Shohidul Islam  
> (\*equal contribution)

RAVLM augments a strong **patch-based** vision–language backbone with **detector-based region features** for image captioning.  
It is built on top of **[mPLUG](https://github.com/X-PLUG/mPLUG)** and uses **Faster R-CNN region features** following the official **Bottom-Up and Top-Down Attention** implementation:

- **mPLUG (official repo)**: <https://github.com/X-PLUG/mPLUG>  
- **Bottom-Up Attention (official repo)**: <https://github.com/peteanderson80/bottom-up-attention>  
- **RAVLM code**: <https://github.com/alamgirustc/RAVLM>

On **MS COCO (Karpathy split)**, our best configuration (ViT patches + Faster R-CNN backbone tokens) achieves:

- **BLEU-4:** 45.4  
- **CIDEr:** 151.7  

using **only cross-entropy training** (no CIDEr-based reinforcement learning), starting from the same 14M-image pre-training as mPLUG. :contentReference[oaicite:0]{index=0}  

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

Recent vision–language models for image captioning mostly operate on **dense grids / ViT patch tokens** and often discard explicit **object-region information**. This can make it harder to describe salient entities and relationships, especially in complex scenes. :contentReference[oaicite:1]{index=1}  

**RAVLM** revisits region-aware modeling on top of a modern backbone and shows that **lightweight detector features are still complementary** to powerful patch-based encoders:

- Use **mPLUG** (CLIP-ViT + BERT + cross-modal skip-connections + PrefixLM decoder) as the base vision–language model. :contentReference[oaicite:2]{index=2}  
- Extract **region features** using a **Faster R-CNN** detector following the **bottom-up attention** pipeline. :contentReference[oaicite:3]{index=3}  
- Project region descriptors into the same space as ViT patches and **concatenate region tokens with patch tokens**.  
- Keep the original **mPLUG pre-training, fusion network, and decoder unchanged** – only the visual tokens are modified. :contentReference[oaicite:4]{index=4}  

We study several ways to inject region information and find that **Faster R-CNN backbone tokens** give the strongest gains over a patch-only baseline, while remaining efficient and easy to plug into existing pipelines. :contentReference[oaicite:5]{index=5}  

---

## Method

### Base Architecture: mPLUG Captioning

We start from the **official mPLUG captioning setup**:

- **Visual encoder:** CLIP-style ViT-B/16, producing a sequence of patch tokens  
  \[
  V = \{v_{\mathrm{cls}}, v_1, \dots, v_M\}
  \]
- **Text encoder:** BERT-base, producing token representations  
  \[
  L = \{l_{\mathrm{cls}}, l_1, \dots, l_N\}
  \]
- **Fusion:** cross-modal skip-connected Transformer network  
- **Decoder:** Transformer with **Prefix Language Modeling** (PrefixLM) objective for caption generation :contentReference[oaicite:6]{index=6}  

We **do not change** any of these components; we only augment the **visual input**.

### Region Feature Extraction

For each image, we run a **Faster R-CNN** detector (as in **Bottom-Up and Top-Down Attention**):

- Up to **K** detected regions per image  
- For each region \(i\):  
  - ROI **appearance feature** \(f_i \in \mathbb{R}^{d_{\text{det}}}\)  
  - Bounding box \((x_{\min}, y_{\min}, x_{\max}, y_{\max})\) used to derive normalized **geometry** \(g_i\) (box coordinates, width, height, area) :contentReference[oaicite:7]{index=7}  

In addition, we optionally use **Faster R-CNN backbone features** \(\tilde{f}_i\) as extra region descriptors. :contentReference[oaicite:8]{index=8}  

### Region Tokens & Visual Sequence

We project region descriptors into the ViT hidden dimension and turn them into **region tokens**:

- **Appearance-only (Patch + ROI features)**  
  \[
  r_i^{\text{feat}} = W_{\text{feat}} f_i + b_{\text{feat}}
  \]
- **Geometry-only (Patch + ROI boxes)**  
  \[
  r_i^{\text{box}} = W_{\text{box}} g_i + b_{\text{box}}
  \]
- **Full ROI (features + boxes)**  
  \[
  r_i^{\text{full}} = W_{\text{full}} [f_i ; g_i] + b_{\text{full}}
  \]
- **Faster R-CNN backbone tokens (RAVLM)**  
  \[
  r_i^{\text{rcnn}} = W_{\text{rcnn}} \tilde{f}_i + b_{\text{rcnn}}
  \] :contentReference[oaicite:9]{index=9}  

These tokens are **concatenated** with ViT patch tokens:

\[
\hat{V} = \{v_{\mathrm{cls}}, v_1, \dots, v_M, r_1, \dots, r_K\}
\]

and fed to the **unchanged mPLUG fusion network + PrefixLM decoder**. Training is done with **cross-entropy (XE)** only:

\[
\mathcal{L}_\text{CE} = - \sum_{t=1}^{T} \log p(y_t \mid y_{<t}, I)
\] :contentReference[oaicite:10]{index=10}  

### Configurations

We evaluate four visual configurations: :contentReference[oaicite:11]{index=11}  

1. **Patch-only (baseline)** — original mPLUG captioning  
2. **Full ROI (features + boxes)**  
3. **Geometry-only ROI**  
4. **RAVLM (Patch + Faster R-CNN features)** ← **default / best**

---

## News

- **2025-XX-XX** – Initial public release of RAVLM.  
- **2025-XX-XX** – Release of pretrained COCO checkpoint and generated captions.  

---

## Results

### COCO Captioning (Karpathy Test, XE Only)

All models use **MS COCO Karpathy split**, **XE-only training**, and the same mPLUG 14M-image pre-training. :contentReference[oaicite:12]{index=12}  

**Ablation over visual configurations:**

| Visual configuration                    | B@4 | METEOR | CIDEr | SPICE |
|----------------------------------------|:---:|:------:|:-----:|:-----:|
| Patch-only (mPLUG baseline)            | 43.1 | 31.4 | 141.0 | 24.2 |
| Full ROI (features + boxes)            | 43.8 | 31.5 | 147.9 | 24.1 |
| Geometry-only ROI                      | 44.5 | 31.5 | 149.7 | 24.1 |
| **RAVLM (Patch + Faster R-CNN feat.)** | **45.4** | **31.8** | **151.7** | **24.4** | :contentReference[oaicite:13]{index=13}  

RAVLM consistently improves over the strong patch-only baseline **without any RL / CIDEr optimization**.

---

## Installation

### 1. Environment

We recommend Python ≥ 3.8 and PyTorch ≥ 1.11 with CUDA.

```bash
git clone https://github.com/alamgirustc/RAVLM.git
cd RAVLM

# (Optional but recommended) create a conda env
conda create -n ravlm python=3.10 -y
conda activate ravlm

# Install Python dependencies
pip install -r requirements.txt
