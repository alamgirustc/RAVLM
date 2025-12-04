import json
import numpy as np
import time
import logging
import os
import random
import re

from torch.utils.data import Dataset
import torch

from PIL import Image
from PIL import ImageFile

import oss2
from io import BytesIO
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

from dataset.utils import pre_caption


def decode_int32(ann):
    ann = str(ann)
    server = str(int(ann[-1]) + 1)
    id_ = "0" * (9 - len(ann[:-1])) + ann[:-1]
    assert len(id_) == 9
    ann = server + "/" + id_
    return ann


# ---------------------------
# ROI feature utilities
# ---------------------------

def _extract_numeric_id_from_path(image_path: str):
    """
    Extract multiple candidate IDs from COCO-style filenames.
    Returns (raw_digits, trimmed_digits, stem).
    """
    base = os.path.basename(image_path)
    stem = os.path.splitext(base)[0]

    # Grab the last group of digits in the name
    m = re.findall(r'(\d+)', stem)
    if not m:
        return None, None, stem

    raw = m[-1]              # e.g., "000000123456"
    trimmed = str(int(raw))  # e.g., "123456"
    return raw, trimmed, stem


def _find_roi_files(image_path: str, feat_dir: str, box_dir: str):
    """
    Find (roi_feat_path, roi_box_path) by trying raw digits, trimmed digits,
    zero-padded versions, and full stem.
    """
    if not feat_dir or not box_dir:
        return None, None

    base = os.path.basename(image_path)
    stem = os.path.splitext(base)[0]

    raw_digits, trimmed, _ = _extract_numeric_id_from_path(image_path)

    candidates = []

    if raw_digits:
        candidates.append(raw_digits)  # raw with leading zeros
        if trimmed and trimmed != raw_digits:
            candidates.append(trimmed)  # trimmed version
        for width in (12, 11, 10, 9):  # try common paddings
            candidates.append(trimmed.zfill(width))

    candidates.append(stem)  # fallback

    uniq_candidates = []
    seen = set()
    for c in candidates:
        if c not in seen:
            uniq_candidates.append(c)
            seen.add(c)

    for cand in uniq_candidates:
        f = os.path.join(feat_dir, f"{cand}.npz")
        b = os.path.join(box_dir, f"{cand}.npy")
        if os.path.isfile(f) and os.path.isfile(b):
            return f, b

    return None, None

def _safe_load_rois(image_path: str, feat_dir: str, box_dir: str):
    """
    Load (N,2048) roi_feats and (N,4) roi_boxes if present.
    Returns empty arrays if missing or error.
    """
    if not feat_dir or not box_dir:
        return np.zeros((0, 2048), dtype=np.float32), np.zeros((0, 4), dtype=np.float32)

    feat_path, box_path = _find_roi_files(image_path, feat_dir, box_dir)
    if feat_path is None or box_path is None:
        logging.warning(
            f"[roi] No matching ROI pair for image='{image_path}' "
            f"(feat_dir='{feat_dir}', box_dir='{box_dir}')"
        )
        return np.zeros((0, 2048), dtype=np.float32), np.zeros((0, 4), dtype=np.float32)

    try:
        npz = np.load(feat_path)
        if 'feat' in npz:
            roi_feats = npz['feat'].astype(np.float32, copy=False)
        else:
            roi_feats = npz[list(npz.keys())[0]].astype(np.float32, copy=False)

        roi_boxes = np.load(box_path).astype(np.float32, copy=False)

        # ✅ Debug print (optional)
        # print(f"[DATASET DEBUG] Loaded roi_feats {roi_feats.shape}, roi_boxes {roi_boxes.shape} from {feat_path}, {box_path}")

        if roi_feats.ndim != 2 or roi_feats.shape[1] != 2048:
            logging.warning(f"[roi] Unexpected roi_feats shape {roi_feats.shape} for {feat_path}")
        if roi_boxes.ndim != 2 or roi_boxes.shape[1] != 4:
            logging.warning(f"[roi] Unexpected roi_boxes shape {roi_boxes.shape} for {box_path}")
        if roi_feats.shape[0] != roi_boxes.shape[0]:
            n = min(roi_feats.shape[0], roi_boxes.shape[0])
            logging.warning(
                f"[roi] Mismatch feats({roi_feats.shape[0]}) vs boxes({roi_boxes.shape[0]}), truncating to {n}"
            )
            roi_feats = roi_feats[:n]
            roi_boxes = roi_boxes[:n]

        return roi_feats, roi_boxes

    except Exception as e:
        logging.warning(f"[roi] Failed loading ROI features for {image_path}: {e}")
        return np.zeros((0, 2048), dtype=np.float32), np.zeros((0, 4), dtype=np.float32)

# ---------------------------
# Datasets
# ---------------------------

class re_train_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root, max_words=30):
        self.ann = []
        for f in ann_file:
            self.ann += json.load(open(f, 'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words
        self.img_ids = {}

        n = 0
        for ann in self.ann:
            img_id = ann['image_id']
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        ann = self.ann[index]
        image_path = os.path.join(self.image_root, ann['image'])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        caption = pre_caption(ann['caption'], self.max_words)
        return image, caption, self.img_ids[ann['image_id']]


class re_eval_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root, max_words=30):
        self.ann = json.load(open(ann_file, 'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words

        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}

        txt_id = 0
        for img_id, ann in enumerate(self.ann):
            self.image.append(ann['image'])
            self.img2txt[img_id] = []
            for i, caption in enumerate(ann['caption']):
                self.text.append(pre_caption(caption, self.max_words))
                self.img2txt[img_id].append(txt_id)
                self.txt2img[txt_id] = img_id
                txt_id += 1

    def __len__(self):
        return len(self.image)

    def __getitem__(self, index):
        image_path = os.path.join(self.image_root, self.ann[index]['image'])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        return image, index


class nocaps_dataset(Dataset):
    def __init__(self, ann_file, transform, root_path, max_words=30,
                 read_local_data=True, is_train=True, add_object=False,
                 load_rois=False, roi_feat_dir=None, roi_box_dir=None, return_rois=False):
        self.ann = []
        for f in ann_file:
            self.ann += json.load(open(f, 'r'))
        self.transform = transform
        self.max_words = max_words
        self.read_local_data = read_local_data
        self.root_path = root_path
        self.add_object = add_object

        self.load_rois = load_rois
        self.roi_feat_dir = roi_feat_dir
        self.roi_box_dir = roi_box_dir
        self.return_rois = return_rois

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        ann = self.ann[index]
        image_id = ann['img_id']

        if self.read_local_data:
            image_path = os.path.join(self.root_path, ann['image'])
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image)

            # ✅ Use only basename for ROI matching (important!)
            img_basename = os.path.basename(ann['image'])
            roi_feats, roi_boxes = _safe_load_rois(
                img_basename, self.roi_feat_dir, self.roi_box_dir
            ) if self.load_rois else (
                np.zeros((0,2048), dtype=np.float32),
                np.zeros((0,4), dtype=np.float32)
            )

        else:
            while not self.bucket.object_exists("mm_feature/" + ann['image']):
                index = 0 if index == (len(self) - 1) else index + 1
                ann = self.ann[index]
            while True:
                try:
                    image = self.bucket.get_object("mm_feature/" + ann['image'])
                    image = BytesIO(image.read())
                    image = Image.open(image).convert('RGB')
                    image = self.transform(image)
                    roi_feats, roi_boxes = (
                        np.zeros((0,2048), dtype=np.float32),
                        np.zeros((0,4), dtype=np.float32)
                    )
                except:
                    index = 0 if index == (len(self) - 1) else index + 1
                    ann = self.ann[index]
                    continue
                break

        if self.load_rois and self.return_rois:
            return image, image_id, roi_feats, roi_boxes
        else:
            return image, image_id

class coco_dataset(Dataset):
    def __init__(self, ann_file, transform, root_path, max_words=30,
                 read_local_data=True, is_train=True, add_object=False,
                 load_rois=False, roi_feat_dir=None, roi_box_dir=None, return_rois=False):
        self.ann = []
        for f in ann_file:
            self.ann += json.load(open(f, 'r'))
        self.transform = transform
        self.max_words = max_words
        self.read_local_data = read_local_data
        self.root_path = root_path
        self.ann_new = []
        self.add_object = add_object

        self.load_rois = load_rois
        self.roi_feat_dir = roi_feat_dir
        self.roi_box_dir = roi_box_dir
        self.return_rois = return_rois

        for each in self.ann:
            filename = each["filename"]
            sentences = each["sentences"]
            filepath = each["filepath"]
            if filepath == "val2014":
                file_root = "val2014_img"
            elif filepath == "train2014":
                file_root = "train2014_img"
            else:
                file_root = filepath
            image_path = os.path.join(file_root, filename)
            gold_caption = [sent["raw"].lower() for sent in sentences]
            if self.add_object:
                object_list = each["object_label"].split("&&")
                new_object_list = list(set(object_list))
                new_object_list.sort(key=object_list.index)
                object_label = " ".join(new_object_list)
            else:
                object_label = ""
            if is_train:
                for sent in sentences:
                    caption = sent["raw"].lower()
                    self.ann_new.append({
                        "image": image_path,
                        "caption": caption,
                        "gold_caption": gold_caption,
                        "object_label": object_label
                    })
            else:
                self.ann_new.append({
                    "image": image_path,
                    "caption": sentences[0]["raw"].lower(),
                    "gold_caption": gold_caption,
                    "object_label": object_label
                })
        self.ann = self.ann_new
        del self.ann_new

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        ann = self.ann[index]
        caption = ann['caption']
        image_id = ann['image'].split("/")[-1]
        object_label = ann['object_label']

        if self.read_local_data:
            image_path = os.path.join(self.root_path, ann['image'])
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image)

            # ✅ Use only basename for ROI matching
            img_basename = os.path.basename(ann['image'])
            roi_feats, roi_boxes = _safe_load_rois(
                img_basename, self.roi_feat_dir, self.roi_box_dir
            ) if self.load_rois else (
                np.zeros((0,2048), dtype=np.float32),
                np.zeros((0,4), dtype=np.float32)
            )

        else:
            while not self.bucket.object_exists("mm_feature/" + ann['image']):
                index = 0 if index == (len(self) - 1) else index + 1
                ann = self.ann[index]
            while True:
                try:
                    image = self.bucket.get_object("mm_feature/" + ann['image'])
                    image = BytesIO(image.read())
                    image = Image.open(image).convert('RGB')
                    image = self.transform(image)
                    roi_feats, roi_boxes = (
                        np.zeros((0,2048),dtype=np.float32),
                        np.zeros((0,4),dtype=np.float32)
                    )
                except:
                    index = 0 if index == (len(self) - 1) else index + 1
                    ann = self.ann[index]
                    continue
                break

        if self.load_rois and self.return_rois:
            return image, caption, object_label, image_id, ann["gold_caption"], roi_feats, roi_boxes
        else:
            return image, caption, object_label, image_id, ann["gold_caption"]


class pretrain_dataset_4m(Dataset):
    def __init__(self, ann_file, transform, max_words=30, read_local_data=True, image_root="", epoch=None):
        self.ann = []
        for f in ann_file:
            self.ann += json.load(open(f, 'r'))
        self.transform = transform
        self.max_words = max_words
        self.read_local_data = read_local_data
        self.image_root = image_root
        if not self.read_local_data:
            bucket_name = "xxxxx"
            auth = oss2.Auth("xxxxx", "xxxxxx")
            self.bucket = oss2.Bucket(auth, "xxxxx", bucket_name)

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        ann = self.ann[index]
        if type(ann['caption']) == list:
            caption = pre_caption(random.choice(ann['caption']), self.max_words)
        else:
            caption = pre_caption(ann['caption'], self.max_words)

        if self.read_local_data:
            image = Image.open(os.path.join(self.image_root, ann['image'])).convert('RGB')
            image = self.transform(image)
        else:
            while True:
                try:
                    image = self.bucket.get_object("mm_feature/" + ann['image'])
                    image = BytesIO(image.read())
                    image = Image.open(image).convert('RGB')
                    image = self.transform(image)
                except:
                    time.sleep(0.1)
                    index = 0 if index == (len(self) - 1) else index + 1
                    ann = self.ann[index]
                    continue
                break

        return image, caption


# ---------------------------
# Collate Functions for ROI
# ---------------------------

def collate_fn_with_rois(batch):
    """
    For datasets returning (image, caption, object_label, image_id, gold_caps, roi_feats, roi_boxes).
    Pads RoIs.
    """
    images, captions, obj_labels, img_ids, gold_caps, roi_feats_list, roi_boxes_list = zip(*batch)
    images = torch.stack(images, dim=0)

    max_rois = max(f.shape[0] for f in roi_feats_list)
    B = len(batch)
    feat_dim = roi_feats_list[0].shape[1] if max_rois > 0 else 2048

    roi_feats = torch.zeros(B, max_rois, feat_dim, dtype=torch.float32)
    roi_boxes = torch.zeros(B, max_rois, 4, dtype=torch.float32)
    roi_masks = torch.zeros(B, max_rois, dtype=torch.long)

    for i in range(B):
        n = roi_feats_list[i].shape[0]
        if n > 0:
            roi_feats[i, :n] = torch.from_numpy(roi_feats_list[i])
            roi_boxes[i, :n] = torch.from_numpy(roi_boxes_list[i])
            roi_masks[i, :n] = 1

    # ✅ Debug
    #print(f"[COLLATE DEBUG] roi_feats batch {roi_feats.shape}, roi_boxes batch {roi_boxes.shape}")

    return {
        "images": images,
        "captions": list(captions),
        "object_labels": list(obj_labels),
        "img_ids": list(img_ids),
        "gold_caps": list(gold_caps),
        "roi_feats": roi_feats,
        "roi_boxes": roi_boxes,
        "roi_masks": roi_masks,
    }


def collate_fn_image_only_with_rois(batch):
    """
    For datasets returning (image, image_id, roi_feats, roi_boxes).
    Pads RoIs.
    """
    images, img_ids, roi_feats_list, roi_boxes_list = zip(*batch)
    images = torch.stack(images, dim=0)

    max_rois = max(f.shape[0] for f in roi_feats_list)
    B = len(batch)
    feat_dim = roi_feats_list[0].shape[1] if max_rois > 0 else 2048

    roi_feats = torch.zeros(B, max_rois, feat_dim, dtype=torch.float32)
    roi_boxes = torch.zeros(B, max_rois, 4, dtype=torch.float32)
    roi_masks = torch.zeros(B, max_rois, dtype=torch.long)

    for i in range(B):
        n = roi_feats_list[i].shape[0]
        if n > 0:
            roi_feats[i, :n] = torch.from_numpy(roi_feats_list[i])
            roi_boxes[i, :n] = torch.from_numpy(roi_boxes_list[i])
            roi_masks[i, :n] = 1

    # ✅ Debug
    #print(f"[COLLATE DEBUG] (image-only) roi_feats batch {roi_feats.shape}, roi_boxes batch {roi_boxes.shape}")

    return {
        "images": images,
        "img_ids": list(img_ids),
        "roi_feats": roi_feats,
        "roi_boxes": roi_boxes,
        "roi_masks": roi_masks,
    }
