import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

from dataset.caption_dataset import (
    re_train_dataset,
    re_eval_dataset,
    pretrain_dataset_4m,
    coco_dataset,
    nocaps_dataset,
)

#print("[DEBUG] Imported datasets from caption_dataset successfully (with ROI support).")

from dataset.nlvr_dataset import nlvr_dataset
from dataset.ve_dataset import ve_dataset
from dataset.vqa_dataset import vqa_dataset
from dataset.grounding_dataset import build_uni_training_dataset, build_vg_dataset
from dataset.videoqa_dataset import videoqa_dataset
from dataset.video_dataset import vatex_video_caps_dataset

from dataset.randaugment import RandomAugment


def create_dataset(dataset, config, epoch=None):

    normalize = transforms.Normalize(
        (0.48145466, 0.4578275, 0.40821073),
        (0.26862954, 0.26130258, 0.27577711),
    )

    pretrain_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                config["image_res"], scale=(0.2, 1.0), interpolation=Image.BICUBIC
            ),
            transforms.RandomHorizontalFlip(),
            RandomAugment(
                2,
                7,
                isPIL=True,
                augs=[
                    "Identity",
                    "AutoContrast",
                    "Equalize",
                    "Brightness",
                    "Sharpness",
                    "ShearX",
                    "ShearY",
                    "TranslateX",
                    "TranslateY",
                    "Rotate",
                ],
            ),
            transforms.ToTensor(),
            normalize,
        ]
    )
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                config["image_res"], scale=(0.5, 1.0), interpolation=Image.BICUBIC
            ),
            transforms.RandomHorizontalFlip(),
            RandomAugment(
                2,
                7,
                isPIL=True,
                augs=[
                    "Identity",
                    "AutoContrast",
                    "Equalize",
                    "Brightness",
                    "Sharpness",
                    "ShearX",
                    "ShearY",
                    "TranslateX",
                    "TranslateY",
                    "Rotate",
                ],
            ),
            transforms.ToTensor(),
            normalize,
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.Resize(
                (config["image_res"], config["image_res"]), interpolation=Image.BICUBIC
            ),
            transforms.ToTensor(),
            normalize,
        ]
    )

    if dataset == "pretrain":
        dataset = pretrain_dataset_4m(
            config["train_file"],
            pretrain_transform,
            read_local_data=config["read_local_data"],
            image_root=config["image_root"],
            epoch=epoch,
        )
        return dataset

    elif dataset == "re":
        train_dataset = re_train_dataset(
            config["train_file"], train_transform, config["image_root"]
        )
        val_dataset = re_eval_dataset(
            config["val_file"], test_transform, config["image_root"]
        )
        test_dataset = re_eval_dataset(
            config["test_file"], test_transform, config["image_root"]
        )
        return train_dataset, val_dataset, test_dataset

    elif dataset == "vqa":
        train_dataset = vqa_dataset(
            config["train_file"],
            train_transform,
            config["vqa_root"],
            config["vg_root"],
            config["gqa_root"],
            split="train",
            read_local_data=config["read_local_data"],
            add_ocr=config["add_ocr"],
            add_object=config["add_object"],
        )
        vqa_test_dataset = vqa_dataset(
            config["test_file"],
            test_transform,
            config["vqa_root"],
            config["vg_root"],
            config["gqa_root"],
            split="test",
            answer_list=config["answer_list"],
            read_local_data=config["read_local_data"],
            add_ocr=config["add_ocr"],
            add_object=config["add_object"],
        )
        vqa_val_dataset = vqa_dataset(
            config["val_file"],
            test_transform,
            config["vqa_root"],
            config["vg_root"],
            config["gqa_root"],
            split="test",
            answer_list=config["answer_list"],
            read_local_data=config["read_local_data"],
            add_ocr=config["add_ocr"],
            add_object=config["add_object"],
        )
        return train_dataset, vqa_val_dataset, vqa_test_dataset

    elif dataset == "nocaps":
        val_dataset = nocaps_dataset(
            config["val_file"],
            test_transform,
            config["nocaps_root"],
            max_words=config["max_length"],
            read_local_data=config["read_local_data"],
            is_train=False,
            add_object=config["add_object"],
            load_rois=config.get("use_roi", False),
            return_rois=config.get("use_roi", False),
            roi_feat_dir=config.get("roi_feat_dir", None),
            roi_box_dir=config.get("roi_box_dir", None),
        )
        test_dataset = nocaps_dataset(
            config["test_file"],
            test_transform,
            config["nocaps_root"],
            max_words=config["max_length"],
            read_local_data=config["read_local_data"],
            is_train=False,
            add_object=config["add_object"],
            load_rois=config.get("use_roi", False),
            return_rois=config.get("use_roi", False),
            roi_feat_dir=config.get("roi_feat_dir", None),
            roi_box_dir=config.get("roi_box_dir", None),
        )
        return val_dataset, test_dataset

    elif dataset == "coco":
        train_dataset = coco_dataset(
            config["train_file"],
            train_transform,
            config["coco_root"],
            max_words=config["max_length"],
            read_local_data=config["read_local_data"],
            is_train=True,
            add_object=config["add_object"],
            load_rois=config.get("use_roi", False),
            return_rois=config.get("use_roi", False),
            roi_feat_dir=config.get("roi_feat_dir", None),
            roi_box_dir=config.get("roi_box_dir", None),
        )
        val_dataset = coco_dataset(
            config["val_file"],
            test_transform,
            config["coco_root"],
            max_words=config["max_length"],
            read_local_data=config["read_local_data"],
            is_train=False,
            add_object=config["add_object"],
            load_rois=config.get("use_roi", False),
            return_rois=config.get("use_roi", False),
            roi_feat_dir=config.get("roi_feat_dir", None),
            roi_box_dir=config.get("roi_box_dir", None),
        )
        test_dataset = coco_dataset(
            config["test_file"],
            test_transform,
            config["coco_root"],
            max_words=config["max_length"],
            read_local_data=config["read_local_data"],
            is_train=False,
            add_object=config["add_object"],
            load_rois=config.get("use_roi", False),
            return_rois=config.get("use_roi", False),
            roi_feat_dir=config.get("roi_feat_dir", None),
            roi_box_dir=config.get("roi_box_dir", None),
        )
        return train_dataset, val_dataset, test_dataset

    elif dataset == "nlvr":
        train_dataset = nlvr_dataset(
            config["train_file"], train_transform, config["image_root"]
        )
        val_dataset = nlvr_dataset(
            config["val_file"], test_transform, config["image_root"]
        )
        test_dataset = nlvr_dataset(
            config["test_file"], test_transform, config["image_root"]
        )
        return train_dataset, val_dataset, test_dataset

    elif dataset == "ve":
        train_dataset = ve_dataset(
            config["train_file"], train_transform, config["image_root"]
        )
        val_dataset = ve_dataset(
            config["val_file"], test_transform, config["image_root"]
        )
        test_dataset = ve_dataset(
            config["test_file"], test_transform, config["image_root"]
        )
        return train_dataset, val_dataset, test_dataset

    elif "vg_" in dataset:
        if "uni" in dataset:
            train_dataset = build_uni_training_dataset(args=config)
            val_dataset = build_vg_dataset(split="val", args=config, dataset_name="unc")
            eval_dataset = "unc"
        else:
            train_dataset = build_vg_dataset(
                split="train", args=config, dataset_name=dataset[3:]
            )
            val_dataset = build_vg_dataset(
                split="val", args=config, dataset_name=dataset[3:]
            )
            eval_dataset = dataset[3:]
        eval_split = {
            "unc": ["testA", "testB"],
            "unc+": ["testA", "testB"],
            "gref_umd": ["test"],
        }
        test_datasets = {
            split: build_vg_dataset(split=split, args=config, dataset_name=eval_dataset)
            for split in eval_split[eval_dataset]
        }
        return train_dataset, val_dataset, test_datasets

    elif dataset == "video_qa":
        train_dataset = videoqa_dataset(
            config["train_file"],
            train_transform,
            config["videoqa_root"],
            split="train",
            read_local_data=config["read_local_data"],
            max_img_size=config["image_res"],
        )
        vqa_test_dataset = videoqa_dataset(
            config["test_file"],
            test_transform,
            config["videoqa_root"],
            split="test",
            answer_list=config["answer_list"],
            read_local_data=config["read_local_data"],
            max_img_size=config["image_res"],
        )
        vqa_val_dataset = videoqa_dataset(
            config["val_file"],
            test_transform,
            config["videoqa_root"],
            split="test",
            answer_list=config["answer_list"],
            read_local_data=config["read_local_data"],
            max_img_size=config["image_res"],
        )
        return train_dataset, vqa_val_dataset, vqa_test_dataset

    elif dataset == "vatex_video_caps":
        test_dataset = vatex_video_caps_dataset(
            config["test_file"],
            config["vatex_video_caps_root"],
            max_words=config["max_length"],
            read_local_data=config["read_local_data"],
            is_train=False,
            num_frm=config["num_frm_test"],
            max_img_size=config["image_res"],
            frm_sampling_strategy="uniform",
        )
        return test_dataset


def videoqa_collate_fn(batch):
    image_list, question_list, answer_list, n = [], [], [], []
    for image, question, answer in batch:
        image_list.append(image)
        question_list.append(question)
        answer_list.append(answer)
        n.append(1)
    return torch.stack(image_list, dim=0), question_list, answer_list, n


def vqa_collate_fn(batch):
    image_list, question_list, answer_list, weight_list, n = [], [], [], [], []
    for image, question, answer, weights in batch:
        image_list.append(image)
        question_list.append(question)
        weight_list += weights
        answer_list += answer
        n.append(len(answer))
    return (
        torch.stack(image_list, dim=0),
        question_list,
        answer_list,
        torch.Tensor(weight_list),
        n,
    )


def nocaps_collate_fn(batch):
    image_list, image_id_list = [], []
    for image, image_id in batch:
        image_list.append(image)
        image_id_list.append(image_id)
    return torch.stack(image_list, dim=0), image_id_list

import torch.nn.functional as F

def coco_collate_fn(batch):
    """
    Collate function that supports:
      - Legacy tuples: (image, caption, object_label, image_id, gold_caption)
      - ROI tuples:   (image, caption, object_label, image_id, gold_caption, roi_feats, roi_boxes)
    """
    if len(batch[0]) == 5:
        # Legacy path
        image_list, caption_list, object_labels, image_id_list, gold_caption_list = [], [], [], [], []
        for image, caption, object_label, image_id, gold_caption in batch:
            image_list.append(image)
            caption_list.append(caption)
            object_labels.append(object_label)
            image_id_list.append(image_id)
            gold_caption_list.append(gold_caption)
        return (
            torch.stack(image_list, dim=0),
            caption_list,
            object_labels,
            image_id_list,
            gold_caption_list,
        )

    elif len(batch[0]) == 7:
        # ROI path
        image_list, caption_list, object_labels, image_id_list, gold_caption_list = [], [], [], [], []
        roi_feats_list, roi_boxes_list = [], []

        for image, caption, object_label, image_id, gold_caption, roi_feats, roi_boxes in batch:
            image_list.append(image)
            caption_list.append(caption)
            object_labels.append(object_label)
            image_id_list.append(image_id)
            gold_caption_list.append(gold_caption)

            # Convert NumPy ? Torch if needed
            if isinstance(roi_feats, np.ndarray):
                roi_feats = torch.from_numpy(roi_feats).float()
            if isinstance(roi_boxes, np.ndarray):
                roi_boxes = torch.from_numpy(roi_boxes).float()

            roi_feats_list.append(roi_feats)  # [Ni, 2048]
            roi_boxes_list.append(roi_boxes)  # [Ni, 4]

        # ---- Padding step ----
        max_rois = max(r.size(0) for r in roi_feats_list)
        padded_feats, padded_boxes, padded_masks = [], [], []
        for feats, boxes in zip(roi_feats_list, roi_boxes_list):
            Ni = feats.size(0)
            pad_len = max_rois - Ni

            padded_feats.append(F.pad(feats, (0, 0, 0, pad_len)))  # (Ni?Nmax, 2048)
            padded_boxes.append(F.pad(boxes, (0, 0, 0, pad_len)))  # (Ni?Nmax, 4)
            padded_masks.append(torch.cat([torch.ones(Ni), torch.zeros(pad_len)]))

        batch_roi_feats = torch.stack(padded_feats, dim=0)  # [B, Nmax, 2048]
        batch_roi_boxes = torch.stack(padded_boxes, dim=0)  # [B, Nmax, 4]
        batch_roi_masks = torch.stack(padded_masks, dim=0).long()  # [B, Nmax]

        return {
            "images": torch.stack(image_list, dim=0),
            "captions": caption_list,
            "object_labels": object_labels,
            "img_ids": image_id_list,
            "gold_caps": gold_caption_list,
            "roi_feats": batch_roi_feats,
            "roi_boxes": batch_roi_boxes,
            "roi_masks": batch_roi_masks,
        }

    else:
        raise ValueError(
            f"[coco_collate_fn] Unexpected number of items in batch: {len(batch[0])}"
        )

def create_sampler(datasets, shuffles, num_tasks, global_rank):
    samplers = []
    for dataset, shuffle in zip(datasets, shuffles):
        sampler = torch.utils.data.DistributedSampler(
            dataset, num_replicas=num_tasks, rank=global_rank, shuffle=shuffle
        )
        samplers.append(sampler)
    return samplers


def create_loader(datasets, samplers, batch_size, num_workers, is_trains, collate_fns):
    loaders = []
    for dataset, sampler, bs, n_worker, is_train, collate_fn in zip(
        datasets, samplers, batch_size, num_workers, is_trains, collate_fns
    ):
        if is_train:
            shuffle = sampler is None
            drop_last = True
        else:
            shuffle = False
            drop_last = False
        loader = DataLoader(
            dataset,
            batch_size=bs,
            num_workers=n_worker,
            pin_memory=True,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last,
        )
        loaders.append(loader)
    return loaders
