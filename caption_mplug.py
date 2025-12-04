import argparse
import os
import ruamel.yaml as yaml
import language_evaluation
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path
import warnings


# ===== CACHE DIRECTORY SETUP =====
#workspace_dir = "C:/RoiBOX"
workspace_dir = "/ghome/Alamgir/RoiBOX"
cache_base = os.path.join(workspace_dir, ".cache")

os.environ['MPLCONFIGDIR'] = os.path.join(cache_base, 'matplotlib')
os.environ['HF_HOME'] = os.path.join(cache_base, 'huggingface')
os.environ['HF_DATASETS_CACHE'] = os.path.join(cache_base, 'huggingface/datasets')

cache_dirs = [
    os.environ['MPLCONFIGDIR'],
    os.environ['HF_HOME'],
    os.environ['HF_DATASETS_CACHE']
]

for cache_dir in cache_dirs:
    os.makedirs(cache_dir, exist_ok=True)
    print(f"? Cache directory: {cache_dir}")
# =================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from models.model_caption_mplug import MPLUG
from models.vit import interpolate_pos_embed, resize_pos_embed
from transformers import BertTokenizer

import utils
from dataset.utils import save_result
from dataset import create_dataset, create_sampler, create_loader, coco_collate_fn

# ✅ FIX: import both optimizers
from optim import create_optimizer, create_two_optimizer
from scheduler import create_scheduler   # ✅ add this

# NEW: ROI-aware collate (gracefully optional)
try:
    from dataset import (
        coco_collate_fn,
        nocaps_collate_fn,
        vqa_collate_fn,
    )
    HAS_ROI_COLLATE = True
    #print("[DEBUG] ROI collate functions loaded successfully.")
except ImportError as e:
    HAS_ROI_COLLATE = False
    warnings.warn(f"ROI collate functions not found; running without region features. ({e})")


def _unpack_train_batch(batch):
    """
    Support both legacy tuple batches and new dict batches (with ROI).
    Returns:
      image, caption, object_labels, image_ids, gold_caption, roi (dict or None)
    """
    if isinstance(batch, dict):
        image = batch["images"]
        captions = batch.get("captions", [])
        object_labels = batch.get("object_labels", [])
        img_ids = batch.get("img_ids", [])
        gold_caps = batch.get("gold_caps", [])
        roi = {
            "roi_feats": batch.get("roi_feats"),
            "roi_boxes": batch.get("roi_boxes"),
            "roi_masks": batch.get("roi_masks"),
        } if "roi_feats" in batch else None
        return image, captions, object_labels, img_ids, gold_caps, roi
    else:
        # legacy path (no ROI)
        image, caption, object_labels, image_ids, gold_caption = batch
        return image, caption, object_labels, image_ids, gold_caption, None


def _unpack_eval_batch(batch):
    """
    For test/eval dataloaders; legacy tuple vs dict.
    Returns:
      image, caption, object_labels, image_ids, gold_caption, roi
    Some eval loaders may not include object_labels or gold_caption — guard them.
    """
    if isinstance(batch, dict):
        image = batch["images"]
        captions = batch.get("captions", [])
        object_labels = batch.get("object_labels", [""] * image.size(0))
        img_ids = batch.get("img_ids", [])
        gold_caps = batch.get("gold_caps", [])
        roi = {
            "roi_feats": batch.get("roi_feats"),
            "roi_boxes": batch.get("roi_boxes"),
            "roi_masks": batch.get("roi_masks"),
        } if "roi_feats" in batch else None
        return image, captions, object_labels, img_ids, gold_caps, roi
    else:
        # legacy; may be (image, caption, object_labels, ids, gold_caps) or (image, ids)
        if len(batch) == 5:
            image, caption, object_labels, image_ids, gold_caption = batch
            return image, caption, object_labels, image_ids, gold_caption, None
        elif len(batch) == 2:
            image, image_ids = batch
            return image, [], ["" for _ in range(image.size(0))], image_ids, [], None
        else:
            raise ValueError("Unexpected eval batch format.")


def train(model, data_loader, optimizer, tokenizer, epoch, warmup_steps, device, scheduler, config, do_amp=False,
          do_two_optim=False, do_accum=False, accum_steps=1, use_roi=False):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    if do_two_optim:
        metric_logger.add_meter('lr1', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
        metric_logger.add_meter('lr2', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    else:
        metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))

    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50
    step_size = 100
    warmup_iterations = warmup_steps * step_size

    for i, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        image, caption, object_labels, image_ids, gold_caption, roi = _unpack_train_batch(batch)

        image = image.to(device, non_blocking=True)
        if config['prompt'] != "":
            caption = [config['prompt'] + each + config['eos'] for each in caption]
        else:
            caption = [each + config['eos'] for each in caption]

        question_input = [config['bos'] + " " + each for each in object_labels]
        if i == 0:
            print(question_input[:4])

        caption = tokenizer(
            caption, padding='longest', truncation=True,
            max_length=args.max_input_length, return_tensors="pt",
            clean_up_tokenization_spaces=False
        ).to(device)
        question_input = tokenizer(
            question_input, padding='longest', truncation=True,
            max_length=args.max_input_length, return_tensors="pt",
            clean_up_tokenization_spaces=False
        ).to(device)

        # Optional ROI tensors (if ROI collate is active)
        roi_kwargs = {}
        if use_roi and roi is not None and roi.get("roi_feats") is not None:
            roi_kwargs = {
                "roi_feats": roi["roi_feats"].to(device, non_blocking=True),
                "roi_boxes": roi["roi_boxes"].to(device, non_blocking=True),
                "roi_masks": roi["roi_masks"].to(device, non_blocking=True),
            }

        if epoch > 0 or not config['warm_up']:
            alpha = config['alpha']
        else:
            alpha = config['alpha'] * min(1, i / len(data_loader))

        loss = model(image, question_input, caption, train=True, **roi_kwargs)

        if accum_steps > 1:
            loss = loss / accum_steps

        if do_amp:
            from apex import amp
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        if (i + 1) % accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        metric_logger.update(loss=loss.item())
        if do_two_optim:
            metric_logger.update(lr1=optimizer.param_groups[0]["lr"])
            metric_logger.update(lr2=optimizer.param_groups[2]["lr"])
        else:
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if epoch == 0 and i % step_size == 0 and i <= warmup_iterations:
            scheduler.step(i // step_size)

        del image, question_input, caption, loss

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluation(model, data_loader, tokenizer, device, config, use_roi=False):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Generate VQA test result:'
    print_freq = 50
    result = []

    for n, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        image, caption, object_labels, image_ids, gold_caption, roi = _unpack_eval_batch(batch)
        image = image.to(device, non_blocking=True)

        caption = [each + config['eos'] for each in caption] if caption else [config['eos']] * image.size(0)
        question_input = [config['bos'] + " " + each for each in object_labels]

        caption = tokenizer(
            caption, padding='longest', truncation=True,
            max_length=args.max_input_length, return_tensors="pt",
            clean_up_tokenization_spaces=False
        ).to(device)
        question_input = tokenizer(
            question_input, padding='longest', truncation=True,
            max_length=args.max_input_length, return_tensors="pt",
            clean_up_tokenization_spaces=False
        ).to(device)

        roi_kwargs = {}
        if use_roi and roi is not None and roi.get("roi_feats") is not None:
            roi_kwargs = {
                "roi_feats": roi["roi_feats"].to(device, non_blocking=True),
                "roi_boxes": roi["roi_boxes"].to(device, non_blocking=True),
                "roi_masks": roi["roi_masks"].to(device, non_blocking=True),
            }

        topk_ids, topk_probs = model(image, question_input, caption, train=False, **roi_kwargs)

        for image_id, topk_id, topk_prob, gold_caption_list in zip(image_ids, topk_ids, topk_probs, gold_caption):
            ans = tokenizer.decode(topk_id[0]).replace("[SEP]", "").replace("[CLS]", "").replace("[PAD]", "").strip()
            result.append({"question_id": image_id, "pred_caption": ans, "gold_caption": gold_caption_list})
    return result


@torch.no_grad()
def evaluate(model, data_loader, dataset, tokenizer, device, config, use_roi=False):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Evaluation:'
    print_freq = 50
    predicts = []
    answers = []

    for n, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        image, caption, object_labels, image_ids, gold_caption, roi = _unpack_eval_batch(batch)
        image = image.to(device, non_blocking=True)

        caption = [each + config['eos'] for each in caption] if caption else [config['eos']] * image.size(0)
        question_input = [config['bos']] * len(caption)

        caption = tokenizer(
            caption, padding='longest', truncation=True,
            max_length=args.max_input_length, return_tensors="pt",
            clean_up_tokenization_spaces=False
        ).to(device)
        question_input = tokenizer(
            question_input, padding='longest', truncation=True,
            max_length=args.max_input_length, return_tensors="pt",
            clean_up_tokenization_spaces=False
        ).to(device)

        for i in range(len(gold_caption)):
            predicts.append(gold_caption[i][0])
            answers.append(gold_caption[i])

        result = cal_metric(predicts, answers)
        metric_logger.meters['Bleu_1'].update(result["Bleu_1"], n=image.size(0))

    torch.cuda.empty_cache()
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {k: "{:.4f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}


def cal_metric(result_file_or_preds, answers=None):
    """
    Backward compatible:
      - If given a JSON file path, load and score.
      - If given lists (predicts, answers), score directly.
    """
    if answers is None and isinstance(result_file_or_preds, str) and os.path.isfile(result_file_or_preds):
        result_list = json.load(open(result_file_or_preds, "r"))
        predicts = [each["pred_caption"] for each in result_list]
        answers = [each["gold_caption"] for each in result_list]
    else:
        predicts = result_file_or_preds

    evaluator = language_evaluation.CocoEvaluator(verbose=False)
    results = evaluator.run_evaluation(predicts, answers)
    print(len(predicts), results)
    return results

def main(args, config):
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['HF_DATASETS_OFFLINE'] = '1'
    os.environ['HF_HUB_OFFLINE'] = '1'

    utils.init_distributed_mode(args)

    # --- PATCH: assign device properly for torchrun/LOCAL_RANK ---
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        device = torch.device('cuda', args.gpu)
    else:
        device = torch.device(args.device)

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    start_epoch = 0
    max_epoch = config['schedular']['epochs']
    warmup_steps = config['schedular']['warmup_epochs']

    print("Creating vqa datasets")
    # IMPORTANT: pass ROI flags/paths via config so dataset factory can forward them
    config['use_roi'] = bool(args.use_roi)
    config['roi_feat_dir'] = args.roi_feat_dir
    config['roi_box_dir'] = args.roi_box_dir

    datasets = create_dataset('coco', config)
    print(f"[DEBUG] Train dataset size = {len(datasets[0])}")
    print(f"[DEBUG] Val dataset size   = {len(datasets[1])}")
    print(f"[DEBUG] Test dataset size  = {len(datasets[2])}")


    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        samplers = create_sampler(datasets, [True, False, False], num_tasks, global_rank)
    else:
        samplers = [None, None, None]

    # Always use the unified ROI-aware collate (it handles both legacy and ROI batches)
    collates = [coco_collate_fn, coco_collate_fn, coco_collate_fn]
    #print(f"[DEBUG] Using unified coco_collate_fn (ROI-aware={bool(args.use_roi)}).")
    #print(f"[DEBUG] ROI feat dir: {args.roi_feat_dir}")
    #print(f"[DEBUG] ROI box  dir: {args.roi_box_dir}")

    train_loader, val_loader, test_loader = create_loader(
        datasets, samplers,
        batch_size=[config['batch_size_train'], config['batch_size_test'], config['batch_size_test']],
        num_workers=[8, 8, 8], is_trains=[True, False, False],
        collate_fns=collates
    )

    print(f"Loading tokenizer from: {args.text_encoder}")
    #local_tokenizer_path = "bert-base-uncased"
    local_tokenizer_path = "/ghome/Alamgir/RoiBOX/bert-base-uncased"
    print(f"Using local tokenizer files from: {local_tokenizer_path}")

    required_files = ['tokenizer_config.json', 'vocab.txt', 'special_tokens_map.json']
    for file in required_files:
        file_path = os.path.join(local_tokenizer_path, file)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Missing tokenizer file: {file}")

    tokenizer = BertTokenizer.from_pretrained(local_tokenizer_path)
    print("? Tokenizer loaded successfully from local files")

    print("Creating model")

    #config['bert_config'] = 'configs/config_bert.json'
    #config['text_encoder'] = "bert-base-uncased"
    #config['text_decoder'] = "bert-base-uncased"
    
    
    config['bert_config'] = '/ghome/Alamgir/RoiBOX/configs/config_bert.json'
    config['text_encoder'] = "/ghome/Alamgir/RoiBOX/bert-base-uncased"
    config['text_decoder'] = "/ghome/Alamgir/RoiBOX/bert-base-uncased"
    

    # Disable gradient checkpointing to avoid DDP issues
    if 'gradient_checkpointing' in config:
        config['gradient_checkpointing'] = False
        print("? Gradient checkpointing disabled to avoid DDP issues")

    model = MPLUG(config=config, tokenizer=tokenizer)
    model = model.to(device)

    if not args.do_two_optim:
        arg_opt = utils.AttrDict(config['optimizer'])
        optimizer = create_optimizer(arg_opt, model)
    else:
        arg_opt = utils.AttrDict(config['optimizer'])
        optimizer = create_two_optimizer(arg_opt, model)

    arg_sche = utils.AttrDict(config['schedular'])
    lr_scheduler, _ = create_scheduler(arg_sche, optimizer)

    if args.do_amp:
        from apex import amp
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

    if args.checkpoint:
        # Add weights_only=False fallback for older torch; keep map_location
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        try:
            state_dict = checkpoint['model']
        except Exception:
            state_dict = checkpoint.get('module', checkpoint)

        if config["clip_name"] == "ViT-B-16":
            num_patches = int(config["image_res"] * config["image_res"] / (16 * 16))
        elif config["clip_name"] == "ViT-L-14":
            num_patches = int(config["image_res"] * config["image_res"] / (14 * 14))
        else:
            num_patches = int(config["image_res"] * config["image_res"] / (16 * 16))

        pos_embed = nn.Parameter(torch.zeros(num_patches + 1, 768).float())

        pos_embed = resize_pos_embed(
            state_dict['visual_encoder.visual.positional_embedding'].unsqueeze(0),
            pos_embed.unsqueeze(0)
        )
        state_dict['visual_encoder.visual.positional_embedding'] = pos_embed

        if not args.evaluate:
            for key in list(state_dict.keys()):
                if ('fusion' in key or 'bert' in key) and 'decode' not in key:
                    encoder_key = key.replace('fusion.', '').replace('bert.', '')
                    state_dict[encoder_key] = state_dict[key]
                    del state_dict[key]

        msg = model.load_state_dict(state_dict, strict=False)
        print('load checkpoint from %s' % args.checkpoint)
        print(msg)

    model_without_ddp = model
    if args.distributed:
        if args.gpu is not None:
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.gpu], output_device=args.gpu, find_unused_parameters=True
            )
        else:
            model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)

        # FIX: Add static graph setting for DDP with gradient checkpointing
        try:
            model._set_static_graph()
            print("? Static graph set for DDP to handle gradient checkpointing")
        except AttributeError:
            print("? Warning: _set_static_graph() not available in this PyTorch version")

        model_without_ddp = model.module

    print("Start training")
    start_time = time.time()

    # Optional pre-eval
    if start_epoch > 0 or args.evaluate:
        vqa_result = evaluation(model, test_loader, tokenizer, device, config, use_roi=args.use_roi)
        result_file = save_result(vqa_result, args.result_dir, 'vqa_result_epoch10')
        if utils.is_main_process():
            result = cal_metric(result_file)
        dist.barrier()

    for epoch in range(start_epoch, max_epoch):
        if epoch > 0:
            lr_scheduler.step(epoch + warmup_steps)

        if not args.evaluate:
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)
            train_stats = train(
                model, train_loader, optimizer, tokenizer, epoch,
                warmup_steps, device, lr_scheduler, config,
                do_amp=args.do_amp, do_two_optim=args.do_two_optim,
                accum_steps=args.accum_steps, use_roi=args.use_roi
            )

        if args.evaluate:
            break

        # Evaluate periodically or at the end of training
        if (epoch + 1) % config.get('eval_interval', 1) == 0 or epoch == max_epoch - 1:
            vqa_result = evaluation(model, test_loader, tokenizer, device, config, use_roi=args.use_roi)
            result_file = save_result(vqa_result, args.result_dir, f'vqa_result_epoch{epoch}')
            if utils.is_main_process():
                result = cal_metric(result_file)
                log_stats = {
                    **{f'train_{k}': v for k, v in train_stats.items()},
                    **{f'eval_{k}': v for k, v in result.items()},
                    'epoch': epoch
                }
                with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                # Save checkpoint
                if (epoch + 1) % config.get('save_interval', 1) == 0 or epoch == max_epoch - 1:
                    torch.save({
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'config': config,
                        'epoch': epoch,
                    }, os.path.join(args.output_dir, f'checkpoint_{epoch:02d}.pth'))

        dist.barrier()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/VQA.yaml')
    parser.add_argument('--checkpoint', default='')
    parser.add_argument('--output_dir', default='output/vqa')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--text_encoder', default='bert-base-uncased')
    parser.add_argument('--text_decoder', default='bert-base-uncased')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--min_length', default=1, type=int)
    parser.add_argument('--lr', default=2e-5, type=float)
    parser.add_argument('--max_length', default=10, type=int)
    parser.add_argument('--max_input_length', default=25, type=int)
    parser.add_argument('--beam_size', default=5, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--gpu', default=None, type=int, help='GPU id to use')
    parser.add_argument('--distributed', action='store_true')  # safer than type=bool

    # --- PATCH: add local_rank ---
    parser.add_argument('--local_rank', type=int, default=-1, help='local rank passed by torch.distributed/torchrun')

    parser.add_argument('--do_two_optim', action='store_true')
    parser.add_argument('--add_object', action='store_true')
    parser.add_argument('--do_amp', action='store_true')
    parser.add_argument('--no_init_decocde', action='store_true')
    parser.add_argument('--do_accum', action='store_true')
    parser.add_argument('--accum_steps', default=4, type=int)

    # ROI options (NEW)
    parser.add_argument('--use_roi', action='store_true', help='enable bottom-up region features')
    parser.add_argument('--roi_feat_dir', type=str, default='/gdata/Alamgir/mscoco/feature/up_down_100',
                        help='directory of .npz ROI features')
    parser.add_argument('--roi_box_dir', type=str, default='/gdata/Alamgir/mscoco/feature/up_down_100_box',
                        help='directory of .npy ROI boxes')
    
    parser.add_argument(
        '--roi_mode',
        type=str,
        default='full',
        choices=['none', 'feat', 'box', 'full'],
        help='ROI ablation mode: '
             'none=patch only, '
             'feat=patch+roi feats (no boxes), '
             'box=patch+roi boxes (no feats), '
             'full=patch+roi feats+boxes'
    )

    # Additional arguments for better control
    parser.add_argument('--eval_interval', type=int, default=1, help='Evaluation interval in epochs')
    parser.add_argument('--save_interval', type=int, default=1, help='Checkpoint saving interval in epochs')

    args = parser.parse_args()

    # --- PATCH: normalize gpu from LOCAL_RANK env ---
    if args.local_rank == -1 and 'LOCAL_RANK' in os.environ:
        args.local_rank = int(os.environ['LOCAL_RANK'])
    if args.gpu is None and args.local_rank is not None and args.local_rank >= 0:
        args.gpu = args.local_rank

    from ruamel.yaml import YAML
    yaml = YAML(typ='rt')
    config = yaml.load(open(args.config, 'r'))

    args.result_dir = os.path.join(args.output_dir, 'result')
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)

    config["min_length"] = args.min_length
    config["max_length"] = args.max_length
    config["add_object"] = args.add_object
    config["beam_size"] = args.beam_size
    config['text_encoder'] = args.text_encoder
    config['text_decoder'] = args.text_decoder
    config['eval_interval'] = args.eval_interval
    config['save_interval'] = args.save_interval

    # Propagate ROI settings into config for dataset factory
    config['use_roi'] = bool(args.use_roi)
    config['roi_feat_dir'] = args.roi_feat_dir
    config['roi_box_dir'] = args.roi_box_dir
    config['roi_mode'] = args.roi_mode

        # If running patch-only, skip ROI loading entirely
    if args.roi_mode == 'none':
        config['load_rois'] = False
        config['return_rois'] = False
        args.use_roi = False

    # ROI feats only
    elif args.roi_mode == 'feat':
        args.use_roi = True
        config['use_roi'] = True
        config['roi_feat_dir'] = args.roi_feat_dir
        config['roi_box_dir'] = None  # ✅ don't load boxes

    # ROI boxes only
    elif args.roi_mode == 'box':
        args.use_roi = True
        config['use_roi'] = True
        config['roi_feat_dir'] = None  # ✅ don't load feats
        config['roi_box_dir'] = args.roi_box_dir

    # ROI feats + boxes (full)
    elif args.roi_mode == 'full':
        args.use_roi = True
        config['use_roi'] = True
        config['roi_feat_dir'] = args.roi_feat_dir
        config['roi_box_dir'] = args.roi_box_dir

    print(f"[DATASET] ROI mode = {args.roi_mode}, "
          f"roi_feat_dir={config['roi_feat_dir']}, "
          f"roi_box_dir={config['roi_box_dir']}")

    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))
    main(args, config)