from functools import partial
from models.vit import VisionTransformer
from models.modeling_mplug import BertConfig, BertModel, BertPrefixModel, FusionModel
from models.visual_transformers import initialize_clip
from models.predictor import TextGenerator

import torch
from torch import nn
import torch.nn.functional as F

import numpy as np


class MPLUG(nn.Module):
    def __init__(
        self,
        tokenizer=None,
        config=None,
    ):
        super().__init__()

        self.tokenizer = tokenizer
        self.config = config if config is not None else {}
        self._debug_printed = False  # only print ROI debug once

        self.module_setting(self.config)

        # visual backbone (CLIP-like)
        self.visual_encoder, _ = initialize_clip(self.config)

        # text encoders/decoders
        self.text_encoder = BertModel.from_pretrained(
            self.config["text_encoder"], config=self.config_encoder, add_pooling_layer=False
        )
        self.fusion_encoder = FusionModel.from_pretrained(
            self.config["text_encoder"], config=self.config_fusion, add_pooling_layer=False
        )
        self.text_decoder = BertPrefixModel.from_pretrained(
            self.config["text_decoder"], config=self.config_decoder
        )
        self.beam_generator = TextGenerator(self.config, self.text_decoder)

        # ---------- ROI projection & geometry encoding ----------
        # Hidden size used by the BERT encoder/decoder
        self.hidden_size = self.config_encoder.hidden_size

        # Project bottom-up region features (2048 -> hidden)
        self.reg_proj = nn.Linear(2048, self.hidden_size)

        # Encode box geometry (x1/W, y1/H, x2/W, y2/H, area, aspect) -> hidden
        self.box_mlp = nn.Sequential(
            nn.Linear(6, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size)   # ✅ add normalization for stability
        )

        # Slight regularization for region stream
        self.roi_dropout = nn.Dropout(self.config_encoder.hidden_dropout_prob)

        # === ROI MODE DEBUG BANNER ===
        roi_mode = self.config.get("roi_mode", "full")
        use_roi = self.config.get("use_roi", True)
        print(f"[MODEL] ROI mode = {roi_mode} | use_roi={use_roi} | "
              f"geom_used={'yes' if roi_mode=='full' else 'no'}")

    def forward(
        self,
        image,
        question,            # kept for backward-compat (unused here)
        answer=None,
        train=True,
        out_size=5,
        scst=False,
        roi_feats=None,      # (B, N, 2048) or None
        roi_boxes=None,      # (B, N, 4)    or None, xyxy in pixels
        roi_masks=None,      # (B, N)       or None, 1=valid
    ):
        """
        When roi_feats/roi_boxes are provided, we augment visual tokens:
          image_embeds = [patch_tokens ; region_tokens]
          image_atts   = [patch_attn   ; region_attn]
        Fallbacks to patch-only path when ROI is absent.
        """
        if scst:
            # (kept as-is; if you need SCST with ROI, mirror the logic from below)
            return self.beam_search(image, question, answer, train=True, out_size=out_size)

        # ----- Encode image into patch tokens -----
        image = image.to(dtype=next(self.parameters()).dtype)
        image_embeds = self.visual_encoder.visual(
            image, skip_last_layer=True, use_checkpoint=self.use_checkpoint
        )  # [B, P, vision_width]

        # Map to hidden size if needed
        if self.large:
            image_embeds = self.dropout(self.visn_layer_norm(self.visn_fc(image_embeds)))  # [B, P, H]
        # Attention mask for patches
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image.device)  # [B, P]

        # ----- Optional: fuse region features (ablation-aware) -----
        roi_mode = self.config.get("roi_mode", "full")  # 'none' | 'feat' | 'box' | 'full'

        # ✅ Print once at the beginning for clarity
        if not hasattr(self, "_roi_mode_printed"):
            print(f"[MPLUG] Running in ROI mode = '{roi_mode}'")
            if roi_mode == "none":
                print(" → Using PATCH tokens only (original mPLUG baseline).")
            elif roi_mode == "feat":
                print(" → Using PATCH tokens + ROI features (no boxes).")
            elif roi_mode == "box":
                print(" → Using PATCH tokens + ROI boxes (no feats).")
            elif roi_mode == "full":
                print(" → Using PATCH tokens + ROI features + ROI boxes (full setup).")
            self._roi_mode_printed = True

        use_roi_stream = (
            roi_mode != "none" and (
                (roi_mode in ["feat", "full"] and roi_feats is not None and roi_feats.numel() > 0) or
                (roi_mode in ["box", "full"] and roi_boxes is not None and roi_boxes.numel() > 0)
            )
        )
        if use_roi_stream:
            roi_embeds = None

            # Case 1: ROI feats (feat or full)
            if roi_mode in ["feat", "full"] and roi_feats is not None and roi_feats.numel() > 0:
                roi_feats = F.normalize(roi_feats, dim=-1)   # ✅ standard step
                roi_embeds = self.reg_proj(roi_feats)  # [B, N, H]

            # Case 2: ROI boxes (box or full)
            if roi_mode in ["box", "full"] and roi_boxes is not None and roi_boxes.numel() > 0:
                # --- FIX: detect true image H,W instead of assuming square image_res ---
                if "orig_size" in self.config:
                    H, W = self.config["orig_size"]
                else:
                    # infer from max coords in roi_boxes
                    max_xy = roi_boxes.max(dim=1).values  # [B, 4]
                    W = max_xy[:, [0, 2]].max().item()
                    H = max_xy[:, [1, 3]].max().item()
                    H, W = int(H), int(W)
                    if H <= 0 or W <= 0:
                        # fallback to image_res if something is wrong
                        H = W = int(self.config.get("image_res", 224))

                x1, y1, x2, y2 = roi_boxes.unbind(-1)  # [B, N]
                w = (x2 - x1).clamp(min=1e-6)
                h = (y2 - y1).clamp(min=1e-6)
                area = (w * h) / float(H * W + 1e-6)
                aspect = torch.log((w / h).clamp(min=1e-6))   # ✅ safer log aspect ratio
                geom = torch.stack(
                    [x1 / W, y1 / H, x2 / W, y2 / H, area, aspect], dim=-1
                )  # [B, N, 6]
                box_embeds = self.box_mlp(geom)  # [B, N, H]

                if roi_embeds is None:
                    roi_embeds = box_embeds
                else:
                    roi_embeds = roi_embeds + box_embeds

            roi_embeds = self.roi_dropout(roi_embeds)

            # ROI attention mask
            if roi_masks is None or roi_masks.numel() == 0:
                roi_masks = torch.ones(
                    roi_embeds.size()[:-1], dtype=torch.long, device=image.device
                )

            # Concatenate patch tokens + region tokens
            image_embeds = torch.cat([image_embeds, roi_embeds], dim=1)  # [B, P+N, H]
            image_atts = torch.cat([image_atts, roi_masks.long()], dim=1)  # [B, P+N]

            # ✅ debug block moved inside
            with torch.no_grad():
                # First 2 ROI tokens, first 5 dims
                roi_vals = roi_embeds[0, :2, :5].detach().cpu().numpy()
                #print(f"[FUSION DEBUG] roi_embeds sample:\n{roi_vals}")

                # First 2 patch tokens, first 5 dims
                patch_vals = image_embeds[0, :2, :5].detach().cpu().numpy()
                #print(f"[FUSION DEBUG] patch_embeds sample:\n{patch_vals}")

                # First 20 entries of the attention mask
                att_vals = image_atts[0, :20].detach().cpu().numpy()
                #print(f"[FUSION DEBUG] image_atts sample (first 20): {att_vals}")

            # 🔎 Debug fused embeddings
            #print(f"[FUSION CHECK] batch={image_embeds.size(0)}, "
            #      f"patch_tokens={image_embeds.size(1)-roi_embeds.size(1)}, "
            #      f"roi_tokens={roi_embeds.size(1)}, "
            #      f"fused_tokens={image_embeds.size(1)}, "
            #      f"roi_mask_sum={roi_masks.sum().item()}")

            # ✅ Safety check
            assert image_embeds.size(1) > roi_embeds.size(1), "[ERROR] ROI tokens not fused!"
            if not self._debug_printed:
                #print(f"[CHECK] image_embeds: {image_embeds.shape}, "
                #      f"roi_embeds: {roi_embeds.shape}, "
                #      f"roi_boxes (first 2): {roi_boxes[0, :2] if roi_boxes is not None else None}")
                self._debug_printed = True

            # 🔎 Debug fused embedding shapes (only print once)
            if not self._debug_printed:
                P = image_embeds.size(1) - roi_embeds.size(1)
                N = roi_embeds.size(1)
                #print(f"[DEBUG] Patch tokens: {image_embeds[:, :-roi_embeds.size(1)].shape}")
                #print(f"[DEBUG] ROI tokens: {roi_embeds.shape}")
                #print(f"[DEBUG] Fused image_embeds: {image_embeds.shape}")
                self._debug_printed = True

        # ----- Training vs Inference -----
        if train:
            # Prepare labels (ignore pad)
            answer_targets = answer.input_ids.masked_fill(
                answer.input_ids == self.tokenizer.pad_token_id, -100
            )

            # ✅ Print debug info only once
            if not hasattr(self, "_xattn_debug_printed"):
                #print(
                #    f"[XATTN DEBUG] Decoder sees {image_embeds.shape[1]} encoder tokens "
                #    f"(patch+roi), attn_mask sum={image_atts.sum().item()}"
                #)
                self._xattn_debug_printed = True

            #print(f"[XATTN DEBUG] Decoder sees {image_embeds.shape[1]} encoder tokens "
            #f"(patch+roi), attn_mask sum={image_atts.sum().item()}")

            # Decoder with cross-attention over image tokens
            answer_output = self.text_decoder(
                answer.input_ids,
                attention_mask=answer.attention_mask,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                labels=answer_targets,
                return_dict=True,
                reduction="none",
            )
            loss = answer_output.loss
            return loss

        else:
            # Greedy/beam decoding
            topk_ids, topk_probs = self.generation(image_embeds, image_atts)
            return topk_ids, topk_probs

    def module_setting(self, config):
        self.config_encoder = BertConfig.from_json_file(config["bert_config"])
        self.config_encoder.num_hidden_layers = self.config_encoder.text_encoder_layers

        self.config_fusion = BertConfig.from_json_file(config["bert_config"])

        self.config_decoder = BertConfig.from_json_file(config["bert_config"])
        self.config_decoder.add_cross_attention = True
        self.config_decoder.num_hidden_layers = self.config_decoder.text_decode_layers

        self.large = False
        if self.config_encoder.hidden_size != config["vision_width"]:
            self.visn_fc = nn.Linear(config["vision_width"], self.config_encoder.hidden_size)
            self.visn_layer_norm = nn.LayerNorm(self.config_encoder.hidden_size, eps=1e-12)
            self.dropout = nn.Dropout(self.config_encoder.hidden_dropout_prob)
            self.large = True

        self.use_checkpoint = config["use_checkpoint"] if "use_checkpoint" in config else True
        print("use_checkpoint: ", self.use_checkpoint)

    def beam_search(self, image, question, answer=None, train=True, out_size=5):
        image_embeds = self.visual_encoder.visual(image, skip_last_layer=True)
        if self.large:
            image_embeds = self.dropout(self.visn_layer_norm(self.visn_fc(image_embeds)))
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
        topk_ids, topk_probs = self.generation(image_embeds, image_atts, out_size=out_size)
        return topk_ids, topk_probs

    @torch.no_grad()
    def copy_params(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient

    @torch.no_grad()
    def _momentum_update(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1.0 - self.momentum)

    def generation(self, question_states, question_atts, out_size=1):
        encoder_inputs = [question_states, question_atts]
        topk_ids, topk_probs = self.beam_generator.translate_batch_scst(
            encoder_inputs, out_size=out_size
        )
        return topk_ids, topk_probs

    def rank_answer(self, question_states, question_atts, answer_ids, answer_atts, k):
        num_ques = question_states.size(0)
        start_ids = answer_ids[0, 0].repeat(num_ques, 1)  # bos token

        start_output = self.text_decoder(
            start_ids,
            encoder_hidden_states=question_states,
            encoder_attention_mask=question_atts,
            return_dict=True,
            reduction="none",
        )
        logits = start_output.logits[:, 0, :]  # first token's logit

        # topk_probs: top-k probability
        # topk_ids: [num_question, k]
        answer_first_token = answer_ids[:, 1]
        prob_first_token = F.softmax(logits, dim=1).index_select(dim=1, index=answer_first_token)
        topk_probs, topk_ids = prob_first_token.topk(k, dim=1)

        # answer input: [num_question*k, answer_len]
        input_ids = []
        input_atts = []
        for b, topk_id in enumerate(topk_ids):
            input_ids.append(answer_ids.index_select(dim=0, index=topk_id))
            input_atts.append(answer_atts.index_select(dim=0, index=topk_id))
        input_ids = torch.cat(input_ids, dim=0)
        input_atts = torch.cat(input_atts, dim=0)

        targets_ids = input_ids.masked_fill(input_ids == self.tokenizer.pad_token_id, -100)

        # repeat encoder's output for top-k answers
        question_states = tile(question_states, 0, k)
        question_atts = tile(question_atts, 0, k)

        output = self.text_decoder(
            input_ids,
            attention_mask=input_atts,
            encoder_hidden_states=question_states,
            encoder_attention_mask=question_atts,
            labels=targets_ids,
            return_dict=True,
            reduction="none",
        )

        answer_loss = output.loss
        answer_loss = answer_loss.view(input_ids.size(0), -1)

        # topk_prob: first token probability
        topk_probs = topk_probs.view(-1, 1)
        log_probs = torch.cat([topk_probs.log(), -answer_loss], dim=1)

        # re-calculate log probabilities for the answer sequences using chain rule
        log_probs_sum = log_probs.sum(1)
        log_probs_sum = log_probs_sum.view(num_ques, k)

        topk_probs = F.softmax(log_probs_sum, dim=-1)
        # get top-k after re-ranking
        topk_probs, rerank_id = topk_probs.topk(k, dim=1)
        topk_ids = torch.gather(topk_ids, 1, rerank_id)

        return topk_ids, topk_probs


def tile(x, dim, n_tile):
    init_dim = x.size(dim)
    repeat_idx = [1] * x.dim()
    repeat_idx[dim] = n_tile
    x = x.repeat(*(repeat_idx))
    order_index = torch.LongTensor(
        np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])
    )
    return torch.index_select(x, dim, order_index.to(x.device))
