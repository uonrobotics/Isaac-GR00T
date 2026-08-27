# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn
import torch.nn.functional as F


def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    # The head predicts cx/cy/w/h because this avoids invalid x1>x2 boxes during
    # early training. Losses and visual metrics often expect xyxy, so keep the
    # conversion local and clamp to normalized image coordinates.
    cx, cy, w, h = boxes.unbind(dim=-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return torch.stack([x1, y1, x2, y2], dim=-1).clamp(0.0, 1.0)


def generalized_box_iou_loss(pred_xyxy: torch.Tensor, target_xyxy: torch.Tensor) -> torch.Tensor:
    # Small, dependency-free GIoU implementation for normalized boxes. This keeps
    # the grounding branch self-contained and avoids pulling torchvision ops into
    # model forward paths that may later be exported or run on deployment targets.
    pred_x1, pred_y1, pred_x2, pred_y2 = pred_xyxy.unbind(dim=-1)
    tgt_x1, tgt_y1, tgt_x2, tgt_y2 = target_xyxy.unbind(dim=-1)

    inter_x1 = torch.maximum(pred_x1, tgt_x1)
    inter_y1 = torch.maximum(pred_y1, tgt_y1)
    inter_x2 = torch.minimum(pred_x2, tgt_x2)
    inter_y2 = torch.minimum(pred_y2, tgt_y2)

    inter_w = (inter_x2 - inter_x1).clamp_min(0.0)
    inter_h = (inter_y2 - inter_y1).clamp_min(0.0)
    inter_area = inter_w * inter_h

    pred_area = (pred_x2 - pred_x1).clamp_min(0.0) * (pred_y2 - pred_y1).clamp_min(0.0)
    tgt_area = (tgt_x2 - tgt_x1).clamp_min(0.0) * (tgt_y2 - tgt_y1).clamp_min(0.0)
    union = pred_area + tgt_area - inter_area
    iou = inter_area / union.clamp_min(1e-6)

    enc_x1 = torch.minimum(pred_x1, tgt_x1)
    enc_y1 = torch.minimum(pred_y1, tgt_y1)
    enc_x2 = torch.maximum(pred_x2, tgt_x2)
    enc_y2 = torch.maximum(pred_y2, tgt_y2)
    enc_area = (enc_x2 - enc_x1).clamp_min(0.0) * (enc_y2 - enc_y1).clamp_min(0.0)

    giou = iou - (enc_area - union) / enc_area.clamp_min(1e-6)
    return (1.0 - giou).mean()


class SignGroundingHead(nn.Module):
    """Predict target-sign status and normalized bbox from one VLM query state.

    The input is the hidden state at the SIGN_QUERY position in the Cosmos/Qwen
    token sequence. That single vector has attended over the image tokens, the
    target-area instruction, and nearby sign text; the bbox/status heads turn it
    into explicit supervision signals without asking the language model to
    autoregressively generate coordinates.
    """

    def __init__(self, vlm_dim: int, num_status_classes: int = 3):
        super().__init__()
        # Centers and sizes use separate ranges. In particular, keeping sizes
        # away from zero prevents the width/height sigmoid from collapsing into
        # a nearly point-sized box with vanishing gradients.
        self.min_bbox_size = 0.01
        self.max_bbox_size = 0.5
        self.initial_bbox_size = 0.05
        self.bbox_head = nn.Sequential(
            nn.Linear(vlm_dim, vlm_dim),
            nn.GELU(),
            nn.Linear(vlm_dim, 4),
        )
        self._initialize_bbox_output_layer()
        # Status lets the model say "not found" or "ambiguous" instead of being
        # forced to hallucinate a box for every frame.
        self.status_head = nn.Sequential(
            nn.Linear(vlm_dim, vlm_dim),
            nn.GELU(),
            nn.Linear(vlm_dim, num_status_classes),
        )

    def _initialize_bbox_output_layer(self) -> None:
        """Start from a non-degenerate box prior instead of a 0.5-sized box.

        The old default initialization produced widths/heights near 0.5. The
        strongly weighted GIoU term then drove their logits deep into sigmoid's
        negative saturation region. A small positive prior matches SignNav's
        typical boxes more closely and avoids that destructive first update.
        """
        output_layer = self.bbox_head[-1]
        nn.init.zeros_(output_layer.weight)
        nn.init.zeros_(output_layer.bias)
        size_probability = (self.initial_bbox_size - self.min_bbox_size) / (
            self.max_bbox_size - self.min_bbox_size
        )
        size_logit = torch.logit(torch.tensor(size_probability)).item()
        with torch.no_grad():
            output_layer.bias[2:].fill_(size_logit)

    def forward(self, sign_hidden: torch.Tensor) -> dict[str, torch.Tensor]:
        raw_bbox = self.bbox_head(sign_hidden)
        center = torch.sigmoid(raw_bbox[..., :2])
        size = self.min_bbox_size + (self.max_bbox_size - self.min_bbox_size) * torch.sigmoid(
            raw_bbox[..., 2:]
        )
        bbox_cxcywh = torch.cat([center, size], dim=-1)
        status_logits = self.status_head(sign_hidden)
        return {
            "sign_hidden": sign_hidden,
            "sign_bbox_cxcywh": bbox_cxcywh,
            "sign_bbox_xyxy": box_cxcywh_to_xyxy(bbox_cxcywh),
            "sign_status_logits": status_logits,
        }


class GroundedTokenFusion(nn.Module):
    """Fuse sign semantics and predicted bbox into one DiT conditioning token.

    GR00T's action DiT consumes the VLM sequence as cross-attention memory. This
    module creates one extra memory token that explicitly combines:
    - sign_hidden: what the target sign means and which panel was selected
    - bbox_cxcywh: where that panel appears in the camera frame

    The resulting token has the same dimensionality as the existing VLM tokens,
    so it can be appended to ``backbone_features`` without changing DiT internals.
    """

    def __init__(self, vlm_dim: int, condition_dim: int):
        super().__init__()
        # Project both branches into the DiT cross-attention condition space. In
        # N1.7 this is normally the same as the Cosmos hidden size (2048), but the
        # arguments stay explicit for small tests and future model variants.
        self.sign_projector = nn.Linear(vlm_dim, condition_dim)
        self.bbox_projector = nn.Linear(4, condition_dim)
        # Concatenation followed by an MLP gives the model a chance to learn how
        # much spatial information versus sign semantics should influence action.
        self.fusion = nn.Sequential(
            nn.Linear(condition_dim * 2, condition_dim),
            nn.GELU(),
            nn.Linear(condition_dim, condition_dim),
            nn.LayerNorm(condition_dim),
        )

    def forward(
        self,
        sign_hidden: torch.Tensor,
        bbox_cxcywh: torch.Tensor,
        status_logits: torch.Tensor | None = None,
        gt_status: torch.Tensor | None = None,
        found_status_id: int = 1,
        use_status_gate: bool = True,
        use_gt_status_gate: bool = True,
    ) -> torch.Tensor:
        sign_feature = self.sign_projector(sign_hidden)
        bbox_feature = self.bbox_projector(bbox_cxcywh)
        grounded_feature = self.fusion(torch.cat([sign_feature, bbox_feature], dim=-1))

        if use_status_gate:
            # If the sign is absent or ambiguous, a strong grounded token can
            # mislead action prediction. During supervised training, GT status is
            # the stable gate; at inference we fall back to predicted P(found).
            if gt_status is not None and use_gt_status_gate:
                gate = (gt_status == found_status_id).to(dtype=grounded_feature.dtype).unsqueeze(-1)
            elif status_logits is not None:
                gate = torch.softmax(status_logits, dim=-1)[:, found_status_id].unsqueeze(-1)
            else:
                gate = None
            if gate is not None:
                grounded_feature = grounded_feature * gate

        # Shape becomes [B, 1, condition_dim] so the caller can append it as one
        # additional cross-attention memory token.
        return grounded_feature.unsqueeze(1)


def compute_sign_grounding_losses(
    pred_bbox_cxcywh: torch.Tensor,
    status_logits: torch.Tensor,
    gt_bbox_cxcywh: torch.Tensor | None,
    gt_status: torch.Tensor | None,
    found_status_id: int = 1,
) -> dict[str, torch.Tensor]:
    # Keep zero losses connected to the graph, so batches without found signs or
    # batches used before labels are wired in can still backpropagate cleanly.
    zero = pred_bbox_cxcywh.sum() * 0.0
    losses = {
        "sign_bbox_l1_loss": zero,
        "sign_bbox_giou_loss": zero,
        "sign_status_loss": zero,
        "sign_grounding_loss": zero,
    }

    if gt_status is not None:
        gt_status = gt_status.long()
        losses["sign_status_loss"] = F.cross_entropy(status_logits, gt_status)

    if gt_bbox_cxcywh is not None and gt_status is not None:
        # A bbox target is meaningful only when the target sign is actually
        # present. not_found/ambiguous samples train status but do not punish the
        # continuous box regressor.
        found = gt_status == found_status_id
        if found.any():
            pred_found = pred_bbox_cxcywh[found]
            gt_found = gt_bbox_cxcywh[found].to(dtype=pred_bbox_cxcywh.dtype)
            losses["sign_bbox_l1_loss"] = F.l1_loss(pred_found, gt_found)
            losses["sign_bbox_giou_loss"] = generalized_box_iou_loss(
                box_cxcywh_to_xyxy(pred_found),
                box_cxcywh_to_xyxy(gt_found),
            )

    losses["sign_grounding_loss"] = (
        losses["sign_status_loss"]
        + losses["sign_bbox_l1_loss"]
        + losses["sign_bbox_giou_loss"]
    )
    return losses
