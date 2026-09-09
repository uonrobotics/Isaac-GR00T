# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7
from gr00t.model.gr00t_n1d7.sign_grounding import (
    GroundedTokenFusion,
    SignGroundingHead,
    box_cxcywh_to_xyxy,
    compute_sign_grounding_losses,
)
import pytest
import torch
from transformers.feature_extraction_utils import BatchFeature


def test_prepare_input_preserves_bbox_precision():
    passthrough = SimpleNamespace(prepare_input=lambda batch: BatchFeature(data=batch))
    model = SimpleNamespace(
        backbone=passthrough,
        action_head=passthrough,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
    )
    target = torch.tensor([[0.50123, 0.49917, 0.01234, 0.05678]])
    inputs = {"gt_sign_bbox_cxcywh": target, "sign_bbox_cxcywh": target, "state": target}
    for batch in Gr00tN1d7.prepare_input(model, inputs):
        for key in ("gt_sign_bbox_cxcywh", "sign_bbox_cxcywh"):
            assert batch[key].dtype == torch.float32
            assert torch.equal(batch[key], target)
        assert batch.state.dtype == torch.bfloat16


@pytest.mark.parametrize("autocast", [False, True])
def test_bbox_fp32_and_fusion_backward(autocast):
    head = SignGroundingHead(8).to(torch.bfloat16)
    fusion = GroundedTokenFusion(8, 8).to(torch.bfloat16)
    hidden = torch.randn(2, 8, dtype=torch.bfloat16, requires_grad=True)
    target = torch.tensor([[0.50123, 0.49917, 0.01234, 0.05678]]).repeat(2, 1)
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
        output = head(hidden)
        losses = compute_sign_grounding_losses(
            output["sign_bbox_cxcywh"], output["sign_status_logits"], target, torch.ones(2).long()
        )
        token = fusion(hidden, output["sign_bbox_cxcywh"], use_status_gate=False)
        for key in ("sign_bbox_cxcywh", "sign_bbox_xyxy"):
            assert output[key].dtype == torch.float32
        for key in ("sign_bbox_l1_loss", "sign_bbox_giou_loss"):
            assert losses[key].dtype == torch.float32
            assert torch.isfinite(losses[key])
        expected_l1 = (output["sign_bbox_cxcywh"] - target).abs().mean()
        torch.testing.assert_close(losses["sign_bbox_l1_loss"], expected_l1)
        loss = losses["sign_grounding_loss"] + token.float().square().mean()
    loss.backward()
    gradient = head.bbox_head[-1].weight.grad
    assert torch.isfinite(gradient).all()
    assert gradient.abs().sum() > 0


def test_out_of_frame_edges_keep_gradients():
    boxes = torch.tensor([[0.0, 1.0, 0.5, 0.5]], dtype=torch.bfloat16, requires_grad=True)
    xyxy = box_cxcywh_to_xyxy(boxes)
    assert xyxy.dtype == torch.float32
    torch.testing.assert_close(xyxy, torch.tensor([[-0.25, 0.75, 0.25, 1.25]]))
    (xyxy[0, 0] + xyxy[0, 3]).backward()
    torch.testing.assert_close(boxes.grad.float(), torch.tensor([[1.0, 1.0, -0.5, 0.5]]))
