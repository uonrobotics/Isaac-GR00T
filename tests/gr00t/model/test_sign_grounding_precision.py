# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7, Gr00tN1d7ActionHead
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


@pytest.mark.parametrize(
    ("mode", "expected_bbox_key", "uses_gt_status"),
    [
        ("pred", "sign_bbox_cxcywh", False),
        ("gt_bbox", "gt_sign_bbox_cxcywh", False),
        ("gt_bbox_status", "gt_sign_bbox_cxcywh", True),
    ],
)
def test_action_grounded_token_selects_requested_bbox(mode, expected_bbox_key, uses_gt_status):
    class CaptureFusion:
        def __call__(self, **kwargs):
            self.kwargs = kwargs
            return torch.zeros(1, 1, 4)

    fusion = CaptureFusion()
    model = SimpleNamespace(
        grounded_token_fusion=fusion,
        config=SimpleNamespace(
            sign_found_status_id=1,
            sign_use_status_gate=True,
            sign_use_gt_status_gate=True,
        ),
    )
    grounding = BatchFeature(
        data={
            "sign_hidden": torch.ones(1, 4),
            "sign_bbox_cxcywh": torch.tensor([[0.1, 0.2, 0.3, 0.4]]),
            "gt_sign_bbox_cxcywh": torch.tensor([[0.5, 0.6, 0.2, 0.1]]),
            "sign_status_logits": torch.tensor([[0.0, 1.0, 0.0]]),
            "gt_sign_status": torch.ones(1, dtype=torch.long),
        }
    )
    backbone = BatchFeature(
        data={
            "backbone_features": torch.zeros(1, 2, 4),
            "backbone_attention_mask": torch.ones(1, 2),
        }
    )

    Gr00tN1d7._append_grounded_token(model, backbone, grounding, conditioning_mode=mode)

    torch.testing.assert_close(fusion.kwargs["bbox_cxcywh"], grounding[expected_bbox_key])
    if uses_gt_status:
        torch.testing.assert_close(fusion.kwargs["gt_status"], grounding.gt_sign_status)
    else:
        assert fusion.kwargs["gt_status"] is None


@pytest.mark.parametrize(
    ("ablation_mode", "zero_sign", "zero_bbox"),
    [
        ("normal", False, False),
        ("hidden_only", False, True),
        ("bbox_only", True, False),
    ],
)
def test_action_grounded_token_ablation_selects_projection_branch(
    ablation_mode, zero_sign, zero_bbox
):
    class CaptureFusion:
        def __call__(self, **kwargs):
            self.kwargs = kwargs
            return torch.zeros(1, 1, 4)

    fusion = CaptureFusion()
    model = SimpleNamespace(
        grounded_token_fusion=fusion,
        config=SimpleNamespace(
            sign_found_status_id=1,
            sign_use_status_gate=True,
            sign_use_gt_status_gate=True,
        ),
    )
    grounding = BatchFeature(
        data={
            "sign_hidden": torch.ones(1, 4),
            "sign_bbox_cxcywh": torch.tensor([[0.1, 0.2, 0.3, 0.4]]),
            "sign_status_logits": torch.tensor([[0.0, 1.0, 0.0]]),
        }
    )
    backbone = BatchFeature(
        data={
            "backbone_features": torch.zeros(1, 2, 4),
            "backbone_attention_mask": torch.ones(1, 2),
        }
    )

    Gr00tN1d7._append_grounded_token(
        model,
        backbone,
        grounding,
        conditioning_mode="pred",
        ablation_mode=ablation_mode,
    )

    assert fusion.kwargs["zero_sign_feature"] is zero_sign
    assert fusion.kwargs["zero_bbox_feature"] is zero_bbox
    assert backbone["grounded_token_count"] == 1


def test_grounded_state_residual_uses_last_condition_token():
    head = Gr00tN1d7ActionHead.__new__(Gr00tN1d7ActionHead)
    torch.nn.Module.__init__(head)
    head.config = SimpleNamespace(sign_action_residual_scale=0.1)
    head.grounded_state_projector = torch.nn.Linear(4, 3, bias=False)
    with torch.no_grad():
        head.grounded_state_projector.weight.copy_(
            torch.tensor(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                ]
            )
        )

    vl_embeds = torch.tensor([[[9.0, 9.0, 9.0, 9.0], [1.0, 2.0, 3.0, 4.0]]])
    state_features = torch.tensor([[[10.0, 20.0, 30.0]]])
    backbone = BatchFeature(data={"grounded_token_count": 1})

    result = head._add_grounded_state_residual(backbone, vl_embeds, state_features)
    torch.testing.assert_close(result, torch.tensor([[[10.1, 20.2, 30.3]]]))

    unchanged = head._add_grounded_state_residual(BatchFeature(), vl_embeds, state_features)
    torch.testing.assert_close(unchanged, state_features)


@pytest.mark.parametrize("conditioning_mode", ["pred", "gt_bbox_status"])
def test_training_routes_selected_grounding_condition(conditioning_mode):
    captured = {}
    backbone_output = BatchFeature(data={"backbone_features": torch.zeros(1, 1, 4)})
    grounding_output = BatchFeature(data={"sign_hidden": torch.zeros(1, 4)})

    def append_grounded(backbone, grounding, conditioning_mode=None):
        captured["conditioning_mode"] = conditioning_mode
        return backbone

    model = SimpleNamespace(
        config=SimpleNamespace(sign_training_conditioning_mode=conditioning_mode),
        prepare_input=lambda inputs: ({}, BatchFeature()),
        backbone=lambda inputs: backbone_output,
        _compute_sign_grounding=lambda backbone, action: grounding_output,
        _append_grounded_token=append_grounded,
        action_head=lambda backbone, action: BatchFeature(data={"loss": torch.tensor(0.0)}),
        _merge_grounding_loss=lambda action, grounding: action,
    )

    Gr00tN1d7.forward(model, {})
    assert captured["conditioning_mode"] == conditioning_mode


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
