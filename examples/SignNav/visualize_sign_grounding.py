#!/usr/bin/env python3

"""Visualize SignNav ground-truth and predicted bounding boxes.

The script uses the checkpoint's own processor and the regular training dataset
loader, but stops inference after the sign-grounding branch.  It writes one PNG
per found-sign sample plus a JSON summary containing IoU and prediction spread.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch
from transformers import AutoModel, AutoProcessor

import gr00t.model  # noqa: F401  # Register GR00T AutoModel/AutoProcessor classes.
from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import MessageType


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--processor",
        type=Path,
        default=None,
        help="Processor directory. Defaults to CHECKPOINT/processor, then CHECKPOINT.parent/processor.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("sign_grounding_visualizations"))
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--max-episodes", type=int, default=10)
    parser.add_argument("--found-status-id", type=int, default=1)
    parser.add_argument("--embodiment-tag", default="NEW_EMBODIMENT")
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def resolve_processor_path(checkpoint: Path, explicit: Path | None) -> Path:
    candidates = [explicit, checkpoint / "processor", checkpoint.parent / "processor"]
    for candidate in candidates:
        if candidate is not None and (candidate / "processor_config.json").is_file():
            return candidate
    raise FileNotFoundError(
        "Could not find processor_config.json. Pass the training run's processor directory "
        "with --processor."
    )


def scalar_int(value) -> int:
    return int(np.asarray(value).reshape(-1)[0])


def cxcywh_to_xyxy(box: np.ndarray) -> np.ndarray:
    cx, cy, width, height = box.astype(np.float32)
    return np.clip(
        np.array(
            [cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2],
            dtype=np.float32,
        ),
        0.0,
        1.0,
    )


def box_iou(first: np.ndarray, second: np.ndarray) -> float:
    x1 = max(float(first[0]), float(second[0]))
    y1 = max(float(first[1]), float(second[1]))
    x2 = min(float(first[2]), float(second[2]))
    y2 = min(float(first[3]), float(second[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = max(0.0, float(first[2] - first[0])) * max(
        0.0, float(first[3] - first[1])
    )
    second_area = max(0.0, float(second[2] - second[0])) * max(
        0.0, float(second[3] - second[1])
    )
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def pixel_box(box: np.ndarray, width: int, height: int) -> tuple[int, int, int, int]:
    return (
        round(float(box[0]) * width),
        round(float(box[1]) * height),
        round(float(box[2]) * width),
        round(float(box[3]) * height),
    )


def draw_result(
    image: Image.Image,
    gt_xyxy: np.ndarray,
    pred_xyxy: np.ndarray,
    iou: float,
    pred_status: int,
    found_probability: float,
) -> Image.Image:
    result = image.convert("RGB").copy()
    draw = ImageDraw.Draw(result)
    gt_pixels = pixel_box(gt_xyxy, result.width, result.height)
    pred_pixels = pixel_box(pred_xyxy, result.width, result.height)
    line_width = max(2, round(min(result.size) / 150))
    draw.rectangle(gt_pixels, outline=(0, 255, 0), width=line_width)
    draw.rectangle(pred_pixels, outline=(255, 40, 40), width=line_width)
    label = f"GT=green  pred=red  IoU={iou:.3f}  status={pred_status}  P(found)={found_probability:.3f}"
    text_box = draw.textbbox((0, 0), label)
    text_height = text_box[3] - text_box[1]
    draw.rectangle((0, 0, result.width, text_height + 8), fill=(0, 0, 0))
    draw.text((4, 4), label, fill=(255, 255, 255))
    return result


def main() -> None:
    args = parse_args()
    if args.num_samples <= 0:
        raise ValueError("--num-samples must be positive")

    checkpoint = args.checkpoint.resolve()
    processor_path = resolve_processor_path(checkpoint, args.processor)
    embodiment = EmbodimentTag.resolve(args.embodiment_tag)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model = AutoModel.from_pretrained(checkpoint)
    model.eval().to(device=args.device, dtype=torch.bfloat16)
    processor = AutoProcessor.from_pretrained(processor_path)
    processor.eval()

    all_configs = processor.get_modality_configs()
    if embodiment.value not in all_configs:
        raise KeyError(
            f"Embodiment {embodiment.value!r} is absent from processor configs: "
            f"{sorted(all_configs)}"
        )
    modality_configs = all_configs[embodiment.value]
    loader = LeRobotEpisodeLoader(args.dataset, modality_configs)

    records: list[dict] = []
    pred_boxes: list[np.ndarray] = []
    gt_boxes: list[np.ndarray] = []
    video_key = modality_configs["video"].modality_keys[0]
    max_action_delta = max(modality_configs["action"].delta_indices)

    for episode_position in range(min(len(loader), args.max_episodes)):
        episode = loader[episode_position]
        last_valid_step = len(episode) - max_action_delta
        candidate_steps = [
            step
            for step in range(max(0, last_valid_step))
            if scalar_int(episode["gt_sign_status"].iloc[step]) == args.found_status_id
        ]
        if not candidate_steps:
            continue

        remaining = args.num_samples - len(records)
        # Spread samples over the episode instead of taking adjacent video frames.
        take = min(remaining, len(candidate_steps))
        selected = np.linspace(0, len(candidate_steps) - 1, num=take, dtype=int)
        for selected_position in selected:
            step = candidate_steps[int(selected_position)]
            vla_step = extract_step_data(
                episode,
                step,
                modality_configs,
                embodiment,
                allow_padding=False,
            )
            processed = processor(
                [{"type": MessageType.EPISODE_STEP.value, "content": vla_step}]
            )
            model_inputs = processor.collator([processed])["inputs"]

            with torch.inference_mode():
                backbone_inputs, action_inputs = model.prepare_input(model_inputs)
                backbone_outputs = model.backbone(backbone_inputs)
                grounding = model._compute_sign_grounding(backbone_outputs, action_inputs)
            if grounding is None:
                raise RuntimeError("The model did not produce sign-grounding outputs")

            pred_cxcywh = grounding.sign_bbox_cxcywh[0].float().cpu().numpy()
            logits = grounding.sign_status_logits[0].float().cpu()
            probabilities = torch.softmax(logits, dim=-1).numpy()
            pred_status = int(probabilities.argmax())
            found_probability = float(probabilities[args.found_status_id])
            gt_cxcywh = np.asarray(vla_step.metadata["gt_sign_bbox_cxcywh"], dtype=np.float32)
            gt_xyxy = cxcywh_to_xyxy(gt_cxcywh)
            pred_xyxy = cxcywh_to_xyxy(pred_cxcywh)
            iou = box_iou(gt_xyxy, pred_xyxy)

            raw_image = episode[f"video.{video_key}"].iloc[step]
            if not isinstance(raw_image, Image.Image):
                raw_image = Image.fromarray(np.asarray(raw_image))
            rendered = draw_result(
                raw_image,
                gt_xyxy,
                pred_xyxy,
                iou,
                pred_status,
                found_probability,
            )
            filename = f"sample_{len(records):03d}_episode_{episode_position:03d}_step_{step:06d}.png"
            rendered.save(args.output_dir / filename)

            pred_boxes.append(pred_cxcywh)
            gt_boxes.append(gt_cxcywh)
            records.append(
                {
                    "file": filename,
                    "episode_position": episode_position,
                    "step": step,
                    "iou": iou,
                    "gt_bbox_cxcywh": gt_cxcywh.tolist(),
                    "pred_bbox_cxcywh": pred_cxcywh.tolist(),
                    "pred_status": pred_status,
                    "found_probability": found_probability,
                }
            )
            print(f"[{len(records):02d}/{args.num_samples}] {filename}: IoU={iou:.3f}")
            if len(records) >= args.num_samples:
                break
        if len(records) >= args.num_samples:
            break

    if not records:
        raise RuntimeError(
            f"No samples with gt_sign_status={args.found_status_id} were found in the first "
            f"{args.max_episodes} episodes"
        )

    pred_array = np.stack(pred_boxes)
    gt_array = np.stack(gt_boxes)
    ious = np.asarray([record["iou"] for record in records], dtype=np.float32)
    summary = {
        "checkpoint": str(checkpoint),
        "processor": str(processor_path),
        "dataset": str(args.dataset.resolve()),
        "num_samples": len(records),
        "mean_iou": float(ious.mean()),
        "median_iou": float(np.median(ious)),
        "iou_at_50": float((ious >= 0.5).mean()),
        "pred_bbox_mean_cxcywh": pred_array.mean(axis=0).tolist(),
        "pred_bbox_std_cxcywh": pred_array.std(axis=0).tolist(),
        "gt_bbox_mean_cxcywh": gt_array.mean(axis=0).tolist(),
        "gt_bbox_std_cxcywh": gt_array.std(axis=0).tolist(),
        "samples": records,
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Saved {len(records)} visualizations to {args.output_dir}")
    print(f"Mean IoU: {summary['mean_iou']:.4f}; IoU@0.5: {summary['iou_at_50']:.4f}")
    print(f"Prediction std (cx, cy, w, h): {summary['pred_bbox_std_cxcywh']}")


if __name__ == "__main__":
    main()


'''
srun \
  --partition=all \
  --nodelist=nv178 \
  --nodes=1 \
  --ntasks=1 \
  --gpus=1 \
  --cpus-per-task=16 \
  --mem=64G \
  --container-image=/purestorage/uonrobotics/sujinkim/enroot/gr00t_n1d7_torch290_cu128.sqsh \
  --container-mounts=/purestorage/uonrobotics/sujinkim/:/workspace,/purestorage/uonrobotics/sujinkim/dataset/:/dataset \
  --container-writable \
  --pty bash
  
cd /workspace/Isaac-GR00T

export PYTHONPATH=/workspace/Isaac-GR00T:$PYTHONPATH
export HF_HOME=/workspace/hf_cache
export HF_HUB_CACHE=/workspace/hf_cache/transformers
export TRANSFORMERS_CACHE=/workspace/hf_cache/transformers

RUN_DIR=/workspace/finetune_test/SignNav/gr00t_n1d7/gr00t_n1d7-finetune+sim_v2_lerobot_sign_grounding/gr00t_n1d7-finetune+sim_v2_lerobot_sign_grounding--20260827

python examples/SignNav/visualize_sign_grounding.py \
  --checkpoint "$RUN_DIR/checkpoint-10000" \
  --processor "$RUN_DIR/processor" \
  --dataset /dataset/SignNav/sim_v2_lerobot_sign_grounding \
  --output-dir /workspace/sign_grounding_visualizations/bbox_head_fixed_init \
  --num-samples 20 \
  --device cuda:0
  

'''