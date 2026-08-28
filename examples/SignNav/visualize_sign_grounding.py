#!/usr/bin/env python3

"""Visualize SignNav ground-truth and predicted bounding boxes.

The script uses the checkpoint's own processor and the regular training dataset
loader, but stops inference after the sign-grounding branch.  It writes one PNG
per found-sign sample plus a JSON summary containing IoU and prediction spread.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import textwrap

from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import MessageType
import gr00t.model  # noqa: F401  # Register GR00T AutoModel/AutoProcessor classes.
from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import SIGN_QUERY_MARKER
import numpy as np
from PIL import Image, ImageDraw
import torch
from transformers import AutoModel, AutoProcessor


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
    parser.add_argument(
        "--num-samples-per-episode",
        type=int,
        default=20,
        help="Number of random found-sign frames drawn from one random episode per source goal.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for reproducible goal-balanced sampling.",
    )
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


def ensure_sign_query_marker(text: str) -> str:
    if SIGN_QUERY_MARKER.lower() in text.lower() or "target sign" in text.lower():
        return text
    return f"{text}\n{SIGN_QUERY_MARKER}"


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
    gt_xyxy: np.ndarray | None,
    pred_xyxy: np.ndarray,
    iou: float | None,
    pred_status: int,
    found_probability: float,
    source_goal: str,
    goal: str,
) -> Image.Image:
    result = image.convert("RGB").copy()
    draw = ImageDraw.Draw(result)
    pred_pixels = pixel_box(pred_xyxy, result.width, result.height)
    line_width = max(2, round(min(result.size) / 150))
    if gt_xyxy is not None:
        gt_pixels = pixel_box(gt_xyxy, result.width, result.height)
        draw.rectangle(gt_pixels, outline=(0, 255, 0), width=line_width)
    draw.rectangle(pred_pixels, outline=(255, 40, 40), width=line_width)
    iou_text = f"{iou:.3f}" if iou is not None else "NA"
    metric_label = (
        f"GT=green(if available)  pred=red  IoU={iou_text}  "
        f"status={pred_status}  P(found)={found_probability:.3f}"
    )
    goal_label = (
        f"Goal [{source_goal}]: {textwrap.shorten(goal, width=120, placeholder='...')}"
    )
    label = f"{metric_label}\n{goal_label}"
    text_box = draw.multiline_textbbox((0, 0), label, spacing=2)
    text_height = text_box[3] - text_box[1]
    draw.rectangle((0, 0, result.width, text_height + 8), fill=(0, 0, 0))
    draw.multiline_text((4, 4), label, fill=(255, 255, 255), spacing=2)
    return result


def load_episodes_by_source_goal(dataset: Path) -> dict[str, list[int]]:
    """Read episode-to-goal mappings without decoding any video frames."""
    episodes_path = dataset / "meta" / "episodes.jsonl"
    if not episodes_path.is_file():
        raise FileNotFoundError(f"Missing episode metadata: {episodes_path}")

    episodes_by_goal: dict[str, list[int]] = defaultdict(list)
    with episodes_path.open() as file:
        for line in file:
            metadata = json.loads(line)
            source_goal = metadata.get("source_goal")
            if source_goal is None:
                raise KeyError(f"source_goal is missing from {episodes_path}")
            episodes_by_goal[str(source_goal)].append(int(metadata["episode_index"]))
    return dict(episodes_by_goal)


def main() -> None:
    args = parse_args()
    if args.num_samples_per_episode <= 0:
        raise ValueError("--num-samples-per-episode must be positive")

    checkpoint = args.checkpoint.resolve()
    processor_path = resolve_processor_path(checkpoint, args.processor)
    embodiment = EmbodimentTag.resolve(args.embodiment_tag)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model = AutoModel.from_pretrained(checkpoint, dtype=torch.bfloat16)
    model.eval().to(device=args.device)
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
    ious: list[float] = []
    video_key = modality_configs["video"].modality_keys[0]
    max_action_delta = max(modality_configs["action"].delta_indices)
    rng = np.random.default_rng(args.seed)
    episodes_by_goal = load_episodes_by_source_goal(args.dataset)
    selected_candidates: list[tuple[int, int, str]] = []
    selected_episode_by_goal: dict[str, int] = {}
    for source_goal in sorted(episodes_by_goal):
        candidate_steps = []
        episode_position = -1
        for candidate_episode in rng.permutation(episodes_by_goal[source_goal]):
            episode_position = int(candidate_episode)
            episode = loader[episode_position]
            last_valid_step = len(episode) - max_action_delta
            candidate_steps = list(range(max(0, last_valid_step)))
            if candidate_steps:
                break
        if not candidate_steps:
            print(f"Warning: no valid action-horizon frames available for {source_goal}")
            continue

        selected_episode_by_goal[source_goal] = episode_position
        print(f"Selected episode {episode_position} for {source_goal}")
        take = min(args.num_samples_per_episode, len(candidate_steps))
        if take < args.num_samples_per_episode:
            print(
                f"Warning: {source_goal} episode {episode_position} has only {take} "
                "valid action-horizon frames"
            )
        if take:
            selected_steps = rng.choice(candidate_steps, size=take, replace=False)
            selected_candidates.extend(
                (episode_position, int(step), source_goal) for step in selected_steps
            )
    # Process one episode at a time so decoded video frames from many episodes
    # are never retained in memory simultaneously. Selection remains random;
    # sorting only changes the order in which the chosen samples are rendered.
    selected_candidates.sort(key=lambda candidate: candidate[0])
    loaded_episode_position = None
    episode = None
    for episode_position, step, source_goal in selected_candidates:
        if episode_position != loaded_episode_position:
            episode = loader[episode_position]
            loaded_episode_position = episode_position
        assert episode is not None
        vla_step = extract_step_data(
            episode,
            step,
            modality_configs,
            embodiment,
            allow_padding=False,
        )
        # Sign grounding should be driven by the inference prompt marker, not by GT labels.
        goal = ensure_sign_query_marker(str(vla_step.text))
        vla_step.text = goal
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
        pred_xyxy = cxcywh_to_xyxy(pred_cxcywh)
        gt_cxcywh = None
        gt_xyxy = None
        gt_status = None
        iou = None
        if "gt_sign_bbox_cxcywh" in vla_step.metadata:
            gt_cxcywh = np.asarray(vla_step.metadata["gt_sign_bbox_cxcywh"], dtype=np.float32)
            gt_xyxy = cxcywh_to_xyxy(gt_cxcywh)
            iou = box_iou(gt_xyxy, pred_xyxy)
            ious.append(iou)
            gt_boxes.append(gt_cxcywh)
        if "gt_sign_status" in vla_step.metadata:
            gt_status = scalar_int(vla_step.metadata["gt_sign_status"])

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
            source_goal,
            goal,
        )
        filename = (
            f"sample_{len(records):03d}_{source_goal}_"
            f"episode_{episode_position:03d}_step_{step:06d}.png"
        )
        rendered.save(args.output_dir / filename)

        pred_boxes.append(pred_cxcywh)
        records.append(
            {
                "file": filename,
                "episode_position": episode_position,
                "step": step,
                "source_goal": source_goal,
                "goal": goal,
                "iou": iou,
                "gt_bbox_cxcywh": gt_cxcywh.tolist() if gt_cxcywh is not None else None,
                "gt_sign_status": gt_status,
                "pred_bbox_cxcywh": pred_cxcywh.tolist(),
                "pred_status": pred_status,
                "found_probability": found_probability,
            }
        )
        iou_text = f"{iou:.3f}" if iou is not None else "NA"
        print(
            f"[{len(records):03d}/{len(selected_candidates)}] {filename}: "
            f"IoU={iou_text} goal={source_goal!r}"
        )

    if not records:
        raise RuntimeError("No valid action-horizon samples were found in the selected episodes")

    pred_array = np.stack(pred_boxes)
    iou_array = np.asarray(ious, dtype=np.float32)
    summary = {
        "checkpoint": str(checkpoint),
        "processor": str(processor_path),
        "dataset": str(args.dataset.resolve()),
        "num_samples": len(records),
        "num_samples_per_episode": args.num_samples_per_episode,
        "seed": args.seed,
        "selected_episode_by_goal": selected_episode_by_goal,
        "source_goal_counts": dict(
            sorted(Counter(record["source_goal"] for record in records).items())
        ),
        "goal_counts": dict(sorted(Counter(record["goal"] for record in records).items())),
        "num_samples_with_gt_bbox": len(gt_boxes),
        "mean_iou": float(iou_array.mean()) if len(iou_array) else None,
        "median_iou": float(np.median(iou_array)) if len(iou_array) else None,
        "iou_at_50": float((iou_array >= 0.5).mean()) if len(iou_array) else None,
        "pred_bbox_mean_cxcywh": pred_array.mean(axis=0).tolist(),
        "pred_bbox_std_cxcywh": pred_array.std(axis=0).tolist(),
        "gt_bbox_mean_cxcywh": np.stack(gt_boxes).mean(axis=0).tolist() if gt_boxes else None,
        "gt_bbox_std_cxcywh": np.stack(gt_boxes).std(axis=0).tolist() if gt_boxes else None,
        "samples": records,
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Saved {len(records)} visualizations to {args.output_dir}")
    if summary["mean_iou"] is not None:
        print(f"Mean IoU: {summary['mean_iou']:.4f}; IoU@0.5: {summary['iou_at_50']:.4f}")
    else:
        print("Mean IoU: NA; no GT bbox values were available")
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

RUN_DIR=/workspace/finetune_test/SignNav/gr00t_n1d7/gr00t_n1d7-finetune+sim_v2_lerobot_sign_grounding/gr00t_n1d7-finetune+sim_v2_lerobot_sign_grounding--20260827--w1_5_0p5--signw0p05
RUN_DIR=/workspace/finetune_test/SignNav/gr00t_n1d7/gr00t_n1d7-finetune+sim_v2_lerobot_sign_grounding/gr00t_n1d7-finetune+sim_v2_lerobot_sign_grounding--20260827--w0p5_2_2--signw0p05

python examples/SignNav/visualize_sign_grounding.py \
  --checkpoint "$RUN_DIR/checkpoint-100000" \
  --processor "$RUN_DIR/processor" \
  --dataset /dataset/SignNav/sim_v2_lerobot_sign_grounding \
  --output-dir /workspace/sign_grounding_visualizations/w0p5_2_2--signw0p05@100000 \
  --num-samples-per-episode 20 \
  --seed 42 \
  --device cuda:0
  
  
'''
