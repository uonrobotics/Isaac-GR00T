#!/usr/bin/env python3

"""Visualize SignNav predicted grounding boxes for images or random dataset frames."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any

from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import MessageType, VLAStepData
import gr00t.model  # noqa: F401  # Register GR00T AutoModel/AutoProcessor classes.
from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import SIGN_QUERY_MARKER
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from transformers import AutoModel, AutoProcessor


PROMPT_TEMPLATE = (
    "Find the sign panel containing Area {area} and use it to choose the navigation action."
)
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}
SIGN_QUERY_MARKERS = (SIGN_QUERY_MARKER, SIGN_QUERY_MARKER.lower(), "target sign")
# The processor appends the sign-query marker when grounding metadata exists.
# Use a dummy status to reproduce that inference query point, then remove the
# dummy tensor before forward so only predictions are visualized.
APPEND_QUERY_MARKER_METADATA = {"gt_sign_status": np.asarray(0, dtype=np.int64)}


@dataclass
class Sample:
    image: Image.Image
    prompt: str
    source: str
    output_name: str
    source_goal: str = ""
    gt_bbox_cxcywh: np.ndarray | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path, help="Input image file or directory of images.")
    source.add_argument(
        "--dataset", type=Path, help="LeRobot dataset to sample random frames from."
    )
    parser.add_argument(
        "--processor",
        type=Path,
        default=None,
        help="Processor directory. Defaults to CHECKPOINT, CHECKPOINT/processor, then CHECKPOINT.parent/processor.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("sign_grounding_visualizations"))
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Single output image path. Only valid when --image points to one file.",
    )
    parser.add_argument(
        "--area",
        type=int,
        default=None,
        help="Target area for image/file-directory mode. Ignored when --prompt is provided.",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Prompt for image/file-directory mode. Defaults to an Area prompt from --area.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively collect images when --image points to a directory.",
    )
    parser.add_argument(
        "--num-samples-per-episode",
        type=int,
        default=20,
        help="Dataset mode: random frames drawn from one random episode per source goal.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Dataset mode: random seed used for reproducible goal-balanced sampling.",
    )
    parser.add_argument("--found-status-id", type=int, default=1)
    parser.add_argument("--embodiment-tag", default="NEW_EMBODIMENT")
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def resolve_processor_path(checkpoint: Path, explicit: Path | None) -> Path:
    candidates = [explicit, checkpoint, checkpoint / "processor", checkpoint.parent / "processor"]
    for candidate in candidates:
        if candidate is not None and (candidate / "processor_config.json").is_file():
            return candidate
    raise FileNotFoundError(
        "Could not find processor_config.json. Pass the training run's processor directory "
        "with --processor."
    )


def state_dim(processor: Any, embodiment: EmbodimentTag, key: str) -> int:
    try:
        params = processor.state_action_processor.norm_params[embodiment.value]["state"][key]
        return int(np.asarray(params["dim"]).reshape(-1)[0])
    except Exception:
        return 1


def build_prompt(area: int | None, explicit_prompt: str | None) -> str:
    if explicit_prompt is not None:
        return explicit_prompt
    if area is None:
        raise ValueError("Image mode requires --area unless --prompt is provided.")
    return PROMPT_TEMPLATE.format(area=area)


def build_step_data(
    image: Image.Image,
    processor: Any,
    modality_configs: dict[str, Any],
    embodiment: EmbodimentTag,
    prompt: str,
) -> VLAStepData:
    image = image.convert("RGB")
    images = {key: [image.copy()] for key in modality_configs["video"].modality_keys}
    states = {
        key: np.zeros((1, state_dim(processor, embodiment, key)), dtype=np.float32)
        for key in modality_configs["state"].modality_keys
    }
    return VLAStepData(
        images=images,
        states=states,
        actions={},
        text=prompt,
        embodiment=embodiment,
        metadata=APPEND_QUERY_MARKER_METADATA,
    )


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


def pixel_box(box: np.ndarray, width: int, height: int) -> tuple[int, int, int, int]:
    return (
        round(float(box[0]) * width),
        round(float(box[1]) * height),
        round(float(box[2]) * width),
        round(float(box[3]) * height),
    )


def area_label(source_goal: str, prompt: str) -> str:
    for text in (source_goal, prompt):
        match = re.search(r"area\s*[_-]?(\d+)", text, flags=re.IGNORECASE)
        if match:
            return f"area {int(match.group(1))}"

        match = re.search(r"goal\s*[_-]?(\d+)", text, flags=re.IGNORECASE)
        if match:
            return f"area {int(match.group(1))}"
    return source_goal or "unknown"


def safe_stem(text: str, max_len: int = 80) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text.strip())
    text = text.strip("._") or "sample"
    return text[:max_len].strip("._") or "sample"


def goal_stem(source_goal: str, prompt: str, area: int | None = None) -> str:
    if area is not None:
        return f"goal_area_{area:02d}"

    label = area_label(source_goal, prompt)
    match = re.search(r"area\s*(\d+)", label, flags=re.IGNORECASE)
    if match:
        return f"goal_area_{int(match.group(1)):02d}"
    return f"goal_{safe_stem(label, max_len=48)}"


def find_sign_query_index(
    tokenizer: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    rendered_text: str,
) -> int:
    valid_positions = attention_mask.nonzero(as_tuple=False).flatten()
    unpadded_ids = input_ids[valid_positions]

    for marker in SIGN_QUERY_MARKERS:
        marker_ids = tokenizer(
            marker,
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"][0].to(input_ids.device)
        marker_len = marker_ids.numel()
        if marker_len == 0 or unpadded_ids.numel() < marker_len:
            continue

        matches = (
            (unpadded_ids.unfold(0, marker_len, 1) == marker_ids.unsqueeze(0))
            .all(dim=1)
            .nonzero(as_tuple=False)
        )
        if matches.numel() > 0:
            marker_start = int(matches[-1].item())
            return int(valid_positions[marker_start + marker_len - 1].item())

    rendered_lower = rendered_text.lower()
    marker = next(
        (candidate for candidate in SIGN_QUERY_MARKERS if candidate in rendered_lower), None
    )
    if marker is not None:
        marker_end = rendered_lower.rindex(marker) + len(marker)
        prefix_ids = tokenizer(
            rendered_text[:marker_end],
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"][0]
        return int(valid_positions[0].item() + prefix_ids.numel() - 1)

    raise RuntimeError(f"Could not find sign query marker in prompt: {rendered_text!r}")


def add_sign_query_index(
    model_inputs: dict[str, Any], processed: dict[str, Any], processor: Any
) -> None:
    if "sign_query_index" in model_inputs:
        return
    rendered_text = processed["vlm_content"]["text"]
    tokenizer = processor.collator.processor.tokenizer
    model_inputs["sign_query_index"] = torch.tensor(
        [
            find_sign_query_index(tokenizer, input_ids, attention_mask, rendered_text)
            for input_ids, attention_mask in zip(
                model_inputs["input_ids"],
                model_inputs["attention_mask"],
            )
        ],
        dtype=torch.long,
    )


def remove_inference_marker_metadata(model_inputs: dict[str, Any]) -> None:
    for key in ("gt_sign_bbox_cxcywh", "gt_sign_status"):
        model_inputs.pop(key, None)


def predict_grounding(
    model: Any,
    processor: Any,
    modality_configs: dict[str, Any],
    embodiment: EmbodimentTag,
    image: Image.Image,
    prompt: str,
    found_status_id: int,
) -> dict[str, Any]:
    vla_step = build_step_data(image, processor, modality_configs, embodiment, prompt)
    processed = processor([{"type": MessageType.EPISODE_STEP.value, "content": vla_step}])
    model_inputs = processor.collator([processed])["inputs"]
    add_sign_query_index(model_inputs, processed, processor)
    remove_inference_marker_metadata(model_inputs)

    with torch.inference_mode():
        backbone_inputs, action_inputs = model.prepare_input(model_inputs)
        backbone_outputs = model.backbone(backbone_inputs)
        grounding = model._compute_sign_grounding(backbone_outputs, action_inputs)
    if grounding is None:
        raise RuntimeError(
            "The model did not produce sign-grounding outputs. "
            f"model_input_keys={sorted(model_inputs.keys())}"
        )

    pred_cxcywh = grounding.sign_bbox_cxcywh[0].float().cpu().numpy()
    pred_xyxy = cxcywh_to_xyxy(pred_cxcywh)
    logits = grounding.sign_status_logits[0].float().cpu()
    probabilities = torch.softmax(logits, dim=-1).numpy()
    pred_status = int(probabilities.argmax())
    return {
        "model_input_keys": sorted(model_inputs.keys()),
        "sign_query_index": model_inputs["sign_query_index"].tolist(),
        "pred_bbox_cxcywh": pred_cxcywh,
        "pred_bbox_xyxy": pred_xyxy,
        "pred_status": pred_status,
        "found_probability": float(probabilities[found_status_id]),
        "probabilities": probabilities,
    }


def load_overlay_font(image: Image.Image) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    font_size = max(24, round(min(image.size) / 18))
    for font_path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    ):
        if Path(font_path).is_file():
            return ImageFont.truetype(font_path, font_size)
    return ImageFont.load_default()


def draw_result(
    image: Image.Image,
    pred_xyxy: np.ndarray,
    pred_status: int,
    found_probability: float,
    source_goal: str,
    prompt: str,
    gt_bbox_cxcywh: np.ndarray | None = None,
) -> Image.Image:
    result = image.convert("RGB").copy()
    draw = ImageDraw.Draw(result)
    line_width = max(2, round(min(result.size) / 150))
    if gt_bbox_cxcywh is not None:
        gt_pixels = pixel_box(cxcywh_to_xyxy(gt_bbox_cxcywh), result.width, result.height)
        draw.rectangle(gt_pixels, outline=(46, 204, 113), width=line_width)
    if pred_status != 0:
        pred_pixels = pixel_box(pred_xyxy, result.width, result.height)
        draw.rectangle(pred_pixels, outline=(255, 40, 40), width=line_width)

    status_label = "no bbox" if pred_status == 0 else str(pred_status)
    label = (
        f"goal: {area_label(source_goal, prompt)}\n"
        f"bbox status: {status_label}\n"
        f"P(found): {found_probability:.3f}"
    )
    font = load_overlay_font(result)
    spacing = max(4, round(min(result.size) / 120))
    text_box = draw.multiline_textbbox((0, 0), label, font=font, spacing=spacing)
    text_width = text_box[2] - text_box[0]
    text_height = text_box[3] - text_box[1]
    padding = max(12, round(min(result.size) / 60))
    x0 = padding
    y0 = result.height - text_height - padding * 3
    status_color = (46, 204, 113) if pred_status != 0 else (255, 80, 80)
    draw.rectangle(
        (x0, y0, x0 + text_width + padding * 2, y0 + text_height + padding * 2),
        fill=(0, 0, 0),
    )
    draw.rectangle(
        (x0, y0, x0 + max(6, padding // 2), y0 + text_height + padding * 2),
        fill=status_color,
    )
    draw.multiline_text(
        (x0 + padding, y0 + padding),
        label,
        font=font,
        fill=status_color,
        spacing=spacing,
        stroke_width=max(1, round(min(result.size) / 360)),
        stroke_fill=(0, 0, 0),
    )
    return result


def iter_image_files(path: Path, recursive: bool) -> list[Path]:
    if path.is_file():
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image extension: {path}")
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(path)
    globber = path.rglob if recursive else path.glob
    images = sorted(p for p in globber("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    if not images:
        raise RuntimeError(f"No images found under {path}")
    return images


def make_image_samples(
    image_path: Path,
    prompt: str,
    area: int | None,
    output_dir: Path,
    single_output: Path | None,
    recursive: bool,
) -> tuple[list[Sample], Path]:
    image_files = iter_image_files(image_path, recursive)
    if single_output is not None and len(image_files) != 1:
        raise ValueError("--output is only valid when --image points to one file.")

    output_root = single_output.parent if single_output is not None else output_dir
    samples = []
    goal = goal_stem("", prompt, area)
    for path in image_files:
        output_name = (
            single_output.name
            if single_output is not None
            else f"{safe_stem(path.stem)}__{goal}__sign_grounding.png"
        )
        samples.append(
            Sample(
                image=Image.open(path).convert("RGB"),
                prompt=prompt,
                source=str(path),
                output_name=output_name,
            )
        )
    return samples, output_root


def load_episodes_by_source_goal(dataset: Path) -> dict[str, list[int]]:
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


def make_dataset_samples(
    dataset: Path,
    modality_configs: dict[str, Any],
    embodiment: EmbodimentTag,
    num_samples_per_episode: int,
    seed: int,
) -> tuple[list[Sample], dict[str, int]]:
    if num_samples_per_episode <= 0:
        raise ValueError("--num-samples-per-episode must be positive")

    loader = LeRobotEpisodeLoader(dataset, modality_configs)
    video_key = modality_configs["video"].modality_keys[0]
    max_action_delta = max(modality_configs["action"].delta_indices)
    rng = np.random.default_rng(seed)
    episodes_by_goal = load_episodes_by_source_goal(dataset)
    selected_episode_by_goal: dict[str, int] = {}
    selected_candidates: list[tuple[int, int, str]] = []

    for source_goal, episodes in sorted(episodes_by_goal.items()):
        candidate_steps = []
        episode_position = -1
        for candidate_episode in rng.permutation(episodes):
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
        take = min(num_samples_per_episode, len(candidate_steps))
        if take < num_samples_per_episode:
            print(
                f"Warning: {source_goal} episode {episode_position} has only {take} "
                "valid action-horizon frames"
            )
        selected_steps = rng.choice(candidate_steps, size=take, replace=False)
        selected_candidates.extend(
            (episode_position, int(step), source_goal) for step in selected_steps
        )

    samples = []
    selected_candidates.sort(key=lambda candidate: candidate[0])
    loaded_episode_position = None
    episode = None
    for idx, (episode_position, step, source_goal) in enumerate(selected_candidates):
        if episode_position != loaded_episode_position:
            episode = loader[episode_position]
            loaded_episode_position = episode_position
        assert episode is not None

        source_step = extract_step_data(
            episode,
            step,
            modality_configs,
            embodiment,
            allow_padding=False,
        )
        metadata = source_step.metadata or {}
        gt_bbox = None
        if "gt_sign_bbox_cxcywh" in metadata:
            gt_bbox = np.asarray(metadata["gt_sign_bbox_cxcywh"], dtype=np.float32)

        raw_image = episode[f"video.{video_key}"].iloc[step]
        if not isinstance(raw_image, Image.Image):
            raw_image = Image.fromarray(np.asarray(raw_image))
        prompt = str(source_step.text)
        goal = goal_stem(source_goal, prompt)
        filename = (
            f"sample_{idx:03d}__{safe_stem(source_goal, max_len=48)}__{goal}__"
            f"episode_{episode_position:03d}__step_{step:06d}.png"
        )
        samples.append(
            Sample(
                image=raw_image.convert("RGB"),
                prompt=prompt,
                source=f"dataset:{dataset}:episode={episode_position}:step={step}",
                output_name=filename,
                source_goal=source_goal,
                gt_bbox_cxcywh=gt_bbox,
            )
        )

    if not samples:
        raise RuntimeError("No valid action-horizon samples were found in the selected episodes")
    return samples, selected_episode_by_goal


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    processor_path = resolve_processor_path(checkpoint, args.processor)
    embodiment = EmbodimentTag.resolve(args.embodiment_tag)

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

    selected_episode_by_goal = None
    if args.image is not None:
        prompt = build_prompt(args.area, args.prompt)
        samples, output_dir = make_image_samples(
            args.image.expanduser().resolve(),
            prompt,
            args.area,
            args.output_dir,
            args.output.expanduser().resolve() if args.output is not None else None,
            args.recursive,
        )
    else:
        samples, selected_episode_by_goal = make_dataset_samples(
            args.dataset.expanduser().resolve(),
            modality_configs,
            embodiment,
            args.num_samples_per_episode,
            args.seed,
        )
        output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    pred_boxes: list[np.ndarray] = []
    for idx, sample in enumerate(samples):
        prediction = predict_grounding(
            model,
            processor,
            modality_configs,
            embodiment,
            sample.image,
            sample.prompt,
            args.found_status_id,
        )
        pred_boxes.append(prediction["pred_bbox_cxcywh"])
        output_path = output_dir / sample.output_name
        rendered = draw_result(
            sample.image,
            prediction["pred_bbox_xyxy"],
            prediction["pred_status"],
            prediction["found_probability"],
            sample.source_goal,
            sample.prompt,
            sample.gt_bbox_cxcywh,
        )
        rendered.save(output_path)

        record = {
            "file": output_path.name,
            "source": sample.source,
            "source_goal": sample.source_goal,
            "prompt": sample.prompt,
            "model_input_keys": prediction["model_input_keys"],
            "sign_query_index": prediction["sign_query_index"],
            "pred_bbox_cxcywh": prediction["pred_bbox_cxcywh"].tolist(),
            "pred_bbox_xyxy": prediction["pred_bbox_xyxy"].tolist(),
            "pred_bbox_pixel_xyxy": list(
                pixel_box(prediction["pred_bbox_xyxy"], sample.image.width, sample.image.height)
            ),
            "pred_status": prediction["pred_status"],
            "found_probability": prediction["found_probability"],
            "probabilities": prediction["probabilities"].tolist(),
        }
        records.append(record)
        print(
            f"[{idx + 1:03d}/{len(samples):03d}] saved {output_path} "
            f"status={record['pred_status']} P(found)={record['found_probability']:.4f}"
        )

    pred_array = np.stack(pred_boxes)
    summary = {
        "checkpoint": str(checkpoint),
        "processor": str(processor_path),
        "mode": "image" if args.image is not None else "dataset",
        "num_samples": len(records),
        "pred_bbox_mean_cxcywh": pred_array.mean(axis=0).tolist(),
        "pred_bbox_std_cxcywh": pred_array.std(axis=0).tolist(),
        "source_goal_counts": dict(
            sorted(Counter(record["source_goal"] for record in records).items())
        ),
        "samples": records,
    }
    if args.image is not None:
        summary["image"] = str(args.image.expanduser().resolve())
        summary["area"] = args.area
    else:
        summary.update(
            {
                "dataset": str(args.dataset.expanduser().resolve()),
                "num_samples_per_episode": args.num_samples_per_episode,
                "seed": args.seed,
                "selected_episode_by_goal": selected_episode_by_goal,
            }
        )

    if args.image is not None:
        summary_name = (
            f"summary__{goal_stem('', build_prompt(args.area, args.prompt), args.area)}.json"
        )
    else:
        summary_name = f"summary__dataset_seed_{args.seed}.json"
    summary_path = output_dir / summary_name
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Saved {len(records)} visualizations to {output_dir}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()


"""
# 이미지 1장
uv run python examples/SignNav/visualize_sign_grounding.py \
  --checkpoint /nas/sujinkim/model/SignNav/gr00t_n1d7/gr00t_n1d7-finetune+sim_v2_lerobot_sign_grounding/gr00t_n1d7-finetune+sim_v2_lerobot_sign_grounding--20260828--w0p5_2_2--signw0p05/gr00t_n1d7-finetune+sim_v2_lerobot_sign_grounding--20260828--w0p5_2_2--signw0p05/checkpoint-100000 \
  --processor /nas/sujinkim/model/SignNav/gr00t_n1d7/gr00t_n1d7-finetune+sim_v2_lerobot_sign_grounding/gr00t_n1d7-finetune+sim_v2_lerobot_sign_grounding--20260828--w0p5_2_2--signw0p05/gr00t_n1d7-finetune+sim_v2_lerobot_sign_grounding--20260828--w0p5_2_2--signw0p05/processor \
  --image /home/sujin/workspace/physical-ai/Isaac-GR00T/visualization_test/warehouse1.jpg \
  --area 6 \
  --output-dir /home/sujin/workspace/physical-ai/Isaac-GR00T/visualization_test/output \
  --device cuda:0
  
# 이미지 폴더
uv run python examples/SignNav/visualize_sign_grounding.py \
  --checkpoint /nas/sujinkim/model/SignNav/gr00t_n1d7/gr00t_n1d7-finetune+sim_v2_lerobot_sign_grounding/gr00t_n1d7-finetune+sim_v2_lerobot_sign_grounding--20260828--w0p5_2_2--signw0p05/gr00t_n1d7-finetune+sim_v2_lerobot_sign_grounding--20260828--w0p5_2_2--signw0p05/checkpoint-100000 \
  --processor /nas/sujinkim/model/SignNav/gr00t_n1d7/gr00t_n1d7-finetune+sim_v2_lerobot_sign_grounding/gr00t_n1d7-finetune+sim_v2_lerobot_sign_grounding--20260828--w0p5_2_2--signw0p05/gr00t_n1d7-finetune+sim_v2_lerobot_sign_grounding--20260828--w0p5_2_2--signw0p05/processor \
  --image /home/sujin/workspace/physical-ai/Isaac-GR00T/visualization_test/from_dataset \
  --area 1 \
  --output-dir /home/sujin/workspace/physical-ai/Isaac-GR00T/visualization_test/output/from_dataset \
  --device cuda:0
  
# dataset에서 랜덤 frame 긁기
uv run python examples/SignNav/visualize_sign_grounding.py \
  --checkpoint /path/to/checkpoint \
  --processor /path/to/processor \
  --dataset /path/to/lerobot_dataset \
  --output-dir /home/sujin/workspace/physical-ai/Isaac-GR00T/visualization_test/output \
  --num-samples-per-episode 20 \
  --seed 42 \
  --device cuda:0
"""
