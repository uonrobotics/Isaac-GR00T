#!/usr/bin/env python3

"""Persistent SAM3 + Qwen3 target-crop worker for E2-A inference."""

from __future__ import annotations

import argparse
import base64
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import re
import sys
import time
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline-module-path", type=Path, required=True)
    parser.add_argument("--sam-model-id", default="facebook/sam3")
    parser.add_argument("--qwen-model-id", default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto")
    parser.add_argument("--qwen-max-pixels", type=int, default=1024 * 28 * 28)
    return parser.parse_args()


def load_pipeline(path: Path):
    path = path.expanduser().resolve()
    spec = importlib.util.spec_from_file_location("signnav_sam3_qwen3_pipeline", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load SAM3+Qwen3 module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def decode_image_b64(image_b64: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(image_b64))).convert("RGB")


def encode_image_b64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def is_eligible_candidate(candidate: dict[str, Any]) -> bool:
    reading = candidate["qwen"]
    if reading["status"] != "readable":
        return False
    if candidate["area_label_count"] != 1:
        return False
    return (
        reading["panel_count"] == 1
        and reading["full_panel_visible"]
        and reading["text_complete"]
        and reading["arrow_visible"]
        and reading["text_arrow_same_panel"]
    )


def select_target_candidate(
    candidates: list[dict[str, Any]],
    target_area: int,
    size_dominance_ratio: float,
) -> tuple[int | None, str | None, list[int]]:
    matching_indices = [
        index
        for index, candidate in enumerate(candidates)
        if candidate.get("eligible_for_matching", False)
        and target_area in candidate.get("included_areas", [])
    ]
    if len(matching_indices) == 0:
        return None, None, matching_indices
    if len(matching_indices) == 1:
        return matching_indices[0], "single_matching_candidate", matching_indices

    ranked_by_size = sorted(
        matching_indices,
        key=lambda index: candidates[index]["mask_area_pixels"],
        reverse=True,
    )
    largest_index, second_index = ranked_by_size[:2]
    largest_area = candidates[largest_index]["mask_area_pixels"]
    second_area = candidates[second_index]["mask_area_pixels"]
    if largest_area / max(second_area, 1) >= size_dominance_ratio:
        return largest_index, "largest_mask_area", matching_indices

    minimum_tie_area = largest_area / size_dominance_ratio
    tie_candidate_indices = [
        index
        for index in ranked_by_size
        if candidates[index]["mask_area_pixels"] >= minimum_tie_area
    ]
    selected_index = max(
        tie_candidate_indices,
        key=lambda index: (
            candidates[index]["sam_score"],
            candidates[index]["mask_area_pixels"],
            -index,
        ),
    )
    return selected_index, "sam_score_size_tiebreak", matching_indices


def clamp_pixel_box(
    box: tuple[int, int, int, int],
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    x1 = max(0, min(width - 1, int(x1)))
    y1 = max(0, min(height - 1, int(y1)))
    x2 = max(x1 + 1, min(width, int(x2)))
    y2 = max(y1 + 1, min(height, int(y2)))
    return x1, y1, x2, y2


def pad_to_aspect_ratio(image: Image.Image, target_size: tuple[int, int]) -> Image.Image:
    target_width, target_height = target_size
    target_aspect = target_width / target_height
    crop_aspect = image.width / image.height
    if abs(crop_aspect - target_aspect) < 1e-6:
        return image

    if crop_aspect > target_aspect:
        padded_width = image.width
        padded_height = round(image.width / target_aspect)
    else:
        padded_height = image.height
        padded_width = round(image.height * target_aspect)

    canvas = Image.new("RGB", (padded_width, padded_height), color=(0, 0, 0))
    offset = ((padded_width - image.width) // 2, (padded_height - image.height) // 2)
    canvas.paste(image, offset)
    return canvas


def draw_selected_bbox_view(
    image: Image.Image,
    candidates: list[dict[str, Any]],
    selected_index: int | None,
    target_area: int,
) -> Image.Image:
    canvas = image.copy().convert("RGB")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    thin = max(2, image.width // 400)
    thick = max(4, image.width // 250)
    for index, candidate in enumerate(candidates):
        box = tuple(round(v) for v in candidate["bbox_xyxy"])
        color = "#00a7ff"
        if index == selected_index:
            color = "#ff3030"
        elif candidate.get("eligible_for_matching", False):
            color = "#7fd36b"
        draw.rectangle(box, outline=color, width=thick if index == selected_index else thin)
        reading = candidate.get("qwen", {})
        label = f"#{index} {reading.get('sign_text') or '-'}"
        draw.text(
            (box[0] + 3, max(0, box[1] - 16)),
            label,
            fill="white",
            font=font,
            stroke_width=2,
            stroke_fill="#000000",
        )

    header = f"target AREA_{target_area}"
    if selected_index is not None:
        selected = candidates[selected_index]
        reading = selected["qwen"]
        header += f" | selected #{selected_index}: {reading['sign_text']} / {reading['arrow_direction']}"
    else:
        header += " | no target marker"
    draw.rectangle((0, 0, image.width, 24), fill="#000000")
    draw.text((8, 6), header, fill="#ffffff", font=font)
    return canvas


def sanitize_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    keep = {
        "bbox_xyxy",
        "sam_score",
        "bbox_height_ratio",
        "bbox_area_ratio",
        "mask_area_pixels",
        "candidate_index",
        "crop_bbox_xyxy",
        "qwen",
        "included_areas",
        "area_label_count",
        "eligible_for_matching",
    }
    return {key: candidate[key] for key in keep if key in candidate}


def process_request(
    request: dict[str, Any],
    pipeline,
    sam_model,
    sam_processor,
    qwen_model,
    qwen_processor,
    device: str,
    dtype,
    sam_model_id: str,
    qwen_model_id: str,
) -> dict[str, Any]:
    image = decode_image_b64(request["image_b64"])
    target_area = int(request["target_area"])
    start = time.time()

    with contextlib.redirect_stdout(sys.stderr):
        candidates = pipeline.run_sam3(
            image=image,
            model=sam_model,
            processor=sam_processor,
            prompt=request["sam_prompt"],
            threshold=float(request["sam_threshold"]),
            mask_threshold=float(request["sam_mask_threshold"]),
            device=device,
            dtype=dtype,
            min_height_ratio=float(request["min_box_height_ratio"]),
            min_area_ratio=float(request["min_box_area_ratio"]),
            nms_iou_threshold=float(request["nms_iou"]),
            merged_containment_threshold=float(request["merged_containment_threshold"]),
            merged_child_max_iou=float(request["merged_child_max_iou"]),
        )

        for index, candidate in enumerate(candidates):
            crop, crop_box = pipeline.make_crop(
                image,
                candidate["bbox_xyxy"],
                float(request["crop_padding_ratio"]),
                float(request["crop_upscale"]),
            )
            reading, raw_response = pipeline.read_panel_with_qwen(
                crop,
                qwen_model,
                qwen_processor,
                int(request["qwen_max_new_tokens"]),
            )
            candidate["candidate_index"] = index
            candidate["crop_bbox_xyxy"] = crop_box
            candidate["qwen"] = reading
            candidate["qwen_raw_response"] = raw_response
            candidate["included_areas"] = pipeline.parse_area_set(reading["sign_text"])
            candidate["area_label_count"] = pipeline.count_area_labels(reading["sign_text"])
            candidate["eligible_for_matching"] = is_eligible_candidate(candidate)

    selected_index, selection_reason, matching_indices = select_target_candidate(
        candidates,
        target_area,
        float(request["size_dominance_ratio"]),
    )
    selected = candidates[selected_index] if selected_index is not None else None
    bbox_view = draw_selected_bbox_view(image, candidates, selected_index, target_area)

    if selected is None:
        bbox_state = {
            "bbox_status": 0.0,
            "bbox_x1": 0.0,
            "bbox_y1": 0.0,
            "bbox_x2": 0.0,
            "bbox_y2": 0.0,
        }
        crop_image = Image.new("RGB", image.size)
    else:
        x1, y1, x2, y2 = clamp_pixel_box(
            tuple(round(v) for v in selected["bbox_xyxy"]),
            image.width,
            image.height,
        )
        crop_x1, crop_y1, crop_x2, crop_y2 = clamp_pixel_box(
            tuple(round(v) for v in selected["crop_bbox_xyxy"]),
            image.width,
            image.height,
        )
        bbox_state = {
            "bbox_status": 1.0,
            "bbox_x1": x1 / image.width,
            "bbox_y1": y1 / image.height,
            "bbox_x2": x2 / image.width,
            "bbox_y2": y2 / image.height,
        }
        crop_image = image.crop((crop_x1, crop_y1, crop_x2, crop_y2))
        crop_image = pad_to_aspect_ratio(crop_image, image.size)
        crop_image = crop_image.resize(image.size, Image.Resampling.BICUBIC)

    timing = {
        "status": "found" if selected is not None else "not_found",
        "target_area": target_area,
        "candidate_count": len(candidates),
        "selected_candidate_index": selected_index,
        "matching_candidate_indices": matching_indices,
        "selection_reason": selection_reason,
        "matched_sign_text": selected["qwen"]["sign_text"] if selected else None,
        "arrow_direction": selected["qwen"]["arrow_direction"] if selected else None,
        "bbox_state": bbox_state,
        "bbox_xyxy": selected["bbox_xyxy"] if selected else None,
        "bbox_normalized_xyxy": (
            [
                selected["bbox_xyxy"][0] / image.width,
                selected["bbox_xyxy"][1] / image.height,
                selected["bbox_xyxy"][2] / image.width,
                selected["bbox_xyxy"][3] / image.height,
            ]
            if selected
            else None
        ),
        "sam_model_id": sam_model_id,
        "qwen_model_id": qwen_model_id,
        "worker_compute_ms": (time.time() - start) * 1000.0,
    }
    return {
        "ok": True,
        "crop_b64": encode_image_b64(crop_image),
        "bbox_view_b64": encode_image_b64(bbox_view),
        "timing": timing,
        "candidates": [sanitize_candidate(candidate) for candidate in candidates],
    }


def main() -> None:
    args = parse_args()
    pipeline = load_pipeline(args.pipeline_module_path)
    device = pipeline.resolve_device(args.device)
    dtype = pipeline.resolve_dtype(args.dtype, device)
    with contextlib.redirect_stdout(sys.stderr):
        sam_model, sam_processor = pipeline.load_sam3(args.sam_model_id, device, dtype)
        qwen_model, qwen_processor = pipeline.load_qwen(
            args.qwen_model_id,
            device,
            dtype,
            args.qwen_max_pixels,
        )
    print("[SAM3+Qwen3 worker] ready", file=sys.stderr, flush=True)

    for line in sys.stdin:
        try:
            request = json.loads(line)
            response = process_request(
                request,
                pipeline,
                sam_model,
                sam_processor,
                qwen_model,
                qwen_processor,
                device,
                dtype,
                args.sam_model_id,
                args.qwen_model_id,
            )
        except Exception as exc:
            response = {"ok": False, "error": str(exc)}
            print(f"[SAM3+Qwen3 worker] request failed: {exc}", file=sys.stderr, flush=True)
        print(json.dumps(response, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
