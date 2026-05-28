"""
Convert custom PointNav dataset to GR00T LeRobot v2 format.

Raw structure:
  root/
    action/<goal_name>/<episode_id>.json
    review/labels.json                       # optional; rejected episodes are skipped
    rgb_single_x/<goal_name>/<episode_id>/rgb_0000.png, ...
    rgb_multiview_x/<goal_name>/<episode_id>/front_view/rgb_0000.png, ...
    rgb_multiview_x/<goal_name>/<episode_id>/left_view/rgb_0000.png, ...
    rgb_multiview_x/<goal_name>/<episode_id>/right_view/rgb_0000.png, ...

Output structure (GR00T LeRobot v2):
  output/
    meta/modality.json, episodes.jsonl, tasks.jsonl, info.json
    data/chunk-000/episode_XXXXXX.parquet
    videos/chunk-000/observation.images.ego_view/episode_XXXXXX.mp4
    videos/chunk-000/observation.images.left_view/episode_XXXXXX.mp4   # multiview
    videos/chunk-000/observation.images.right_view/episode_XXXXXX.mp4  # multiview
"""

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image


IMAGE_EXTS = {".png", ".jpg", ".jpeg"}
REJECTED_STATUS = "rejected"
DEFAULT_VIDEO_PREFIX = "observation.images"


# ─── Geometry helpers ────────────────────────────────────────────────────────

def world_to_robot(points_world: np.ndarray, robot_pose: dict) -> np.ndarray:
    """Transform (N, 2) world-frame XY points into the robot frame."""
    origin = np.array([robot_pose["x"], robot_pose["y"]])
    yaw = robot_pose["yaw"]
    c, s = math.cos(yaw), math.sin(yaw)
    R_inv = np.array([[c, s], [-s, c]])  # world → robot
    return (points_world - origin) @ R_inv.T


def compute_route(current_pose: dict, goal_pose: dict, n_segments: int = 10) -> list:
    """
    Build route: straight line from current position to goal, divided into n_segments.
    Each segment = (x_start, y_start, x_end, y_end) → 40 floats total.
    Matches COMPASS upsample_segments() approach (d_max=1.0m, linear interpolation).
    """
    start = np.array([current_pose["x"], current_pose["y"]], dtype=np.float64)
    goal = np.array([goal_pose["x"], goal_pose["y"]], dtype=np.float64)

    t = np.linspace(0, 1, n_segments + 1)
    waypoints = start + t[:, None] * (goal - start)  # (n_segments+1, 2)

    local = world_to_robot(waypoints, current_pose)

    route = []
    for i in range(n_segments):
        route.extend([local[i, 0], local[i, 1], local[i + 1, 0], local[i + 1, 1]])
    return route


def compute_goal_heading(current_pose: dict, goal_pose: dict) -> list:
    """Goal direction (cos θ, sin θ) expressed in the robot frame."""
    dx = goal_pose["x"] - current_pose["x"]
    dy = goal_pose["y"] - current_pose["y"]
    angle_world = math.atan2(dy, dx)
    angle_robot = angle_world - current_pose["yaw"]
    return [math.cos(angle_robot), math.sin(angle_robot)]


# ─── Video helper ─────────────────────────────────────────────────────────────

def images_to_mp4(img_paths: list, out_path: Path, fps: float) -> int:
    """Encode images to mp4 and return the actual number of encoded frames."""
    import av

    if not img_paths:
        raise ValueError(f"No images found for {out_path}")

    first = np.array(Image.open(img_paths[0]).convert("RGB"))
    h, w = first.shape[:2]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    container = av.open(str(out_path), mode="w")
    stream = container.add_stream("libx264", rate=round(fps))
    stream.width = w
    stream.height = h
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": "18", "preset": "fast"}

    for img_path in img_paths:
        frame_np = np.array(Image.open(img_path).convert("RGB"))
        frame = av.VideoFrame.from_ndarray(frame_np, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)

    for packet in stream.encode():
        container.mux(packet)
    container.close()

    # 실제 인코딩된 프레임 수를 읽어서 반환
    with av.open(str(out_path)) as verify:
        actual_frames = verify.streams.video[0].frames
    return actual_frames


def slugify(value: str) -> str:
    value = re.sub(r"[^0-9A-Za-z_]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "camera"


def normalize_episode_id(episode_id: str) -> str:
    return Path(str(episode_id)).stem


def load_review_labels(raw_root: Path) -> Dict[str, dict]:
    labels_path = raw_root / "review" / "labels.json"
    if not labels_path.is_file():
        return {}
    with open(labels_path, encoding="utf-8") as f:
        labels = json.load(f)

    if isinstance(labels, dict):
        return labels
    if isinstance(labels, list):
        normalized = {}
        for item in labels:
            if not isinstance(item, dict):
                continue
            goal_name = item.get("goal_name") or item.get("goal") or item.get("task")
            episode_id = item.get("episode_id") or item.get("episode") or item.get("id")
            if goal_name is None or episode_id is None:
                key = item.get("key") or item.get("episode_key")
            else:
                key = episode_label_key(str(goal_name), str(episode_id))
            if key:
                normalized[str(key)] = item
        return normalized
    return {}


def episode_label_key(goal_name: str, episode_id: str) -> str:
    return f"{goal_name}/{normalize_episode_id(episode_id)}"


def is_rejected(labels: Dict[str, dict], goal_name: str, episode_id: str) -> bool:
    label = review_label_for(labels, goal_name, episode_id)
    if isinstance(label, str):
        status = label
    elif isinstance(label, dict):
        status = label.get("status") or label.get("review") or label.get("label")
    else:
        status = None
    return str(status).strip().lower() == REJECTED_STATUS


def review_label_for(labels: Dict[str, dict], goal_name: str, episode_id: str):
    episode_id = normalize_episode_id(episode_id)
    possible_keys = (
        episode_label_key(goal_name, episode_id),
        f"{goal_name}/{episode_id}.json",
        episode_id,
        f"{episode_id}.json",
    )
    return next((labels[key] for key in possible_keys if key in labels), {})


def frame_number(path: Path) -> Optional[int]:
    match = re.search(r"(\d+)(?=\.[^.]+$)", path.name)
    return int(match.group(1)) if match else None


def sorted_image_files(path: Path) -> List[Path]:
    if not path.is_dir():
        return []
    return sorted(
        (p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS),
        key=lambda p: (frame_number(p) is None, frame_number(p) or 0, p.name),
    )


def camera_stream_order(stream_name: str) -> tuple:
    order = {
        "front_view": 0,
        "Replicator": 0,
        "left_view": 1,
        "Replicator_01": 1,
        "right_view": 2,
        "Replicator_02": 2,
    }
    return (order.get(stream_name, 99), stream_name)


def camera_key_for(rgb_folder: str, stream_name: Optional[str]) -> str:
    return f"{DEFAULT_VIDEO_PREFIX}.{camera_view_name(stream_name)}"


def camera_view_name(stream_name: Optional[str]) -> str:
    view_names = {
        None: "ego_view",
        "": "ego_view",
        "front_view": "ego_view",
        "Replicator": "ego_view",
        "left_view": "left_view",
        "Replicator_01": "left_view",
        "right_view": "right_view",
        "Replicator_02": "right_view",
    }
    return view_names.get(stream_name, slugify(stream_name or "ego_view"))


def discover_episode_camera_images(episode_rgb_dir: Path) -> Dict[str, List[Path]]:
    """Return {stream_name: image_paths} for one rgb folder episode dir."""
    direct_images = sorted_image_files(episode_rgb_dir)
    if direct_images:
        return {"": direct_images}

    streams: Dict[str, List[Path]] = {}
    for child in sorted((p for p in episode_rgb_dir.iterdir() if p.is_dir()), key=lambda p: camera_stream_order(p.name)):
        if child.name in {"front_view", "left_view", "right_view"}:
            images = sorted_image_files(child)
        elif child.name.startswith("Replicator"):
            images = sorted_image_files(child / "rgb")
        else:
            images = sorted_image_files(child)
        if images:
            streams[child.name] = images
    return streams


def list_rgb_folders(raw_root: Path, selected: Optional[List[str]] = None) -> List[str]:
    if selected:
        return [Path(folder).name for folder in selected]
    return sorted(p.name for p in raw_root.iterdir() if p.is_dir() and p.name.startswith("rgb"))


def existing_rgb_folders(raw_root: Path, selected: Optional[List[str]] = None) -> List[str]:
    rgb_folders = list_rgb_folders(raw_root, selected)
    missing = [folder for folder in rgb_folders if not (raw_root / folder).is_dir()]
    if missing:
        raise FileNotFoundError(f"RGB folder(s) not found under {raw_root}: {missing}")
    return rgb_folders


def rgb_folder_output_suffix(rgb_folder: str) -> str:
    return rgb_folder.removeprefix("rgb_")


def collect_camera_specs(raw_root: Path, raw_episodes: list, rgb_folders: List[str]) -> List[dict]:
    """Discover camera/video streams from the first episodes that have RGB."""
    specs_by_source: Dict[tuple, dict] = {}

    for goal_name, json_file in raw_episodes:
        with open(json_file, encoding="utf-8") as f:
            ep_data = json.load(f)
        episode_id = ep_data["episode_id"]

        for rgb_folder in rgb_folders:
            episode_rgb_dir = raw_root / rgb_folder / goal_name / episode_id
            if not episode_rgb_dir.is_dir():
                continue
            for stream_name, images in discover_episode_camera_images(episode_rgb_dir).items():
                stream_or_none = stream_name or None
                video_key = camera_key_for(rgb_folder, stream_or_none)
                source_key = (rgb_folder, stream_or_none)
                if source_key in specs_by_source:
                    continue
                image_shape = list(np.array(Image.open(images[0]).convert("RGB")).shape)
                specs_by_source[source_key] = {
                    "rgb_folder": rgb_folder,
                    "stream": stream_or_none,
                    "video_key": video_key,
                    "modality_name": camera_view_name(stream_or_none),
                    "image_shape": image_shape,
                }

    specs = sorted(
        specs_by_source.values(),
        key=lambda spec: (spec["rgb_folder"], camera_stream_order(spec["stream"] or "")),
    )
    video_key_sources: Dict[str, List[str]] = {}
    for spec in specs:
        source = spec["rgb_folder"] if spec["stream"] is None else f"{spec['rgb_folder']}/{spec['stream']}"
        video_key_sources.setdefault(spec["video_key"], []).append(source)
    duplicates = {key: sources for key, sources in video_key_sources.items() if len(sources) > 1}
    if duplicates:
        details = ", ".join(f"{key}: {sources}" for key, sources in duplicates.items())
        raise ValueError(
            "Multiple RGB sources map to the same LeRobot video key. "
            "Use --split-rgb-folders or select one --rgb-folder at a time. "
            f"Duplicates: {details}"
        )
    return specs


def episode_images_for_spec(raw_root: Path, goal_name: str, episode_id: str, spec: dict) -> List[Path]:
    episode_rgb_dir = raw_root / spec["rgb_folder"] / goal_name / episode_id
    if not episode_rgb_dir.is_dir():
        return []
    streams = discover_episode_camera_images(episode_rgb_dir)
    stream_key = spec["stream"] or ""
    return streams.get(stream_key, [])


def create_modality_json(modality_json_src: str, camera_specs: List[dict], out_path: Path) -> None:
    with open(modality_json_src, encoding="utf-8") as f:
        modality = json.load(f)

    modality["video"] = {
        spec["modality_name"]: {"original_key": spec["video_key"]}
        for spec in camera_specs
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(modality, f, indent=2)


# ─── Main conversion ──────────────────────────────────────────────────────────

def collect_episodes(action_root: Path) -> list:
    """Return sorted list of (goal_name, json_path)."""
    episodes = []
    for goal_dir in sorted(action_root.iterdir()):
        if not goal_dir.is_dir():
            continue
        for json_file in sorted(goal_dir.glob("*.json")):
            episodes.append((goal_dir.name, json_file))
    return episodes


def convert(
    raw_root: str,
    output_root: str,
    modality_json_src: str,
    rgb_folders: Optional[List[str]] = None,
    include_rejected: bool = False,
) -> None:
    import pandas as pd

    raw_root = Path(raw_root)
    output_root = Path(output_root)

    action_root = raw_root / "action"

    data_dir = output_root / "data" / "chunk-000"
    meta_dir = output_root / "meta"
    for d in (data_dir, meta_dir):
        d.mkdir(parents=True, exist_ok=True)

    all_tasks: dict = {}
    episodes_meta = []
    global_index = 0
    dataset_fps = None

    review_labels = load_review_labels(raw_root)
    all_raw_episodes = collect_episodes(action_root)
    raw_episodes = []
    skipped_rejected = 0
    for goal_name, json_file in all_raw_episodes:
        with open(json_file, encoding="utf-8") as f:
            ep_data = json.load(f)
        episode_id = ep_data.get("episode_id", json_file.stem)
        if not include_rejected and is_rejected(review_labels, goal_name, episode_id):
            skipped_rejected += 1
            continue
        raw_episodes.append((goal_name, json_file))

    selected_rgb_folders = existing_rgb_folders(raw_root, rgb_folders)
    camera_specs = collect_camera_specs(raw_root, raw_episodes, selected_rgb_folders)
    if not camera_specs:
        raise FileNotFoundError(
            f"No RGB camera streams found under {raw_root}. "
            f"Looked for folders: {selected_rgb_folders}"
        )

    for spec in camera_specs:
        (output_root / "videos" / "chunk-000" / spec["video_key"]).mkdir(parents=True, exist_ok=True)

    print(f"Found {len(all_raw_episodes)} episodes.")
    print(f"Using {len(raw_episodes)} episodes. Skipped rejected: {skipped_rejected}.")
    print("Camera streams:")
    for spec in camera_specs:
        stream = spec["stream"] or "single"
        print(f"  - {spec['video_key']} ({spec['rgb_folder']}/{stream}) shape={spec['image_shape']}")

    converted_ep_idx = 0
    for source_idx, (goal_name, json_file) in enumerate(raw_episodes):
        with open(json_file, encoding="utf-8") as f:
            ep_data = json.load(f)

        trajectory = ep_data["trajectory"]
        if not trajectory:
            print(f"  [SKIP] {goal_name}/{ep_data.get('episode_id', json_file.stem)} empty trajectory")
            continue

        goal_pose = ep_data["goal_pose"]
        raw_lang = ep_data["language_instruction"]
        goal_name_from_lang = raw_lang.removeprefix("goal_")
        language = f"Go to {goal_name_from_lang}"
        episode_id = ep_data["episode_id"]

        if language not in all_tasks:
            all_tasks[language] = len(all_tasks)
        if "valid" not in all_tasks:
            all_tasks["valid"] = len(all_tasks)

        lang_idx = all_tasks[language]
        valid_idx = all_tasks["valid"]

        # Infer FPS from first episode with enough frames
        if dataset_fps is None and len(trajectory) > 1:
            dt = trajectory[1]["stamp_unix"] - trajectory[0]["stamp_unix"]
            dataset_fps = 1.0 / dt if dt > 0 else 10.0

        t0 = trajectory[0]["stamp_unix"]
        images_by_key = {
            spec["video_key"]: episode_images_for_spec(raw_root, goal_name, episode_id, spec)
            for spec in camera_specs
        }
        missing = [key for key, paths in images_by_key.items() if not paths]
        if missing:
            print(f"  [SKIP] {goal_name}/{episode_id}: missing RGB for {missing}")
            continue

        rows = []
        for step in trajectory:
            map_pose = step.get("map_pose") or step.get("sim_pose")
            cmd = step["cmd_vel"]
            odom = step["odometry"]

            if map_pose is None or cmd is None or odom is None:
                continue
            if cmd.get("linear") is None or cmd.get("angular") is None:
                continue

            speed = odom["twist"]["linear"]["x"]
            route = compute_route(map_pose, goal_pose)
            goal_heading = compute_goal_heading(map_pose, goal_pose)

            state = np.array([speed] + route + goal_heading, dtype=np.float32)
            assert state.shape == (43,), f"State shape mismatch: {state.shape}"

            action = np.array([
                cmd["linear"]["x"],
                cmd["linear"]["y"],
                cmd["angular"]["z"],
            ], dtype=np.float32)

            rows.append({
                "observation.state": state.tolist(),
                "action": action.tolist(),
                "timestamp": float(step["stamp_unix"] - t0),
                "annotation.human.action.task_description": lang_idx,
                "annotation.human.validity": valid_idx,
                "task_index": lang_idx,
                "episode_index": converted_ep_idx,
                "index": global_index,
            })
            global_index += 1

        if not rows:
            print(f"  [SKIP] {goal_name}/{episode_id}: no valid rows")
            continue

        rows_before_trim = len(rows)
        max_frames = min([len(rows)] + [len(paths) for paths in images_by_key.values()])
        if max_frames != len(rows):
            print(
                f"  [WARN] {goal_name}/{episode_id}: rows={len(rows)}, "
                f"camera_frames={[len(paths) for paths in images_by_key.values()]} — trimming to {max_frames}"
            )
            rows = rows[:max_frames]

        actual_frame_counts = []
        for spec in camera_specs:
            video_key = spec["video_key"]
            img_files = images_by_key[video_key][:max_frames]
            out_video = output_root / "videos" / "chunk-000" / video_key / f"episode_{converted_ep_idx:06d}.mp4"
            actual_frames = images_to_mp4(img_files, out_video, fps=dataset_fps or 10.0)
            actual_frame_counts.append(actual_frames)

        min_actual_frames = min(actual_frame_counts)
        if min_actual_frames != len(rows):
            print(
                f"  [WARN] {goal_name}/{episode_id}: encoded frames={actual_frame_counts} "
                f"but rows={len(rows)} — trimming parquet to {min_actual_frames}"
            )
            rows = rows[:min_actual_frames]

        # global_index 재계산 (trim된 경우 보정)
        global_index -= rows_before_trim - len(rows)

        df = pd.DataFrame(rows)
        df.to_parquet(data_dir / f"episode_{converted_ep_idx:06d}.parquet", index=False)

        label = review_label_for(review_labels, goal_name, episode_id)

        episodes_meta.append({
            "episode_index": converted_ep_idx,
            "tasks": [language, "valid"],
            "length": len(rows),
            "source_goal": goal_name,
            "source_episode_id": episode_id,
            "review_label": label,
        })
        converted_ep_idx += 1
        print(f"  [{source_idx + 1}/{len(raw_episodes)}] {goal_name}/{episode_id} — {len(rows)} steps")

    # ── meta files ───────────────────────────────────────────────────────────

    with open(meta_dir / "tasks.jsonl", "w", encoding="utf-8") as f:
        for task, idx in sorted(all_tasks.items(), key=lambda x: x[1]):
            f.write(json.dumps({"task_index": idx, "task": task}) + "\n")

    with open(meta_dir / "episodes.jsonl", "w", encoding="utf-8") as f:
        for ep in episodes_meta:
            f.write(json.dumps(ep) + "\n")

    create_modality_json(modality_json_src, camera_specs, meta_dir / "modality.json")

    total_frames = sum(ep["length"] for ep in episodes_meta)
    fps_rounded = round(dataset_fps or 10.0, 6)
    video_features = {}
    for spec in camera_specs:
        video_features[spec["video_key"]] = {
            "dtype": "video",
            "shape": spec["image_shape"],
            "names": ["height", "width", "channel"],
            "video_info": {
                "video.fps": fps_rounded,
                "video.codec": "h264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "has_audio": False,
            },
        }

    info = {
        "codebase_version": "v2.0",
        "robot_type": "nova_carter",
        "total_episodes": len(episodes_meta),
        "total_frames": total_frames,
        "total_tasks": len(all_tasks),
        "total_videos": len(episodes_meta) * len(camera_specs),
        "total_chunks": 1,
        "chunks_size": len(episodes_meta),
        "fps": fps_rounded,
        "splits": {"train": f"0:{len(episodes_meta)}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "observation.state": {
                "dtype": "float64",
                "shape": [43],
                "names": [f"state_{i}" for i in range(43)],
            },
            "action": {
                "dtype": "float64",
                "shape": [3],
                "names": [f"action_{i}" for i in range(3)],
            },
            "timestamp": {"dtype": "float64", "shape": [1]},
            "annotation.human.action.task_description": {"dtype": "int64", "shape": [1]},
            "task_index": {"dtype": "int64", "shape": [1]},
            "annotation.human.validity": {"dtype": "int64", "shape": [1]},
            "episode_index": {"dtype": "int64", "shape": [1]},
            "index": {"dtype": "int64", "shape": [1]},
            **video_features,
        },
    }
    with open(meta_dir / "info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)

    print(f"\nDone. {len(episodes_meta)} episodes / {total_frames} frames → {output_root}")


def convert_split_by_rgb_folder(
    raw_root: str,
    output_root: str,
    modality_json_src: str,
    rgb_folders: Optional[List[str]] = None,
    include_rejected: bool = False,
) -> None:
    raw_root_path = Path(raw_root)
    output_root_path = Path(output_root)
    selected_rgb_folders = existing_rgb_folders(raw_root_path, rgb_folders)

    print(f"Converting {len(selected_rgb_folders)} RGB folder(s) into separate LeRobot datasets.")
    for rgb_folder in selected_rgb_folders:
        folder_output_root = output_root_path.parent / f"{output_root_path.name}_{rgb_folder_output_suffix(rgb_folder)}"
        print(f"\n=== {rgb_folder} → {folder_output_root} ===")
        convert(
            str(raw_root_path),
            str(folder_output_root),
            modality_json_src,
            rgb_folders=[rgb_folder],
            include_rejected=include_rejected,
        )


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert custom raw data to GR00T LeRobot v2 format.")
    parser.add_argument("--raw-root", required=True, help="Root dir containing action/ and rgb/")
    parser.add_argument("--output-root", required=True, help="Output directory for converted dataset")
    parser.add_argument(
        "--modality-json",
        default="examples/PointNav/modality.json",
        help="Path to modality.json (default: examples/PointNav/modality.json)",
    )
    parser.add_argument(
        "--rgb-folder",
        action="append",
        dest="rgb_folders",
        help="RGB folder to include. Can be passed multiple times. Default: all rgb* folders.",
    )
    parser.add_argument(
        "--include-rejected",
        action="store_true",
        help="Include episodes marked rejected in review/labels.json. Default: skip rejected.",
    )
    parser.add_argument(
        "--split-rgb-folders",
        action="store_true",
        help=(
            "Create one LeRobot dataset per RGB folder as OUTPUT_ROOT_<rgb-folder-suffix>. "
            "Useful for converting each camera preset separately while preserving "
            "multiview folders as one dataset with multiple video streams."
        ),
    )
    args = parser.parse_args()
    if args.split_rgb_folders:
        convert_split_by_rgb_folder(
            args.raw_root,
            args.output_root,
            args.modality_json,
            rgb_folders=args.rgb_folders,
            include_rejected=args.include_rejected,
        )
    else:
        convert(
            args.raw_root,
            args.output_root,
            args.modality_json,
            rgb_folders=args.rgb_folders,
            include_rejected=args.include_rejected,
        )


'''
# Step 1: Convert
uv run python sujinkim/to_lerobot_v2.py \
    --raw-root /nas/sujinkim/data/goto/real_v2_edited \
    --output-root /nas/sujinkim/data/goto/real_v2_lerobot_edited

# Convert each RGB folder separately:
#   /nas/sujinkim/data/goto/sim_v2_lerobot_single_gemini336l
#   /nas/sujinkim/data/goto/sim_v2_lerobot_single_gemini345lg
#   /nas/sujinkim/data/goto/sim_v2_lerobot_multiview_gemini_336
uv run python sujinkim/to_lerobot_v2.py \
    --raw-root /nas/sujinkim/data/goto/sim_v2 \
    --output-root /nas/sujinkim/data/goto/sim_v2_lerobot \
    --split-rgb-folders

# Step 2: Generate stats.json
uv run python gr00t/data/stats.py \
    --dataset-path /nas/sujinkim/data/goto/sim_v2_lerobot_single_gemini345lg \
    --embodiment-tag NEW_EMBODIMENT
    
'''
