"""
Convert custom PointNav dataset to GR00T LeRobot v2 format.

Raw structure:
  root/
    action/<goal_name>/<episode_id>.json
    rgb/<goal_name>/<episode_id>/0000.png, 0001.png, ...

Output structure (GR00T LeRobot v2):
  output/
    meta/modality.json, episodes.jsonl, tasks.jsonl, info.json
    data/chunk-000/episode_XXXXXX.parquet
    videos/chunk-000/observation.images.ego_view/episode_XXXXXX.mp4
"""

import argparse
import json
import math
import shutil
from pathlib import Path

import av
import numpy as np
import pandas as pd
from PIL import Image


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


def convert(raw_root: str, output_root: str, modality_json_src: str) -> None:
    raw_root = Path(raw_root)
    output_root = Path(output_root)

    action_root = raw_root / "action"
    rgb_root = raw_root / "rgb"

    data_dir = output_root / "data" / "chunk-000"
    video_dir = output_root / "videos" / "chunk-000" / "observation.images.ego_view"
    meta_dir = output_root / "meta"
    for d in (data_dir, video_dir, meta_dir):
        d.mkdir(parents=True, exist_ok=True)

    all_tasks: dict = {}
    episodes_meta = []
    global_index = 0
    dataset_fps = None

    raw_episodes = collect_episodes(action_root)
    print(f"Found {len(raw_episodes)} episodes.")

    # Infer image shape from first available image
    image_shape = [320, 512, 3]  # fallback
    for _goal, _json in raw_episodes:
        with open(_json) as _f:
            _ep = json.load(_f)
        _img_dir = rgb_root / _goal / _ep["episode_id"]
        _imgs = sorted(_img_dir.glob("*.png"))
        if _imgs:
            image_shape = list(np.array(Image.open(_imgs[0]).convert("RGB")).shape)
            break

    for ep_idx, (goal_name, json_file) in enumerate(raw_episodes):
        with open(json_file) as f:
            ep_data = json.load(f)

        trajectory = ep_data["trajectory"]
        goal_pose = ep_data["goal_pose"]
        raw_lang = ep_data["language_instruction"]
        goal_name_from_lang = raw_lang.removeprefix("goal_")
        language = f"Go to {goal_name_from_lang}"

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
        img_dir = rgb_root / goal_name / ep_data["episode_id"]
        all_img_files = sorted(img_dir.glob("*.png"))

        rows = []
        valid_img_files = []
        for step_i, step in enumerate(trajectory):
            map_pose = step["map_pose"]
            cmd = step["cmd_vel"]
            odom = step["odometry"]

            if cmd is None or odom is None:
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
                "episode_index": ep_idx,
                "index": global_index,
            })
            if step_i < len(all_img_files):
                valid_img_files.append(all_img_files[step_i])
            global_index += 1

        img_files = valid_img_files
        if len(img_files) != len(rows):
            print(f"  [WARN] ep {ep_idx}: {len(img_files)} images vs {len(rows)} steps — trimming to shorter")
            min_len = min(len(img_files), len(rows))
            img_files = img_files[:min_len]
            rows = rows[:min_len]

        out_video = video_dir / f"episode_{ep_idx:06d}.mp4"
        actual_frames = images_to_mp4(img_files, out_video, fps=dataset_fps or 10.0)

        # 인코딩 후 실제 프레임 수와 parquet row 수가 다르면 parquet을 trim
        if actual_frames != len(rows):
            print(f"  [WARN] ep {ep_idx}: video has {actual_frames} frames but {len(rows)} rows — trimming parquet to {actual_frames}")
            rows = rows[:actual_frames]

        # global_index 재계산 (trim된 경우 보정)
        trimmed = len(valid_img_files) - len(rows)
        global_index -= trimmed

        df = pd.DataFrame(rows)
        df.to_parquet(data_dir / f"episode_{ep_idx:06d}.parquet", index=False)

        episodes_meta.append({
            "episode_index": ep_idx,
            "tasks": [language, "valid"],
            "length": len(rows),
        })
        print(f"  [{ep_idx + 1}/{len(raw_episodes)}] {goal_name}/{ep_data['episode_id']} — {len(rows)} steps")

    # ── meta files ───────────────────────────────────────────────────────────

    with open(meta_dir / "tasks.jsonl", "w") as f:
        for task, idx in sorted(all_tasks.items(), key=lambda x: x[1]):
            f.write(json.dumps({"task_index": idx, "task": task}) + "\n")

    with open(meta_dir / "episodes.jsonl", "w") as f:
        for ep in episodes_meta:
            f.write(json.dumps(ep) + "\n")

    shutil.copy(modality_json_src, meta_dir / "modality.json")

    total_frames = sum(ep["length"] for ep in episodes_meta)
    fps_rounded = round(dataset_fps or 10.0, 6)

    info = {
        "codebase_version": "v2.0",
        "robot_type": "nova_carter",
        "total_episodes": len(episodes_meta),
        "total_frames": total_frames,
        "total_tasks": len(all_tasks),
        "total_videos": len(episodes_meta),
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
            "observation.images.ego_view": {
                "dtype": "video",
                "shape": image_shape,
                "names": ["height", "width", "channel"],
                "video_info": {
                    "video.fps": fps_rounded,
                    "video.codec": "h264",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "has_audio": False,
                },
            },
        },
    }
    with open(meta_dir / "info.json", "w") as f:
        json.dump(info, f, indent=2)

    print(f"\nDone. {len(episodes_meta)} episodes / {total_frames} frames → {output_root}")


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
    args = parser.parse_args()
    convert(args.raw_root, args.output_root, args.modality_json)


'''
# Step 1: Convert
uv run python sujinkim/to_lerobot_v2.py \
    --raw-root /nas/sujinkim/data/goto/sim/20260323/ \
    --output-root /nas/sujinkim/data/goto/sim/20260323_lerobot_v2

# Step 2: Generate stats.json
uv run python gr00t/data/stats.py \
    --dataset-path /nas/sujinkim/data/goto/sim/20260323_lerobot_v2 \
    --embodiment-tag NEW_EMBODIMENT
    
'''