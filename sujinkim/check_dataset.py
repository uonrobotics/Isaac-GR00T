"""
LeRobot v2 dataset integrity checker.

Checks every episode for:
  - video frame count vs parquet row count
  - parquet row count vs episodes.jsonl length field
  - empty episodes
  - missing video or parquet files

Usage:
  uv run python sujinkim/check_dataset.py --dataset-path /path/to/dataset
  
  uv run python sujinkim/check_dataset.py \
  --dataset-path /nas/sujinkim/data/goto/sim/20260323_lerobot_v2
"""

import argparse
import json
from pathlib import Path

import av
import pandas as pd


def get_video_frame_count(video_path: Path) -> int:
    with av.open(str(video_path)) as container:
        return container.streams.video[0].frames


def check_dataset(dataset_path: str) -> None:
    root = Path(dataset_path)
    meta_dir = root / "meta"
    data_dir = root / "data" / "chunk-000"
    video_dir = root / "videos" / "chunk-000" / "observation.images.ego_view"

    # episodes.jsonl 로드
    episodes = []
    with open(meta_dir / "episodes.jsonl") as f:
        for line in f:
            episodes.append(json.loads(line))

    print(f"Total episodes in meta: {len(episodes)}\n")

    issues = []

    for ep in episodes:
        ep_idx = ep["episode_index"]
        meta_length = ep["length"]

        parquet_path = data_dir / f"episode_{ep_idx:06d}.parquet"
        video_path = video_dir / f"episode_{ep_idx:06d}.mp4"

        row = {"ep": ep_idx, "meta_len": meta_length, "parquet_len": None, "video_frames": None, "issues": []}

        # 파일 존재 여부
        if not parquet_path.exists():
            row["issues"].append("MISSING parquet")
        else:
            df = pd.read_parquet(parquet_path)
            row["parquet_len"] = len(df)
            if len(df) == 0:
                row["issues"].append("EMPTY parquet")
            if len(df) != meta_length:
                row["issues"].append(f"meta_length({meta_length}) != parquet_rows({len(df)})")

        if not video_path.exists():
            row["issues"].append("MISSING video")
        else:
            frames = get_video_frame_count(video_path)
            row["video_frames"] = frames
            if row["parquet_len"] is not None and frames != row["parquet_len"]:
                row["issues"].append(f"video_frames({frames}) != parquet_rows({row['parquet_len']})")

        if row["issues"]:
            issues.append(row)
            status = "  [FAIL]"
        else:
            status = "  [ OK ]"

        print(f"{status} ep {ep_idx:06d} | meta={meta_length} parquet={row['parquet_len']} video={row['video_frames']}"
              + (f" → {', '.join(row['issues'])}" if row['issues'] else ""))

    print(f"\n{'='*60}")
    print(f"Result: {len(issues)} / {len(episodes)} episodes have issues")

    if issues:
        print("\nProblematic episodes:")
        for row in issues:
            print(f"  ep {row['ep']:06d}: {', '.join(row['issues'])}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", required=True)
    args = parser.parse_args()
    check_dataset(args.dataset_path)
