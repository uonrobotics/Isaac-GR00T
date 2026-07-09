#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Concatenate left_view and right_view images side by side into ego_view."
    )
    parser.add_argument("src_root", type=Path)
    parser.add_argument("dst_root", type=Path)
    parser.add_argument("--left-name", default="left_view")
    parser.add_argument("--right-name", default="right_view")
    parser.add_argument("--out-name", default="ego_view")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output images.",
    )
    return parser.parse_args()


def image_files(view_dir: Path) -> list[Path]:
    return sorted(
        path for path in view_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def copy_episode_sidecars(src_episode: Path, dst_episode: Path) -> None:
    dst_episode.mkdir(parents=True, exist_ok=True)
    for path in src_episode.iterdir():
        if path.is_file():
            dst_path = dst_episode / path.name
            if not dst_path.exists():
                shutil.copy2(path, dst_path)


def concat_pair(left_path: Path, right_path: Path, dst_path: Path, overwrite: bool) -> None:
    if dst_path.exists() and not overwrite:
        return

    with Image.open(left_path) as left_img, Image.open(right_path) as right_img:
        left = left_img.convert("RGB")
        right = right_img.convert("RGB")

        if left.height != right.height:
            new_width = round(right.width * (left.height / right.height))
            right = right.resize((new_width, left.height), Image.Resampling.LANCZOS)

        canvas = Image.new("RGB", (left.width + right.width, left.height))
        canvas.paste(left, (0, 0))
        canvas.paste(right, (left.width, 0))

        dst_path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(dst_path)


def main() -> int:
    args = parse_args()
    src_root = args.src_root.resolve()
    dst_root = args.dst_root.resolve()

    if not src_root.exists():
        raise FileNotFoundError(f"Source root does not exist: {src_root}")

    tasks: list[tuple[Path, Path, Path]] = []
    missing_right: list[Path] = []

    for left_dir in sorted(src_root.rglob(args.left_name)):
        if not left_dir.is_dir():
            continue
        episode_dir = left_dir.parent
        right_dir = episode_dir / args.right_name
        if not right_dir.is_dir():
            missing_right.append(episode_dir)
            continue

        rel_episode = episode_dir.relative_to(src_root)
        dst_episode = dst_root / rel_episode
        copy_episode_sidecars(episode_dir, dst_episode)

        left_files = image_files(left_dir)
        right_by_name = {path.name: path for path in image_files(right_dir)}
        for left_path in left_files:
            right_path = right_by_name.get(left_path.name)
            if right_path is None:
                missing_right.append(left_path)
                continue
            dst_path = dst_episode / args.out_name / left_path.name
            tasks.append((left_path, right_path, dst_path))

    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(concat_pair, left_path, right_path, dst_path, args.overwrite)
            for left_path, right_path, dst_path in tasks
        ]
        for future in as_completed(futures):
            future.result()
            done += 1
            if done % 1000 == 0:
                print(f"processed {done}/{len(tasks)}")

    print(f"episodes with left/right pairs: {len({path.parent.parent for _, _, path in tasks})}")
    print(f"images processed or already present: {done}")
    print(f"missing right-side matches: {len(missing_right)}")
    print(f"output root: {dst_root}")
    return 0 if not missing_right else 1


if __name__ == "__main__":
    raise SystemExit(main())
