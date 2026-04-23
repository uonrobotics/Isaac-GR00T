"""
LeRobot v2 dataset visualizer.

키보드로 에피소드/프레임 탐색:
  ← → : 프레임 이동
  a d  : 에피소드 이동
  q    : 종료

Usage:
  uv run python sujinkim/visualize_dataset.py --dataset-path /path/to/dataset
  uv run python sujinkim/visualize_dataset.py --dataset-path /path/to/dataset --episode 5
  
  uv run python sujinkim/visualize_dataset.py \
  --dataset-path /nas/sujinkim/data/goto/sim/20260323_lerobot_v2 \
  --episode 1
"""

import argparse
import json
from pathlib import Path

import av
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd


def load_video_frames(video_path: Path) -> list:
    frames = []
    with av.open(str(video_path)) as container:
        for frame in container.decode(video=0):
            frames.append(frame.to_ndarray(format="rgb24"))
    return frames


def load_episode(root: Path, ep_idx: int):
    parquet_path = root / "data" / "chunk-000" / f"episode_{ep_idx:06d}.parquet"
    video_path = root / "videos" / "chunk-000" / "observation.images.ego_view" / f"episode_{ep_idx:06d}.mp4"

    df = pd.read_parquet(parquet_path)
    frames = load_video_frames(video_path)
    return df, frames


class Visualizer:
    def __init__(self, dataset_path: str, start_ep: int = 0):
        self.root = Path(dataset_path)

        with open(self.root / "meta" / "episodes.jsonl") as f:
            self.episodes = [json.loads(l) for l in f]
        with open(self.root / "meta" / "tasks.jsonl") as f:
            self.tasks = {t["task_index"]: t["task"] for t in (json.loads(l) for l in f)}

        self.ep_idx = start_ep
        self.frame_idx = 0
        self.df = None
        self.frames = None
        self._load_episode()
        self._build_fig()

    def _load_episode(self):
        ep = self.episodes[self.ep_idx]
        print(f"Loading episode {self.ep_idx} (length={ep['length']})...")
        self.df, self.frames = load_episode(self.root, ep["episode_index"])
        self.frame_idx = 0
        # mismatch 즉시 표시
        if len(self.frames) != len(self.df):
            print(f"  [WARN] video={len(self.frames)} frames, parquet={len(self.df)} rows")

    def _build_fig(self):
        self.fig = plt.figure(figsize=(18, 8))
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        gs = gridspec.GridSpec(2, 4, figure=self.fig)

        self.ax_img     = self.fig.add_subplot(gs[:, 0])   # 비디오 프레임
        self.ax_act     = self.fig.add_subplot(gs[0, 1])   # action 시계열
        self.ax_state   = self.fig.add_subplot(gs[0, 2])   # speed
        self.ax_heading = self.fig.add_subplot(gs[0, 3])   # goal heading
        self.ax_route   = self.fig.add_subplot(gs[1, 1])   # route BEV
        self.ax_valid   = self.fig.add_subplot(gs[1, 2])   # validity 시계열
        self.ax_info    = self.fig.add_subplot(gs[1, 3])   # 텍스트 정보

        self.ax_img.axis("off")
        self.ax_heading.set_aspect("equal")
        self.ax_info.axis("off")
        self._draw()
        plt.tight_layout()
        plt.show()

    def _draw(self):
        ep_meta = self.episodes[self.ep_idx]
        n_frames = len(self.frames)
        n_rows = len(self.df)
        t = self.frame_idx

        # ── 비디오 프레임
        self.ax_img.cla()
        self.ax_img.axis("off")
        if t < n_frames:
            self.ax_img.imshow(self.frames[t])
        else:
            self.ax_img.text(0.5, 0.5, "No frame", ha="center", va="center", transform=self.ax_img.transAxes)
        self.ax_img.set_title(f"Frame {t}/{n_frames-1}  (parquet rows: {n_rows})", fontsize=9)

        # ── Action 시계열 (전체 + 현재 위치)
        self.ax_act.cla()
        actions = np.array(self.df["action"].tolist())
        for i, label in enumerate(["linear_x", "linear_y", "angular_z"]):
            self.ax_act.plot(actions[:, i], label=label, alpha=0.7)
        if t < n_rows:
            self.ax_act.axvline(t, color="red", linewidth=1)
        self.ax_act.set_title("Action", fontsize=9)
        self.ax_act.legend(fontsize=7)
        self.ax_act.grid(True, alpha=0.3)

        # ── Speed (state[0])
        self.ax_state.cla()
        states = np.array(self.df["observation.state"].tolist())
        self.ax_state.plot(states[:, 0], label="speed", color="purple")
        if t < n_rows:
            self.ax_state.axvline(t, color="red", linewidth=1)
        self.ax_state.set_title("Speed (state[0])", fontsize=9)
        self.ax_state.grid(True, alpha=0.3)

        # ── Goal Heading (state[41:43]) — BEV 화살표
        # (cos θ, sin θ) in robot frame → BEV: plot_x=-sin θ, plot_y=cos θ
        self.ax_heading.cla()
        self.ax_heading.set_xlim(-1.3, 1.3)
        self.ax_heading.set_ylim(-1.3, 1.3)
        self.ax_heading.set_aspect("equal")
        self.ax_heading.grid(True, alpha=0.3)
        if t < n_rows:
            cos_t, sin_t = states[t, 41], states[t, 42]
            self.ax_heading.annotate("", xy=(-sin_t, cos_t), xytext=(0, 0),
                                     arrowprops=dict(arrowstyle="->", color="green", lw=2))
            self.ax_heading.plot(0, 0, "r*", markersize=10)
        self.ax_heading.set_title("Goal Heading (BEV: ↑forward)", fontsize=9)
        self.ax_heading.set_xlabel("← left  |  right →", fontsize=7)
        self.ax_heading.set_ylabel("↑ forward", fontsize=7)

        # ── Route BEV (위=전방, 왼쪽=로봇왼쪽)
        # robot_X=forward → plot_y, robot_Y=left → plot_x 부호 반전
        self.ax_route.cla()
        if t < n_rows:
            route = np.array(self.df["observation.state"].iloc[t][1:41]).reshape(10, 4)
            for seg in route:
                rx0, ry0, rx1, ry1 = seg
                self.ax_route.plot([-ry0, -ry1], [rx0, rx1], "b-o", markersize=3)
            # goal heading 화살표도 route 위에 오버레이
            cos_t, sin_t = states[t, 41], states[t, 42]
            self.ax_route.annotate("", xy=(-sin_t * 0.5, cos_t * 0.5), xytext=(0, 0),
                                   arrowprops=dict(arrowstyle="->", color="green", lw=1.5))
            self.ax_route.plot(0, 0, "r*", markersize=10, label="robot")
        self.ax_route.set_title("Route + Heading (BEV: ↑forward, ←left)", fontsize=9)
        self.ax_route.set_xlabel("← left  |  right →", fontsize=7)
        self.ax_route.set_ylabel("↑ forward", fontsize=7)
        self.ax_route.set_aspect("equal")
        self.ax_route.grid(True, alpha=0.3)
        self.ax_route.legend(fontsize=7)

        # ── Validity 시계열
        self.ax_valid.cla()
        validity = self.df["annotation.human.validity"].tolist()
        validity_idx = int(self.df["annotation.human.validity"].iloc[min(t, n_rows-1)])
        validity_str = self.tasks.get(validity_idx, "?")
        self.ax_valid.plot(validity, color="teal", linewidth=1)
        if t < n_rows:
            self.ax_valid.axvline(t, color="red", linewidth=1)
        self.ax_valid.set_title(f"Validity (cur: {validity_str})", fontsize=9)
        self.ax_valid.set_yticks(sorted(set(validity)))
        self.ax_valid.set_yticklabels([self.tasks.get(v, str(v)) for v in sorted(set(validity))], fontsize=7)
        self.ax_valid.grid(True, alpha=0.3)

        # ── 텍스트 정보
        self.ax_info.cla()
        self.ax_info.axis("off")
        task_idx = int(self.df["task_index"].iloc[min(t, n_rows-1)])
        task_str = self.tasks.get(task_idx, "?")
        mismatch = f"  ⚠ video({n_frames}) ≠ parquet({n_rows})" if n_frames != n_rows else ""
        info = (
            f"Episode: {self.ep_idx}/{len(self.episodes)-1}{mismatch}\n"
            f"Task: {task_str}\n"
            f"Video frames: {n_frames}\n"
            f"Parquet rows: {n_rows}\n\n"
            f"[← →] frame\n[a d] episode\n[q] quit"
        )
        self.ax_info.text(0.05, 0.95, info, va="top", ha="left",
                          transform=self.ax_info.transAxes, fontsize=9,
                          family="monospace",
                          color="red" if mismatch else "black")

        self.fig.canvas.draw_idle()

    def _on_key(self, event):
        n_frames = len(self.frames)
        n_rows = len(self.df)
        max_t = max(n_frames, n_rows) - 1

        if event.key == "right":
            self.frame_idx = min(self.frame_idx + 1, max_t)
        elif event.key == "left":
            self.frame_idx = max(self.frame_idx - 1, 0)
        elif event.key == "d":
            self.ep_idx = min(self.ep_idx + 1, len(self.episodes) - 1)
            self._load_episode()
        elif event.key == "a":
            self.ep_idx = max(self.ep_idx - 1, 0)
            self._load_episode()
        elif event.key == "q":
            plt.close("all")
            return
        self._draw()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--episode", type=int, default=0)
    args = parser.parse_args()
    Visualizer(args.dataset_path, start_ep=args.episode)
