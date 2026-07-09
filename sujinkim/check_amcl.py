"""
Run AMCL trajectory analysis for every collected episode.

Input JSON structure:
  {action_dir}/{goal_name}/{episode_id}.json
  trajectory[*].stamp_ros_sec or stamp_unix
  trajectory[*].map_pose.{x,y,yaw}

Outputs:
  {out_dir}/{goal_name}/{episode_id}.png
  {out_dir}/amcl_events.csv

Usage:
  uv run python check_amcl.py
  uv run python check_amcl.py \
    --action-dir /nas/sujinkim/data/following_lane/real_v1/action \
    --out-dir amcl_report \
    --max-speed 1.5 \
    --max-yaw-rate 2.0
"""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_ACTION_DIR = Path("/nas/sujinkim/data/following_lane/real_v1/action")
DEFAULT_OUT_DIR = Path("amcl_report")


def analyze_amcl_trajectory(
    data,
    time_col="t",
    x_col="x",
    y_col="y",
    yaw_col="yaw",
    output_png="amcl_trajectory_analysis.png",
    max_speed=None,
    max_yaw_rate=None,
    mad_threshold=6.0,
    score_threshold=1.0,
    speed_weight=1.0,
    yaw_rate_weight=1.0,
    curvature_weight=1.0,
    min_curvature_dpos=0.03,
):
    """
    AMCL trajectory jump/anomaly 분석 후 PNG 저장.

    data:
        pandas.DataFrame 또는 dict/list 형태.
        최소 컬럼: t, x, y, yaw
        yaw 단위: rad
    """

    df = pd.DataFrame(data).copy()
    df = df.sort_values(time_col).reset_index(drop=True)

    t = df[time_col].to_numpy(dtype=float)
    x = df[x_col].to_numpy(dtype=float)
    y = df[y_col].to_numpy(dtype=float)
    yaw = df[yaw_col].to_numpy(dtype=float)

    def angle_diff(a, b):
        return np.arctan2(np.sin(a - b), np.cos(a - b))

    dt = np.diff(t)
    dx = np.diff(x)
    dy = np.diff(y)
    dyaw = angle_diff(yaw[1:], yaw[:-1])

    valid = dt > 1e-9

    dpos = np.full_like(dt, np.nan, dtype=float)
    v = np.full_like(dt, np.nan, dtype=float)
    w = np.full_like(dt, np.nan, dtype=float)
    curvature = np.full_like(dt, np.nan, dtype=float)

    dpos[valid] = np.sqrt(dx[valid] ** 2 + dy[valid] ** 2)
    v[valid] = dpos[valid] / dt[valid]
    w[valid] = np.abs(dyaw[valid]) / dt[valid]
    curvature_mask = valid & (dpos >= min_curvature_dpos)
    curvature[curvature_mask] = np.abs(dyaw[curvature_mask]) / dpos[curvature_mask]

    def robust_threshold(arr):
        arr = arr[np.isfinite(arr)]
        if len(arr) == 0:
            return np.inf
        med = np.median(arr)
        mad = np.median(np.abs(arr - med))
        if mad < 1e-12:
            return med
        return med + mad_threshold * 1.4826 * mad

    def robust_zscore(arr):
        z = np.zeros_like(arr, dtype=float)
        finite = np.isfinite(arr)
        vals = arr[finite]
        if len(vals) == 0:
            return z
        med = np.median(vals)
        mad = np.median(np.abs(vals - med))
        if mad < 1e-12:
            return z
        z[finite] = (arr[finite] - med) / (1.4826 * mad)
        z[~np.isfinite(z)] = 0.0
        return np.maximum(z, 0.0)

    v_thr = max_speed if max_speed is not None else robust_threshold(v)
    w_thr = max_yaw_rate if max_yaw_rate is not None else robust_threshold(w)

    curvature_z = robust_zscore(curvature)
    speed_score = np.divide(v, v_thr, out=np.zeros_like(v, dtype=float), where=np.isfinite(v) & (v_thr > 1e-12))
    yaw_rate_score = np.divide(w, w_thr, out=np.zeros_like(w, dtype=float), where=np.isfinite(w) & (w_thr > 1e-12))
    jump_score = speed_weight * speed_score + yaw_rate_weight * yaw_rate_score + curvature_weight * curvature_z

    jump_mask = jump_score >= score_threshold
    jump_indices = np.where(jump_mask)[0] + 1

    events = []
    for i in jump_indices:
        j = i - 1
        event = {
            "index": int(i),
            "time": float(t[i]),
            "dx": float(dx[j]),
            "dy": float(dy[j]),
            "dpos": float(dpos[j]),
            "dyaw_rad": float(dyaw[j]),
            "dyaw_deg": float(np.degrees(dyaw[j])),
            "speed": float(v[j]),
            "yaw_rate": float(w[j]),
            "curvature": float(curvature[j]),
            "curvature_z": float(curvature_z[j]),
            "speed_score": float(speed_score[j]),
            "yaw_rate_score": float(yaw_rate_score[j]),
            "jump_score": float(jump_score[j]),
        }

        if dpos[j] > 0.2 and abs(dyaw[j]) < np.deg2rad(10):
            event["type"] = "translation_jump"
        elif dpos[j] < 0.1 and abs(dyaw[j]) > np.deg2rad(20):
            event["type"] = "yaw_jump"
        else:
            event["type"] = "mixed_jump"

        events.append(event)

    events_df = pd.DataFrame(events)

    fig = plt.figure(figsize=(14, 12))

    ax1 = plt.subplot(3, 2, 1)
    ax1.plot(x, y, linewidth=1)
    if len(jump_indices) > 0:
        ax1.scatter(x[jump_indices], y[jump_indices], marker="x", s=80)
    ax1.set_title("AMCL trajectory")
    ax1.set_xlabel("x [m]")
    ax1.set_ylabel("y [m]")
    ax1.axis("equal")
    ax1.grid(True)

    ax2 = plt.subplot(3, 2, 2)
    ax2.plot(t, x, label="x")
    ax2.plot(t, y, label="y")
    if len(jump_indices) > 0:
        ax2.scatter(t[jump_indices], x[jump_indices], marker="x")
        ax2.scatter(t[jump_indices], y[jump_indices], marker="x")
    ax2.set_title("Position over time")
    ax2.set_xlabel("time [s]")
    ax2.set_ylabel("position [m]")
    ax2.legend()
    ax2.grid(True)

    ax3 = plt.subplot(3, 2, 3)
    ax3.plot(t, yaw)
    if len(jump_indices) > 0:
        ax3.scatter(t[jump_indices], yaw[jump_indices], marker="x")
    ax3.set_title("Yaw over time")
    ax3.set_xlabel("time [s]")
    ax3.set_ylabel("yaw [rad]")
    ax3.grid(True)

    ax4 = plt.subplot(3, 2, 4)
    ax4.plot(t[1:], v)
    ax4.axhline(v_thr, linestyle="--", label=f"threshold={v_thr:.3f}")
    if len(jump_indices) > 0:
        ax4.scatter(t[jump_indices], v[jump_indices - 1], marker="x")
    ax4.set_title("Implied linear speed")
    ax4.set_xlabel("time [s]")
    ax4.set_ylabel("speed [m/s]")
    ax4.legend()
    ax4.grid(True)

    ax5 = plt.subplot(3, 2, 5)
    ax5.plot(t[1:], jump_score, label="jump score")
    ax5.axhline(score_threshold, linestyle="--", label=f"threshold={score_threshold:.3f}")
    if len(jump_indices) > 0:
        ax5.scatter(t[jump_indices], jump_score[jump_indices - 1], marker="x")
    ax5.set_title("Jump score")
    ax5.set_xlabel("time [s]")
    ax5.set_ylabel("score")
    ax5.legend()
    ax5.grid(True)

    ax6 = plt.subplot(3, 2, 6)
    ax6.axis("off")

    summary = [
        f"Total poses: {len(df)}",
        f"Detected jumps: {len(events_df)}",
        f"Speed threshold: {v_thr:.3f} m/s",
        f"Yaw-rate threshold: {w_thr:.3f} rad/s",
        f"Jump-score threshold: {score_threshold:.3f}",
        f"Score = {speed_weight:g}*v/max_v + {yaw_rate_weight:g}*w/max_w + {curvature_weight:g}*curv_z",
    ]

    if len(events_df) > 0:
        summary.append("")
        summary.append("Top jump events:")
        top = events_df.sort_values("jump_score", ascending=False).head(8)
        for _, r in top.iterrows():
            summary.append(
                f"t={r['time']:.2f}s | "
                f"dpos={r['dpos']:.2f}m | "
                f"dyaw={r['dyaw_deg']:.1f}deg | "
                f"v={r['speed']:.2f} | "
                f"w={r['yaw_rate']:.2f} | "
                f"score={r['jump_score']:.2f} | "
                f"{r['type']}"
            )

    ax6.text(
        0.0,
        1.0,
        "\n".join(summary),
        va="top",
        ha="left",
        fontsize=10,
        family="monospace",
    )

    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_png, dpi=160)
    plt.close(fig)

    return events_df


def iter_episode_files(action_dir: Path) -> list[Path]:
    files = []
    for goal_dir in sorted(p for p in action_dir.iterdir() if p.is_dir()):
        files.extend(sorted(goal_dir.glob("*.json")))
    return files


def trajectory_json_to_df(json_path: Path) -> tuple[pd.DataFrame, str, str]:
    with json_path.open("r") as f:
        data = json.load(f)

    goal_name = str(data.get("goal_name") or json_path.parent.name)
    episode_id = str(data.get("episode_id") or json_path.stem)

    rows = []
    for i, step in enumerate(data.get("trajectory", [])):
        pose = step.get("map_pose") or {}
        if not {"x", "y", "yaw"}.issubset(pose):
            continue
        stamp = step.get("stamp_ros_sec", step.get("stamp_unix", i))
        rows.append(
            {
                "t": float(stamp),
                "x": float(pose["x"]),
                "y": float(pose["y"]),
                "yaw": float(pose["yaw"]),
            }
        )

    df = pd.DataFrame(rows)
    if len(df) > 0:
        df["t"] = df["t"] - float(df["t"].iloc[0])
    return df, goal_name, episode_id


def analyze_all_episodes(
    action_dir: Path,
    out_dir: Path,
    max_speed: float | None,
    max_yaw_rate: float | None,
    mad_threshold: float,
    score_threshold: float,
    speed_weight: float,
    yaw_rate_weight: float,
    curvature_weight: float,
    min_curvature_dpos: float,
) -> pd.DataFrame:
    all_events = []
    episode_files = iter_episode_files(action_dir)

    for json_path in episode_files:
        df, goal_name, episode_id = trajectory_json_to_df(json_path)
        if len(df) < 2:
            print(f"[SKIP] {goal_name}/{episode_id}: not enough poses")
            continue

        output_png = out_dir / goal_name / f"{episode_id}.png"
        events = analyze_amcl_trajectory(
            df,
            time_col="t",
            x_col="x",
            y_col="y",
            yaw_col="yaw",
            output_png=output_png,
            max_speed=max_speed,
            max_yaw_rate=max_yaw_rate,
            mad_threshold=mad_threshold,
            score_threshold=score_threshold,
            speed_weight=speed_weight,
            yaw_rate_weight=yaw_rate_weight,
            curvature_weight=curvature_weight,
            min_curvature_dpos=min_curvature_dpos,
        )

        for _, event in events.iterrows():
            row = event.to_dict()
            row.update(
                {
                    "goal_name": goal_name,
                    "episode_id": episode_id,
                    "json_path": str(json_path),
                    "output_png": str(output_png),
                }
            )
            all_events.append(row)

        print(f"[OK] {goal_name}/{episode_id}: poses={len(df)} events={len(events)} -> {output_png}")

    return pd.DataFrame(all_events)


def print_abnormal_episode_lists(events: pd.DataFrame) -> None:
    print("\nAMCL abnormal episodes by goal:")
    if len(events) == 0 or not {"goal_name", "episode_id"}.issubset(events.columns):
        print("  (none)")
        return

    for goal_name in sorted(events["goal_name"].dropna().unique()):
        goal_events = events[events["goal_name"] == goal_name]
        episode_ids = sorted(str(ep) for ep in goal_events["episode_id"].dropna().unique())
        print(f"  {goal_name}: {episode_ids}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--action-dir", type=Path, default=DEFAULT_ACTION_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--max-speed", type=float, default=1.5)
    parser.add_argument("--max-yaw-rate", type=float, default=2.0)
    parser.add_argument("--mad-threshold", type=float, default=6.0)
    parser.add_argument("--jump-score-threshold", type=float, default=20.0)
    parser.add_argument("--speed-weight", type=float, default=1.0)
    parser.add_argument("--yaw-rate-weight", type=float, default=1.0)
    parser.add_argument("--curvature-weight", type=float, default=1.0)
    parser.add_argument(
        "--min-curvature-dpos",
        type=float,
        default=0.03,
        help="Only compute curvature z-score for transitions moving at least this far.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.action_dir.exists():
        raise FileNotFoundError(f"action-dir does not exist: {args.action_dir}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    events = analyze_all_episodes(
        action_dir=args.action_dir,
        out_dir=args.out_dir,
        max_speed=args.max_speed,
        max_yaw_rate=args.max_yaw_rate,
        mad_threshold=args.mad_threshold,
        score_threshold=args.jump_score_threshold,
        speed_weight=args.speed_weight,
        yaw_rate_weight=args.yaw_rate_weight,
        curvature_weight=args.curvature_weight,
        min_curvature_dpos=args.min_curvature_dpos,
    )

    events_csv = args.out_dir / "amcl_events.csv"
    events.to_csv(events_csv, index=False)
    print(f"\nAnalyzed events: {len(events)}")
    print(f"Saved events CSV: {events_csv}")
    print_abnormal_episode_lists(events)


if __name__ == "__main__":
    main()
