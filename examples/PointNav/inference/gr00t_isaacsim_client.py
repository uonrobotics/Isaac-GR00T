"""
Local GR00T Isaac Sim client.

Information flow:
  IsaacSim server (TCP 8765) -> local client -> GR00T inference server (TCP 5000)
  local client -> ROS2 cmd_vel bridge (TCP 8766)
  local client <- ROS2 AMCL pose bridge (TCP 8767)

The heavy GR00T model stays inside the local gr00t_inference_server.py process.
This client only moves JSON/image payloads between local processes.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import socket
import time
import threading
from typing import Optional


ISAACSIM_HOST = "127.0.0.1"
ISAACSIM_PORT = 8765
INFERENCE_HOST = "127.0.0.1"
INFERENCE_PORT = 5000
AMCL_HOST = "127.0.0.1"
AMCL_PORT = 8767
CMD_HOST = "127.0.0.1"
CMD_PORT = 8766
RESET_STOP_REPEATS = 8
RESET_STOP_GAP_SEC = 0.05
RESET_OBS_WARMUP_FRAMES = 8
RESET_SETTLE_SEC = 1.0


# ===== FOR EVALUATION =====
# Values are map/world coordinates: x [m], y [m], yaw [rad].
DEFAULT_FIXED_SPAWN_REPEATS = 10
DEFAULT_FIXED_SPAWN_POSES = [
    # marker 1 --> marker 5
    {"name": "spawn_01", "x": 4.29, "y": 0.35, "yaw": 0.0},
    # marker 5 --> marker 1
    {"name": "spawn_02", "x": 3.5, "y": 11.29, "yaw": 0.0},
    # bottom left --> marker 1
    {"name": "spawn_03", "x": -6.824, "y": 16.089, "yaw": -0.659},
    # bottom right --> marker 5
    {"name": "spawn_04", "x": -3.423, "y": -9.715, "yaw": -1.430},
    # besides forklift --> marker 1
    {"name": "spawn_05", "x": 3.548, "y": -7.491, "yaw": 2.194},
]
# ===== FOR EVALUATION =====


class JsonSocketClient:
    def __init__(self, host: str, port: int, label: str, recv_buf: int = 16 * 1024 * 1024):
        self.host = host
        self.port = port
        self.label = label
        self.recv_buf = recv_buf
        self.sock: Optional[socket.socket] = None
        self.buffer = b""

    def connect(self):
        if self.sock is not None:
            return
        last_log = 0.0
        while self.sock is None:
            try:
                self.sock = socket.create_connection((self.host, self.port))
                self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, self.recv_buf)
                self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                self.buffer = b""
                print(f"[{self.label}] connected to {self.host}:{self.port}")
            except OSError as e:
                now = time.time()
                if now - last_log > 3.0:
                    print(f"[{self.label}] waiting for {self.host}:{self.port} ({e})")
                    last_log = now
                time.sleep(0.5)

    def request(self, payload: dict) -> dict:
        self.connect()
        assert self.sock is not None
        try:
            self.sock.sendall((json.dumps(payload) + "\n").encode("utf-8"))
            while b"\n" not in self.buffer:
                data = self.sock.recv(1 << 20)
                # if not data:
                #     raise RuntimeError(f"{self.label} disconnected")
                self.buffer += data
            line, self.buffer = self.buffer.split(b"\n", 1)
            return json.loads(line.decode("utf-8"))
        except Exception:
            self.close()
            raise

    def close(self):
        if self.sock is not None:
            try:
                self.sock.close()
            except Exception:
                pass
            self.sock = None
            self.buffer = b""


class JsonLineSender:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.sock: Optional[socket.socket] = None

    def _connect(self):
        try:
            self.sock = socket.create_connection((self.host, self.port))
            self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            print(f"[CMD] connected to {self.host}:{self.port}")
        except OSError as e:
            print(f"[CMD] connect failed: {e}")
            self.sock = None

    def send(self, payload: dict):
        if self.sock is None:
            self._connect()
        if self.sock is None:
            return
        try:
            payload.setdefault("client_send_start_time", time.time())
            self.sock.sendall((json.dumps(payload) + "\n").encode("utf-8"))
            payload["client_send_done_time"] = time.time()
        except (ConnectionResetError, BrokenPipeError, OSError) as e:
            print(f"[CMD] send failed, reconnecting: {e}")
            self.close()

    def close(self):
        if self.sock is not None:
            try:
                self.sock.close()
            except Exception:
                pass
            self.sock = None


def yaw_delta(a: float, b: float) -> float:
    return math.atan2(math.sin(a - b), math.cos(a - b))


def extract_pose(resp: dict) -> dict:
    pose = resp.get("pose")
    if isinstance(pose, dict):
        return pose
    if all(k in resp for k in ("x", "y", "yaw")):
        return {"x": resp["x"], "y": resp["y"], "yaw": resp["yaw"]}
    raise RuntimeError(f"response has no pose: {resp}")


def send_stop(cmd: JsonLineSender, repeats: int = 1, gap_sec: float = 0.0):
    for idx in range(max(1, int(repeats))):
        cmd.send({"linear": 0.0, "angular": 0.0})
        if idx < repeats - 1 and gap_sec > 0.0:
            time.sleep(gap_sec)


def normalize_fixed_spawn_pose(raw, idx: int) -> dict:
    if isinstance(raw, dict):
        pose = dict(raw)
    elif isinstance(raw, (list, tuple)) and len(raw) >= 3:
        pose = {"x": raw[0], "y": raw[1], "yaw": raw[2]}
    else:
        raise ValueError(f"invalid fixed spawn pose #{idx + 1}: {raw!r}")
    for key in ("x", "y", "yaw"):
        if key not in pose:
            raise ValueError(f"fixed spawn pose #{idx + 1} missing {key!r}: {raw!r}")
        pose[key] = float(pose[key])
    pose["name"] = str(pose.get("name", f"spawn_{idx + 1:02d}"))
    return pose


def load_fixed_spawn_poses(points_json: str | None, points_file: str | None) -> list[dict]:
    if points_json and points_file:
        raise ValueError("use either --fixed-spawn-points or --fixed-spawn-points-file, not both")
    if points_file:
        with open(os.path.expanduser(points_file), "r", encoding="utf-8") as f:
            raw = json.load(f)
    elif points_json:
        raw = json.loads(points_json)
    else:
        raw = DEFAULT_FIXED_SPAWN_POSES
    if not isinstance(raw, list):
        raise ValueError("fixed spawn points must be a JSON list")
    return [normalize_fixed_spawn_pose(item, idx) for idx, item in enumerate(raw)]


class FixedSpawnSequence:
    def __init__(self, poses: list[dict], repeats_per_pose: int):
        self.poses = poses
        self.repeats_per_pose = max(1, int(repeats_per_pose))
        self.index = 0

    def next(self) -> tuple[dict, int, int]:
        if not self.poses:
            raise RuntimeError("no fixed spawn poses configured")
        pose_idx = (self.index // self.repeats_per_pose) % len(self.poses)
        repeat_idx = (self.index % self.repeats_per_pose) + 1
        self.index += 1
        return self.poses[pose_idx], pose_idx + 1, repeat_idx


def main():
    parser = argparse.ArgumentParser(description="Forward Isaac Sim observations to local GR00T inference server")
    parser.add_argument("--sim-host", default=ISAACSIM_HOST)
    parser.add_argument("--sim-port", type=int, default=ISAACSIM_PORT)
    parser.add_argument("--inference-host", default=INFERENCE_HOST)
    parser.add_argument("--inference-port", type=int, default=INFERENCE_PORT)
    parser.add_argument("--amcl-host", default=AMCL_HOST)
    parser.add_argument("--amcl-port", type=int, default=AMCL_PORT)
    parser.add_argument("--cmd-host", default=CMD_HOST)
    parser.add_argument("--cmd-port", type=int, default=CMD_PORT)
    parser.add_argument("--hz", type=float, default=10.0)
    parser.add_argument("--no-reset", action="store_true")
    parser.add_argument("--use-sim-pose", action="store_true")
    parser.add_argument("--reset-stop-repeats", type=int, default=RESET_STOP_REPEATS)
    parser.add_argument("--reset-stop-gap-sec", type=float, default=RESET_STOP_GAP_SEC)
    parser.add_argument("--reset-obs-warmup-frames", type=int, default=RESET_OBS_WARMUP_FRAMES)
    parser.add_argument("--reset-settle-sec", type=float, default=RESET_SETTLE_SEC)
    parser.add_argument(
        "--fixed-spawn-repeats",
        type=int,
        default=DEFAULT_FIXED_SPAWN_REPEATS,
        help="Number of fixed resets to run at each configured spawn pose.",
    )
    parser.add_argument(
        "--fixed-spawn-points",
        default=None,
        help='JSON list of fixed spawn poses, e.g. \'[{"x":0,"y":0,"yaw":0}]\'.',
    )
    parser.add_argument(
        "--fixed-spawn-points-file",
        default=None,
        help="JSON file containing a list of fixed spawn poses.",
    )
    parser.add_argument(
        "--no-sim-pose-fallback",
        action="store_true",
        help="Stop when AMCL/TF is unavailable instead of using IsaacSim pose.",
    )
    args = parser.parse_args()
    fixed_spawn_sequence = FixedSpawnSequence(
        load_fixed_spawn_poses(args.fixed_spawn_points, args.fixed_spawn_points_file),
        repeats_per_pose=args.fixed_spawn_repeats,
    )

    sim = JsonSocketClient(args.sim_host, args.sim_port, "ISAACSIM")
    infer = JsonSocketClient(args.inference_host, args.inference_port, "GR00T SERVER")
    amcl = JsonSocketClient(args.amcl_host, args.amcl_port, "AMCL BRIDGE")
    cmd = JsonLineSender(args.cmd_host, args.cmd_port)

    period = 1.0 / max(args.hz, 1e-6)
    prev_pose = None
    prev_pose_time = None
    prev_linear = 0.0
    prev_angular = 0.0
    current_episode_id = 0
    cmd_seq = 0
    
    random_reset_requested = threading.Event()
    fixed_reset_requested = threading.Event()

    def keyboard_listener():
        print("[CLIENT] press r + Enter for random reset")
        print("[CLIENT] press f + Enter for fixed-sequence reset")

        while True:
            try:
                key = input().strip().lower()
            except EOFError:
                break

            if key == "r":
                print("\n[CLIENT] random reset requested\n")
                send_stop(cmd, repeats=args.reset_stop_repeats, gap_sec=args.reset_stop_gap_sec)
                random_reset_requested.set()
            elif key == "f":
                print("\n[CLIENT] fixed-sequence reset requested\n")
                send_stop(cmd, repeats=args.reset_stop_repeats, gap_sec=args.reset_stop_gap_sec)
                fixed_reset_requested.set()

    threading.Thread(
        target=keyboard_listener,
        daemon=True,
        name="keyboard-reset",
    ).start()

    def force_stop_for_reset(label: str):
        print(f"[CLIENT] reset: forcing cmd_vel=0 ({label})")
        send_stop(
            cmd,
            repeats=args.reset_stop_repeats,
            gap_sec=args.reset_stop_gap_sec,
        )

    def reset_episode(fixed_pose: dict | None = None, fixed_info: tuple[int, int] | None = None):
        nonlocal current_episode_id
        force_stop_for_reset("before sim reset")
        if args.no_reset:
            obs_resp = sim.request({"cmd": "get_obs"})
            if not obs_resp.get("ok", False):
                raise RuntimeError(f"initial get_obs failed: {obs_resp}")
            pose = extract_pose(obs_resp)
        elif fixed_pose is not None:
            reset_resp = sim.request(
                {
                    "cmd": "reset_to_pose",
                    "x": fixed_pose["x"],
                    "y": fixed_pose["y"],
                    "yaw": fixed_pose["yaw"],
                    "label": fixed_pose["name"],
                }
            )
            if not reset_resp.get("ok", False):
                raise RuntimeError(f"fixed sim reset failed: {reset_resp}")
            pose = extract_pose(reset_resp)
        else:
            reset_resp = sim.request({"cmd": "reset"})
            if not reset_resp.get("ok", False):
                raise RuntimeError(f"sim reset failed: {reset_resp}")
            pose = extract_pose(reset_resp)
        if fixed_pose is not None and fixed_info is not None:
            pose_idx, repeat_idx = fixed_info
            print(
                f"[CLIENT] fixed reset {pose_idx}/{len(fixed_spawn_sequence.poses)} "
                f"repeat {repeat_idx}/{fixed_spawn_sequence.repeats_per_pose} "
                f"name={fixed_pose['name']}"
            )
        print(f"[CLIENT] reset pose x={pose['x']:.2f} y={pose['y']:.2f} yaw={pose['yaw']:.2f}")
        force_stop_for_reset("after sim reset")
        if not args.use_sim_pose:
            amcl.request(
                {
                    "cmd": "publish_initial_pose",
                    "x": pose["x"],
                    "y": pose["y"],
                    "yaw": pose["yaw"],
                    "repeat": 3,
                    "gap_sec": 0.8,
                    "settle_sec": 5.0,
                }
            )
        force_stop_for_reset("after AMCL initialpose")
        if args.reset_settle_sec > 0.0:
            print(f"[CLIENT] reset: settling robot/camera for {args.reset_settle_sec:.1f}s")
            time.sleep(args.reset_settle_sec)
        for idx in range(max(0, int(args.reset_obs_warmup_frames))):
            obs_resp = sim.request({"cmd": "get_obs"})
            if not obs_resp.get("ok", False):
                raise RuntimeError(f"reset warmup get_obs failed: {obs_resp}")
            if idx == 0:
                print("[CLIENT] reset: warming camera frames")
        force_stop_for_reset("before inference resume")
        current_episode_id += 1
        print(f"[CLIENT] reset: episode {current_episode_id} ready; inference may resume")
        return pose

    try:
        sim.connect()
        infer.connect()
        cmd._connect()
        if not args.use_sim_pose:
            amcl.connect()
        reset_episode()

        while True:
            if random_reset_requested.is_set() or fixed_reset_requested.is_set():
                use_fixed_reset = fixed_reset_requested.is_set()
                random_reset_requested.clear()
                fixed_reset_requested.clear()

                print("\n====================")
                print("FIXED RESET EPISODE" if use_fixed_reset else "RANDOM RESET EPISODE")
                print("====================")

                try:
                    if use_fixed_reset:
                        fixed_pose, pose_idx, repeat_idx = fixed_spawn_sequence.next()
                        reset_episode(fixed_pose=fixed_pose, fixed_info=(pose_idx, repeat_idx))
                    else:
                        reset_episode()

                    prev_pose = None
                    prev_pose_time = None
                    prev_linear = 0.0
                    prev_angular = 0.0

                    print("[CLIENT] reset complete\n")

                except Exception as e:
                    print(f"[CLIENT] reset failed: {e}")
                    force_stop_for_reset("reset failed")

                continue
            
            loop_t = time.time()
            obs_resp = sim.request({"cmd": "get_obs"})
            if not obs_resp.get("ok", False):
                print(f"[SIM] get_obs failed: {obs_resp}")
                time.sleep(period)
                continue

            if args.use_sim_pose:
                pose = extract_pose(obs_resp)
            else:
                pose_resp = amcl.request({"cmd": "get_pose"})
                if not pose_resp.get("ok", False):
                    if args.no_sim_pose_fallback:
                        print(f"[AMCL] waiting for pose: {pose_resp}")
                        send_stop(cmd)
                        time.sleep(period)
                        continue
                    pose = extract_pose(obs_resp)
                    print(f"[AMCL] unavailable, using IsaacSim pose: {pose_resp}")
                else:
                    pose = extract_pose(pose_resp)

            now = time.time()
            odom_vlin = 0.0
            odom_vang = 0.0
            if prev_pose is not None and prev_pose_time is not None:
                dt = max(now - prev_pose_time, 1e-6)
                dx = float(pose["x"]) - float(prev_pose["x"])
                dy = float(pose["y"]) - float(prev_pose["y"])
                yaw = float(prev_pose["yaw"])
                odom_vlin = (math.cos(yaw) * dx + math.sin(yaw) * dy) / dt
                odom_vang = yaw_delta(float(pose["yaw"]), float(prev_pose["yaw"])) / dt
            prev_pose = dict(pose)
            prev_pose_time = now

            images_b64 = obs_resp.get("images_b64")
            if isinstance(images_b64, dict):
                images_b64 = dict(images_b64)
            else:
                images_b64 = {"ego_view": obs_resp["image_b64"]}
            legacy_image_b64 = images_b64.get("ego_view") or obs_resp["image_b64"]

            action = infer.request(
                {
                    # Keep "image" for single-view/backward compatibility.
                    # Multiview checkpoints consume the "images" mapping.
                    "image": legacy_image_b64,
                    "images": images_b64,
                    "image_capture_timestamp": obs_resp.get("image_capture_timestamp"),
                    "image_capture_timestamps": obs_resp.get("image_capture_timestamps", {}),
                    "sim_observation_timestamp": obs_resp.get("timestamp"),
                    "episode_id": current_episode_id,
                    "camera_mode": obs_resp.get("camera_mode", "single"),
                    "camera_layout": obs_resp.get("camera_layout", "default"),
                    "views": obs_resp.get("views", list(images_b64.keys())),
                    "amcl_x": float(pose["x"]),
                    "amcl_y": float(pose["y"]),
                    "amcl_yaw": float(pose["yaw"]),
                    "odom_vlin": odom_vlin,
                    "odom_vang": odom_vang,
                    "cmd_linear": prev_linear,
                    "cmd_angular": prev_angular,
                }
            )
            linear = float(action.get("linear", 0.0))
            angular = float(action.get("angular", 0.0))
            prev_linear = linear
            prev_angular = angular
            cmd_seq += 1
            cmd_msg = {
                "linear": linear,
                "angular": angular,
                "seq": cmd_seq,
                "client_send_time": time.time(),
            }
            cmd.send(cmd_msg)
            send_done = cmd_msg.get("client_send_done_time")
            if send_done is not None:
                sendall_ms = (send_done - cmd_msg["client_send_time"]) * 1000.0
                if sendall_ms > 20.0:
                    print(f"[CLIENT CMD LATENCY] seq={cmd_seq} sendall={sendall_ms:.1f}ms")
            print(
                f"[CLIENT] pose=({pose['x']:.2f},{pose['y']:.2f},{pose['yaw']:.2f}) "
                f"cmd seq={cmd_seq} linear={linear:+.3f} angular={angular:+.3f}"
            )

            elapsed = time.time() - loop_t
            if elapsed < period:
                time.sleep(period - elapsed)

    except KeyboardInterrupt:
        print("\n[CLIENT] interrupted")
    finally:
        send_stop(cmd, repeats=RESET_STOP_REPEATS, gap_sec=RESET_STOP_GAP_SEC)
        sim.close()
        infer.close()
        amcl.close()
        cmd.close()


if __name__ == "__main__":
    main()
