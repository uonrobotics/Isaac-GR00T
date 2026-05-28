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
            self.sock.sendall((json.dumps(payload) + "\n").encode("utf-8"))
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
        "--no-sim-pose-fallback",
        action="store_true",
        help="Stop when AMCL/TF is unavailable instead of using IsaacSim pose.",
    )
    args = parser.parse_args()

    sim = JsonSocketClient(args.sim_host, args.sim_port, "ISAACSIM")
    infer = JsonSocketClient(args.inference_host, args.inference_port, "GR00T SERVER")
    amcl = JsonSocketClient(args.amcl_host, args.amcl_port, "AMCL BRIDGE")
    cmd = JsonLineSender(args.cmd_host, args.cmd_port)

    period = 1.0 / max(args.hz, 1e-6)
    prev_pose = None
    prev_pose_time = None
    prev_linear = 0.0
    prev_angular = 0.0
    
    reset_requested = threading.Event()

    def keyboard_listener():
        print("[CLIENT] press r + Enter to reset episode")

        while True:
            try:
                key = input().strip().lower()
            except EOFError:
                break

            if key == "r":
                print("\n[CLIENT] reset requested\n")
                reset_requested.set()

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

    def reset_episode():
        force_stop_for_reset("before sim reset")
        if args.no_reset:
            obs_resp = sim.request({"cmd": "get_obs"})
            if not obs_resp.get("ok", False):
                raise RuntimeError(f"initial get_obs failed: {obs_resp}")
            pose = extract_pose(obs_resp)
        else:
            reset_resp = sim.request({"cmd": "reset"})
            if not reset_resp.get("ok", False):
                raise RuntimeError(f"sim reset failed: {reset_resp}")
            pose = extract_pose(reset_resp)
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
        return pose

    try:
        sim.connect()
        infer.connect()
        cmd._connect()
        if not args.use_sim_pose:
            amcl.connect()
        reset_episode()

        while True:
            if reset_requested.is_set():
                reset_requested.clear()

                print("\n====================")
                print("RESET EPISODE")
                print("====================")

                try:
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
                images_b64.setdefault("ego_view", obs_resp["image_b64"])
            else:
                images_b64 = {"ego_view": obs_resp["image_b64"]}

            action = infer.request(
                {
                    # Keep "image" for single-view/backward compatibility.
                    # Multiview checkpoints consume the "images" mapping.
                    "image": images_b64["ego_view"],
                    "images": images_b64,
                    "image_capture_timestamp": obs_resp.get("image_capture_timestamp"),
                    "image_capture_timestamps": obs_resp.get("image_capture_timestamps", {}),
                    "sim_observation_timestamp": obs_resp.get("timestamp"),
                    "camera_mode": obs_resp.get("camera_mode", "single"),
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
            cmd.send({"linear": linear, "angular": angular})
            print(
                f"[CLIENT] pose=({pose['x']:.2f},{pose['y']:.2f},{pose['yaw']:.2f}) "
                f"cmd linear={linear:+.3f} angular={angular:+.3f}"
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
