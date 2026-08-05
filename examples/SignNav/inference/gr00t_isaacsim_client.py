"""
Local GR00T Isaac Sim client for SignNav.

Information flow:
  IsaacSim server (TCP 8765) -> local client -> GR00T inference server (TCP 5000)
  local client -> ROS2 cmd_vel bridge (TCP 8766)

SignNav does not use localization. The client forwards camera observations and
previous command feedback only; Isaac Sim remains the robot/camera simulator.
"""

from __future__ import annotations

import argparse
from collections import deque
import json
import socket
import time
import threading
from typing import Optional


ISAACSIM_HOST = "127.0.0.1"
ISAACSIM_PORT = 8765
INFERENCE_HOST = "127.0.0.1"
INFERENCE_PORT = 5000
CMD_HOST = "127.0.0.1"
CMD_PORT = 8766
RESET_STOP_REPEATS = 8
RESET_STOP_GAP_SEC = 0.05
RESET_OBS_WARMUP_FRAMES = 8
RESET_SETTLE_SEC = 1.0
TIMING_LOG_EVERY = 10
TIMING_WINDOW = 50
ANSI_RESET = "\033[0m"
ANSI_BOLD = "\033[1m"
ANSI_CYAN = "\033[96m"
ANSI_YELLOW = "\033[93m"
ANSI_MAGENTA = "\033[95m"


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
                if not data:
                    raise RuntimeError(f"{self.label} disconnected")
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


def send_stop(cmd: JsonLineSender, repeats: int = 1, gap_sec: float = 0.0):
    for idx in range(max(1, int(repeats))):
        cmd.send({"linear": 0.0, "angular": 0.0})
        if idx < repeats - 1 and gap_sec > 0.0:
            time.sleep(gap_sec)


def main():
    parser = argparse.ArgumentParser(description="Forward Isaac Sim observations to SignNav GR00T inference server")
    parser.add_argument("--sim-host", default=ISAACSIM_HOST)
    parser.add_argument("--sim-port", type=int, default=ISAACSIM_PORT)
    parser.add_argument("--inference-host", default=INFERENCE_HOST)
    parser.add_argument("--inference-port", type=int, default=INFERENCE_PORT)
    parser.add_argument("--cmd-host", default=CMD_HOST)
    parser.add_argument("--cmd-port", type=int, default=CMD_PORT)
    parser.add_argument("--hz", type=float, default=10.0)
    parser.add_argument("--no-reset", action="store_true")
    parser.add_argument("--reset-stop-repeats", type=int, default=RESET_STOP_REPEATS)
    parser.add_argument("--reset-stop-gap-sec", type=float, default=RESET_STOP_GAP_SEC)
    parser.add_argument("--reset-obs-warmup-frames", type=int, default=RESET_OBS_WARMUP_FRAMES)
    parser.add_argument("--reset-settle-sec", type=float, default=RESET_SETTLE_SEC)
    parser.add_argument(
        "--timing-log-every",
        type=int,
        default=TIMING_LOG_EVERY,
        help="Print loop timing every N inference cycles; set 0 to disable.",
    )
    args = parser.parse_args()

    sim = JsonSocketClient(args.sim_host, args.sim_port, "ISAACSIM")
    infer = JsonSocketClient(args.inference_host, args.inference_port, "GR00T SERVER")
    cmd = JsonLineSender(args.cmd_host, args.cmd_port)

    period = 1.0 / max(args.hz, 1e-6)
    current_episode_id = 0
    cmd_seq = 0
    prev_linear = 0.0
    prev_angular = 0.0
    random_reset_requested = threading.Event()
    fixed_reset_requested = threading.Event()
    fixed_spawn_pose = None
    loop_durations = deque(maxlen=TIMING_WINDOW)
    obs_durations = deque(maxlen=TIMING_WINDOW)
    infer_durations = deque(maxlen=TIMING_WINDOW)
    cmd_durations = deque(maxlen=TIMING_WINDOW)
    sleep_durations = deque(maxlen=TIMING_WINDOW)

    def keyboard_listener():
        print("[CLIENT] press r + Enter for random reset; f + Enter for first-pose reset")
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
                print("\n[CLIENT] first-pose reset requested\n")
                send_stop(cmd, repeats=args.reset_stop_repeats, gap_sec=args.reset_stop_gap_sec)
                fixed_reset_requested.set()

    threading.Thread(target=keyboard_listener, daemon=True, name="keyboard-reset").start()

    def force_stop_for_reset(label: str):
        print(f"[CLIENT] reset: forcing cmd_vel=0 ({label})")
        send_stop(cmd, repeats=args.reset_stop_repeats, gap_sec=args.reset_stop_gap_sec)

    def reset_episode(reset_kind: str = "random"):
        nonlocal current_episode_id
        force_stop_for_reset("before sim reset")
        if args.no_reset:
            obs_resp = sim.request({"cmd": "get_obs"})
            if not obs_resp.get("ok", False):
                raise RuntimeError(f"initial get_obs failed: {obs_resp}")
            pose = obs_resp.get("pose")
        elif reset_kind == "fixed":
            if fixed_spawn_pose is None:
                raise RuntimeError("first-pose reset requested before initial pose was captured")
            reset_resp = sim.request(
                {
                    "cmd": "reset_to_pose",
                    "x": fixed_spawn_pose["x"],
                    "y": fixed_spawn_pose["y"],
                    "yaw": fixed_spawn_pose["yaw"],
                    "label": "first_pose",
                }
            )
            if not reset_resp.get("ok", False):
                raise RuntimeError(f"fixed sim reset failed: {reset_resp}")
            pose = reset_resp.get("pose")
        else:
            reset_resp = sim.request({"cmd": "reset"})
            if not reset_resp.get("ok", False):
                raise RuntimeError(f"sim reset failed: {reset_resp}")
            pose = reset_resp.get("pose")
        force_stop_for_reset("after sim reset")
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
        if isinstance(pose, dict):
            print(
                f"[CLIENT] reset: episode {current_episode_id} ready "
                f"kind={reset_kind} pose=({pose['x']:.2f},{pose['y']:.2f},{pose['yaw']:.2f}); "
                "inference may resume"
            )
        else:
            print(
                f"[CLIENT] reset: episode {current_episode_id} ready "
                f"kind={reset_kind}; inference may resume"
            )
        return pose

    try:
        sim.connect()
        infer.connect()
        cmd._connect()
        fixed_spawn_pose = reset_episode("random")
        if isinstance(fixed_spawn_pose, dict):
            print(
                f"[CLIENT] first-pose reset target captured: "
                f"({fixed_spawn_pose['x']:.2f},{fixed_spawn_pose['y']:.2f},{fixed_spawn_pose['yaw']:.2f})"
            )

        while True:
            if fixed_reset_requested.is_set() or random_reset_requested.is_set():
                reset_kind = "fixed" if fixed_reset_requested.is_set() else "random"
                fixed_reset_requested.clear()
                random_reset_requested.clear()
                try:
                    reset_episode(reset_kind)
                    prev_linear = 0.0
                    prev_angular = 0.0
                    print(f"[CLIENT] {reset_kind} reset complete\n")
                except Exception as e:
                    print(f"[CLIENT] reset failed: {e}")
                    force_stop_for_reset("reset failed")
                continue

            loop_t = time.time()
            obs_t = time.time()
            obs_resp = sim.request({"cmd": "get_obs"})
            obs_sec = time.time() - obs_t
            if not obs_resp.get("ok", False):
                print(f"[SIM] get_obs failed: {obs_resp}")
                time.sleep(period)
                continue

            images_b64 = obs_resp.get("images_b64")
            if isinstance(images_b64, dict):
                images_b64 = dict(images_b64)
            else:
                images_b64 = {"ego_view": obs_resp["image_b64"]}
            legacy_image_b64 = images_b64.get("ego_view") or obs_resp["image_b64"]

            infer_payload = {
                "image": legacy_image_b64,
                "images": images_b64,
                "image_capture_timestamp": obs_resp.get("image_capture_timestamp"),
                "image_capture_timestamps": obs_resp.get("image_capture_timestamps", {}),
                "sim_observation_timestamp": obs_resp.get("timestamp"),
                "episode_id": current_episode_id,
                "camera_mode": obs_resp.get("camera_mode", "single"),
                "camera_layout": obs_resp.get("camera_layout", "default"),
                "views": obs_resp.get("views", list(images_b64.keys())),
                "cmd_linear": prev_linear,
                "cmd_angular": prev_angular,
            }

            infer_t = time.time()
            action = infer.request(infer_payload)
            infer_sec = time.time() - infer_t

            linear = float(action.get("linear", 0.0))
            angular = float(action.get("angular", 0.0))
            action_step = action.get("action_step")
            prev_linear = linear
            prev_angular = angular
            cmd_seq += 1
            cmd_msg = {
                "linear": linear,
                "angular": angular,
                "seq": cmd_seq,
                "action_step": action_step,
                "client_send_time": time.time(),
            }
            cmd_t = time.time()
            cmd.send(cmd_msg)
            cmd_sec = time.time() - cmd_t
            print(
                f"[CLIENT] cmd seq={cmd_seq} action_step={action_step} "
                f"linear={linear:+.3f} angular={angular:+.3f}"
            )

            elapsed = time.time() - loop_t
            sleep_sec = max(0.0, period - elapsed)
            if elapsed < period:
                time.sleep(sleep_sec)
            loop_sec = time.time() - loop_t
            loop_durations.append(loop_sec)
            obs_durations.append(obs_sec)
            infer_durations.append(infer_sec)
            cmd_durations.append(cmd_sec)
            sleep_durations.append(sleep_sec)
            if args.timing_log_every > 0 and cmd_seq % args.timing_log_every == 0:
                avg_loop = sum(loop_durations) / len(loop_durations)
                avg_obs = sum(obs_durations) / len(obs_durations)
                avg_infer = sum(infer_durations) / len(infer_durations)
                avg_cmd = sum(cmd_durations) / len(cmd_durations)
                avg_sleep = sum(sleep_durations) / len(sleep_durations)
                actual_hz = 1.0 / avg_loop if avg_loop > 0.0 else 0.0
                print(
                    f"{ANSI_BOLD}{ANSI_CYAN}⏱️  [TIMING]{ANSI_RESET} "
                    f"{ANSI_YELLOW}seq={cmd_seq} target={args.hz:.2f}Hz "
                    f"actual={actual_hz:.2f}Hz{ANSI_RESET} "
                    f"{ANSI_MAGENTA}loop={loop_sec * 1000:.1f}ms "
                    f"avg={avg_loop * 1000:.1f}ms{ANSI_RESET} "
                    f"obs={obs_sec * 1000:.1f}/{avg_obs * 1000:.1f}ms "
                    f"infer={infer_sec * 1000:.1f}/{avg_infer * 1000:.1f}ms "
                    f"cmd={cmd_sec * 1000:.1f}/{avg_cmd * 1000:.1f}ms "
                    f"sleep={sleep_sec * 1000:.1f}/{avg_sleep * 1000:.1f}ms"
                )

    except KeyboardInterrupt:
        print("\n[CLIENT] interrupted")
    finally:
        send_stop(cmd, repeats=RESET_STOP_REPEATS, gap_sec=RESET_STOP_GAP_SEC)
        sim.close()
        infer.close()
        cmd.close()


if __name__ == "__main__":
    main()
