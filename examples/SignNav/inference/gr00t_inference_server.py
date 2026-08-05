"""
GR00T SignNav inference server.

Protocol (newline-delimited JSON over TCP):
  Request:
    {
      "image": "<base64 JPEG>",
      "images": {"ego_view": "..."},
      "cmd_linear": <float>,
      "cmd_angular": <float>,
      "episode_id": <int>
    }

  Response:
    {
      "linear": <float>,
      "angular": <float>,
      "action_step": <int>
    }

SignNav intentionally does not consume localization. State is only speed, and
the model predicts the 16-step vel_cmd horizon while the server executes one
selected step, matching the PointNav inference pattern.
"""

from __future__ import annotations

import argparse
import base64
from collections import defaultdict, deque
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import io
import json
from pathlib import Path
import queue
import socket
import threading
import time
from urllib.parse import urlparse

import numpy as np
from PIL import Image

from gr00t.policy.gr00t_policy import Gr00tPolicy
from gr00t.data.embodiment_tags import EmbodimentTag


LISTEN_PORT = 5000
WEB_PORT = 9090
WARMUP_STEPS = 4
ACTION_STEP_IDX = 1
DEFAULT_MODALITY_CONFIG = Path(__file__).resolve().parents[1] / "modality_config_signnav.py"
DEFAULT_TARGET_AREA = 1

PROMPT_VERSION = 1 
PROMPT_TEMPLATES = {
    1: (
        "Your goal is to navigate safely to Area {area} using directional signs. "
        "Read the visible sign panels and select the panel whose label includes Area {area}. "
        "Area ranges include all areas within the range, and comma-separated labels include all listed areas. "
        "Use only the arrow attached to the selected panel. "
        "Treat the selected arrow as the route to follow at the next relevant junction, not necessarily as an immediate turn. "
        "Use the current scene geometry to approach and enter the indicated corridor. "
        "If the sign is no longer visible, remember its direction until that route choice has been completed. "
        "Avoid collisions and stop only after reaching Area {area}."
    ),
    2: (
        "Navigate safely to Area {area} using visible directional signs. "
        "Rule 1: IF signs are visible, select the panel whose label includes Area {area}, including ranges and comma-separated lists. "
        "Rule 2: Follow only the arrow attached to the matching panel. "
        "Rule 3: IF the sign leaves view, remember its direction until the related junction is crossed. "
        "Rule 4: Approach the junction and turn only when the indicated corridor becomes reachable. "
        "Rule 5: After crossing the junction, search for the next relevant sign. "
        "Rule 6: Avoid obstacles and stop only after reaching Area {area}."
    ),
    3: (
        "TASK_TYPE: Sign guided navigation "
        "TARGET_AREA: Area {area} "
        "GOAL: Reach the target safely "
        "SIGN_SELECTION: Select the panel whose label includes the target area "
        "AREA_MATCHING: Ranges include all intermediate areas and lists include all listed areas "
        "ARROW_BINDING: Follow only the arrow attached to the matched panel "
        "MEMORY_WRITE: Store the matched arrow direction as the active route "
        "MEMORY_RETAIN: Keep the active route even after the sign leaves view "
        "MEMORY_USE: Apply the active route at the next relevant junction "
        "TURN_TIMING: Turn only when the indicated corridor is reachable "
        "STATIC_CONTROL: Avoid walls and static obstacles "
        "STOP_CONDITION: Stop only after reaching the target area"
    ),
}

_prompt_lock = threading.Lock()
_prompt_state = {
    "prompt_version": PROMPT_VERSION,
    "target_area": DEFAULT_TARGET_AREA,
    "revision": 0,
}


_dash_lock = threading.Lock()
_dash_state = {
    "image_b64": None,
    "images_b64": {},
    "telemetry": {},
}
_sse_subscribers: list[queue.Queue] = []
_sse_lock = threading.Lock()


def normalize_target_area(area: int) -> int:
    area = int(area)
    if not 1 <= area <= 12:
        raise ValueError("target area must be between 1 and 12")
    return area


def normalize_prompt_version(version: int) -> int:
    version = int(version)
    if version not in PROMPT_TEMPLATES:
        raise ValueError(f"prompt version must be one of {sorted(PROMPT_TEMPLATES)}")
    return version


def build_prompt(prompt_version: int, target_area: int) -> str:
    return PROMPT_TEMPLATES[prompt_version].format(area=target_area)


def get_prompt_state() -> dict:
    with _prompt_lock:
        state = dict(_prompt_state)
    state["language"] = build_prompt(state["prompt_version"], state["target_area"])
    return state


def set_prompt_state(prompt_version: int | None = None, target_area: int | None = None) -> dict:
    with _prompt_lock:
        if prompt_version is not None:
            _prompt_state["prompt_version"] = normalize_prompt_version(prompt_version)
        if target_area is not None:
            _prompt_state["target_area"] = normalize_target_area(target_area)
        _prompt_state["revision"] += 1
        state = dict(_prompt_state)
    state["language"] = build_prompt(state["prompt_version"], state["target_area"])
    return state


def print_prompt_state(prefix: str = "[PROMPT]"):
    state = get_prompt_state()
    print(
        f"{prefix} prompt_v{state['prompt_version']} AREA_{state['target_area']} "
        f"revision={state['revision']}"
    )
    print(f"{prefix} {state['language']}")


def _keyboard_listener():
    print("[PROMPT] Keyboard: enter 1..12 or a1..a12 to switch target area; s/show to print current prompt")
    while True:
        try:
            key = input().strip().lower()
        except EOFError:
            break
        if key in {"s", "show"}:
            print_prompt_state()
            continue
        if key.startswith("a"):
            key = key[1:]
        try:
            target_area = int(key)
        except ValueError:
            print("[PROMPT] ignored input. Use 1..12, a1..a12, or show.")
            continue
        try:
            state = set_prompt_state(target_area=target_area)
        except ValueError as e:
            print(f"[PROMPT] {e}")
            continue
        print(
            f"\n[PROMPT SWITCH] AREA_{state['target_area']} "
            f"(prompt_v{state['prompt_version']}, revision={state['revision']})\n"
        )
        print(f"[PROMPT] {state['language']}")


def load_modality_config(path: str | Path):
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"modality config path does not exist: {path}")
    spec = importlib.util.spec_from_file_location("signnav_modality_config", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load modality config: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    print(f"[GR00T] loaded modality config: {path}")


def _push_dashboard(images_b64, telemetry: dict):
    if isinstance(images_b64, dict):
        images = dict(images_b64)
        image_b64 = images.get("ego_view") or next(iter(images.values()), None)
    else:
        image_b64 = images_b64
        images = {"ego_view": image_b64} if image_b64 else {}
    with _dash_lock:
        _dash_state["image_b64"] = image_b64
        _dash_state["images_b64"] = images
        _dash_state["telemetry"] = telemetry
    payload = "data: " + json.dumps(telemetry) + "\n\n"
    with _sse_lock:
        dead = []
        for q in _sse_subscribers:
            try:
                q.put_nowait(payload)
            except queue.Full:
                dead.append(q)
        for q in dead:
            _sse_subscribers.remove(q)


_DASHBOARD_HTML = """\
<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<title>SignNav GR00T Dashboard</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: #101114; color: #e8e8e8; font-family: 'Courier New', monospace;
         display: flex; flex-direction: column; align-items: center; padding: 10px; gap: 10px; }
  h1 { color: #7fd36b; font-size: 1.15rem; letter-spacing: 1px; }
  .grid { width: 100%; display: grid; grid-template-columns: minmax(0, 1fr) 330px; gap: 10px; }
  .views { min-height: calc(100vh - 50px); display: flex; align-items: flex-start; justify-content: center; }
  .camera { background: #050507; border: 1px solid #303038; border-radius: 8px; overflow: hidden; }
  .camera .title { padding: 5px 8px; color: #aeb3bd; background: #15161b; border-bottom: 1px solid #303038; font-size: 0.72rem; text-transform: uppercase; }
  .camera img { width: 100%; height: auto; object-fit: contain; display: block; }
  .views .camera { width: min(100%, 1280px); }
  .panel { background: #191a20; border: 1px solid #303038; border-radius: 8px; padding: 12px; display: flex; flex-direction: column; gap: 10px; }
  .row { display: flex; justify-content: space-between; align-items: center; gap: 12px; }
  .label { color: #8d9098; font-size: 0.75rem; text-transform: uppercase; }
  .value { color: #fff; font-weight: bold; }
  .bar-wrap { background: #0b0c10; border-radius: 4px; height: 10px; overflow: hidden; position: relative; }
  .bar-center { position: absolute; left: 50%; width: 2px; height: 100%; background: #575a63; transform: translateX(-50%); }
  .bar { height: 100%; border-radius: 4px; }
  .bar.pos { background: #7fd36b; }
  .bar.neg { background: #e36060; position: absolute; right: 0; }
  .status { color: #666; font-size: 0.8rem; }
  .status.connected { color: #7fd36b; }
  .save-btn { background: #26311f; color: #dff6d4; border: 1px solid #7fd36b; border-radius: 6px; padding: 9px 10px; font-family: inherit; font-weight: bold; cursor: pointer; }
  .save-btn:hover { background: #314226; }
  .save-btn:disabled { cursor: wait; opacity: 0.65; }
  .save-status { color: #8d9098; font-size: 0.72rem; line-height: 1.35; overflow-wrap: anywhere; }
  .save-status.ok { color: #7fd36b; }
  .save-status.err { color: #e36060; }
  @media (max-width: 1100px) { .grid { grid-template-columns: 1fr; } .views { min-height: auto; } .camera img { height: auto; } }
</style>
</head>
<body>
<h1>SignNav GR00T Dashboard</h1>
<div class="grid">
  <div class="views">
    <div class="camera"><div class="title">ego view / model input</div><img id="cam" src="/image" alt="ego view"></div>
  </div>
  <div class="panel">
    <div class="row"><span class="label">Step</span><span class="value" id="step">-</span></div>
    <div class="row"><span class="label">Episode</span><span class="value" id="episode">-</span></div>
    <div class="row"><span class="label">Target Area</span><span class="value" id="target_area">-</span></div>
    <div class="row"><span class="label">Prompt</span><span class="value" id="prompt_version">-</span></div>
    <div class="row"><span class="label">Speed</span><span class="value" id="speed">-</span></div>
    <div class="row"><span class="label">Action Step</span><span class="value" id="action_step">-</span></div>
    <div class="row"><span class="label">Cam to Input</span><span class="value" id="latency">-</span></div>
    <div class="label">Linear cmd</div>
    <div class="row"><span class="value" id="vx">-</span></div>
    <div class="bar-wrap"><div class="bar-center"></div><div class="bar" id="vx_bar"></div></div>
    <div class="label">Angular cmd</div>
    <div class="row"><span class="value" id="wz">-</span></div>
    <div class="bar-wrap"><div class="bar-center"></div><div class="bar" id="wz_bar"></div></div>
    <button class="save-btn" id="save_ego" type="button">Save ego view</button>
    <div class="save-status" id="save_status">default: ~/Pictures</div>
    <div class="status" id="conn_status">waiting...</div>
  </div>
</div>
<script>
function barUpdate(id, val, maxVal) {
  const bar = document.getElementById(id);
  const pct = Math.min(Math.abs(val) / maxVal * 50, 50);
  bar.style.width = pct + "%";
  if (val >= 0) { bar.className = "bar pos"; bar.style.left = "50%"; bar.style.right = ""; }
  else { bar.className = "bar neg"; bar.style.right = "50%"; bar.style.left = ""; }
}
const es = new EventSource("/stream");
es.onopen = () => { const el = document.getElementById("conn_status"); el.textContent = "connected"; el.className = "status connected"; };
es.onerror = () => { const el = document.getElementById("conn_status"); el.textContent = "disconnected"; el.className = "status"; };
es.onmessage = e => {
  const d = JSON.parse(e.data);
  document.getElementById("step").textContent = d.step ?? "-";
  document.getElementById("episode").textContent = d.episode_id ?? "-";
  document.getElementById("target_area").textContent = d.target_area !== undefined ? "AREA_" + d.target_area : "-";
  document.getElementById("prompt_version").textContent = d.prompt_version !== undefined ? "v" + d.prompt_version : "-";
  document.getElementById("speed").textContent = d.speed !== undefined ? d.speed.toFixed(3) : "-";
  document.getElementById("action_step").textContent =
    d.action_step !== undefined && d.action_horizon !== undefined
      ? d.action_step + " / " + (d.action_horizon - 1)
      : (d.action_step ?? "-");
  document.getElementById("latency").textContent = d.camera_to_model_input_ms !== undefined ? d.camera_to_model_input_ms.toFixed(1) + " ms" : "-";
  document.getElementById("vx").textContent = d.vx !== undefined ? d.vx.toFixed(4) : "-";
  document.getElementById("wz").textContent = d.wz !== undefined ? d.wz.toFixed(4) : "-";
  if (d.vx !== undefined) barUpdate("vx_bar", d.vx, 1.0);
  if (d.wz !== undefined) barUpdate("wz_bar", d.wz, 1.5);
};
setInterval(() => {
  const t = Date.now();
  document.getElementById("cam").src = "/image?" + t;
}, 150);
document.getElementById("save_ego").onclick = async () => {
  const btn = document.getElementById("save_ego");
  const status = document.getElementById("save_status");
  btn.disabled = true;
  status.className = "save-status";
  status.textContent = "saving...";
  try {
    const resp = await fetch("/save_ego", { method: "POST" });
    const data = await resp.json();
    if (!resp.ok || !data.ok) throw new Error(data.error || "save failed");
    status.className = "save-status ok";
    status.textContent = "saved: " + data.path;
  } catch (err) {
    status.className = "save-status err";
    status.textContent = err.message;
  } finally {
    btn.disabled = false;
  }
};
</script>
</body>
</html>
"""


class DashboardHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        return

    def _send_bytes(self, payload: bytes, status: HTTPStatus, content_type: str):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _send_json(self, payload: dict, status: HTTPStatus = HTTPStatus.OK):
        self._send_bytes(
            json.dumps(payload).encode("utf-8"),
            status,
            "application/json",
        )

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/":
            self._send_bytes(
                _DASHBOARD_HTML.encode("utf-8"),
                HTTPStatus.OK,
                "text/html; charset=utf-8",
            )
            return

        if path == "/image":
            with _dash_lock:
                b64 = _dash_state["image_b64"]
            if b64 is None:
                self.send_response(HTTPStatus.NO_CONTENT)
                self.end_headers()
                return
            self._send_bytes(base64.b64decode(b64), HTTPStatus.OK, "image/jpeg")
            return

        if path == "/stream":
            self._stream_events()
            return

        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self):
        path = urlparse(self.path).path
        if path != "/save_ego":
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        with _dash_lock:
            b64 = _dash_state["image_b64"]
        if b64 is None:
            self._send_json(
                {"ok": False, "error": "no ego image available yet"},
                HTTPStatus.CONFLICT,
            )
            return

        save_dir = Path("~/Pictures").expanduser()
        save_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        millis = int((time.time() % 1.0) * 1000)
        save_path = save_dir / f"signnav_ego_{timestamp}_{millis:03d}.jpg"
        save_path.write_bytes(base64.b64decode(b64))
        self._send_json({"ok": True, "path": str(save_path)})

    def _stream_events(self):
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

        q: queue.Queue = queue.Queue(maxsize=5)
        with _sse_lock:
            _sse_subscribers.append(q)

        try:
            with _dash_lock:
                telemetry = _dash_state["telemetry"]
            if telemetry:
                self.wfile.write(("data: " + json.dumps(telemetry) + "\n\n").encode("utf-8"))
                self.wfile.flush()
            while True:
                try:
                    payload = q.get(timeout=30)
                except queue.Empty:
                    payload = ": ping\n\n"
                self.wfile.write(payload.encode("utf-8"))
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            with _sse_lock:
                if q in _sse_subscribers:
                    _sse_subscribers.remove(q)


def _run_web(web_port: int):
    server = ThreadingHTTPServer(("0.0.0.0", web_port), DashboardHandler)
    server.serve_forever()


def decode_image_b64(image_b64):
    raw = base64.b64decode(image_b64)
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    return np.array(img, dtype=np.uint8)


def normalize_images_b64(req: dict) -> dict:
    images = req.get("images")
    if not isinstance(images, dict):
        images = req.get("images_b64")
    normalized = {}
    if isinstance(images, dict):
        for key, value in images.items():
            if isinstance(value, str) and value:
                normalized[str(key)] = value
    if "ego_view" not in normalized and isinstance(req.get("image"), str):
        normalized["ego_view"] = req["image"]
    return normalized


def decode_images_b64(images_b64: dict) -> dict:
    return {key: decode_image_b64(value) for key, value in images_b64.items()}


def get_policy_video_keys(policy) -> list[str]:
    try:
        keys = list(policy.modality_configs["video"].modality_keys)
        if keys:
            return keys
    except Exception:
        pass
    return ["ego_view"]


def get_video_horizon(policy) -> int:
    try:
        return len(policy.modality_configs["video"].delta_indices)
    except Exception:
        return 1


def compute_camera_latency_ms(req: dict, model_input_timestamp: float, video_keys: list[str]) -> tuple[float | None, dict]:
    capture_timestamps = req.get("image_capture_timestamps")
    if not isinstance(capture_timestamps, dict):
        capture_timestamps = {}
    if "ego_view" not in capture_timestamps and isinstance(req.get("image_capture_timestamp"), (int, float)):
        capture_timestamps["ego_view"] = req["image_capture_timestamp"]

    per_view_ms = {}
    for view_name in video_keys:
        ts = capture_timestamps.get(view_name)
        if isinstance(ts, (int, float)):
            per_view_ms[view_name] = max(0.0, (model_input_timestamp - float(ts)) * 1000.0)
    if "ego_view" in per_view_ms:
        return per_view_ms["ego_view"], per_view_ms
    if per_view_ms:
        return max(per_view_ms.values()), per_view_ms
    return None, per_view_ms


def build_observation(frame_buffers, speed: float, language: str, video_keys: list[str], video_horizon: int):
    video = {}
    for key in video_keys:
        frames = list(frame_buffers[key])
        if not frames:
            raise ValueError(f"no frames buffered for video key: {key}")
        while len(frames) < video_horizon:
            frames.insert(0, frames[0])
        video[key] = np.stack(frames[-video_horizon:], axis=0)[np.newaxis].astype(np.uint8)

    return {
        "video": video,
        "state": {
            "speed": np.array([[[float(speed)]]], dtype=np.float32),
        },
        "language": {
            "annotation.human.action.task_description": [[language]],
        },
    }


def update_frame_buffers(frame_buffers, images_np: dict, video_keys: list[str], video_horizon: int):
    fallback = images_np.get("ego_view")
    if fallback is None:
        fallback = next(iter(images_np.values()), None)
    if fallback is None:
        raise ValueError("no decoded RGB images available")
    for key in video_keys:
        image = images_np.get(key)
        if image is None:
            image = fallback
        if key not in frame_buffers:
            frame_buffers[key] = deque(maxlen=video_horizon)
        frame_buffers[key].append(image)


def select_vel_cmd_step(vel_cmd, action_step: int) -> tuple[float, float, int, int]:
    arr = np.asarray(vel_cmd)
    if arr.ndim != 3 or arr.shape[0] < 1:
        raise ValueError(f"vel_cmd must have shape (B,T,D), got {arr.shape}")
    horizon = int(arr.shape[1])
    idx = min(max(0, int(action_step)), horizon - 1)
    step = arr[0, idx]
    vx = float(step[0]) if step.shape[0] > 0 else 0.0
    wz = float(step[2]) if step.shape[0] > 2 else 0.0
    return vx, wz, idx, horizon


def handle_client(conn, addr, policy, action_step: int):
    print(f"[GR00T] connected from {addr}")
    policy.reset()
    buf = b""
    step = 0
    active_episode_id = None
    active_prompt_revision = None
    video_keys = get_policy_video_keys(policy)
    video_horizon = get_video_horizon(policy)
    frame_buffers = defaultdict(lambda: deque(maxlen=video_horizon))

    try:
        while True:
            while b"\n" not in buf:
                chunk = conn.recv(1 << 20)
                if not chunk:
                    return
                buf += chunk
            line, buf = buf.split(b"\n", 1)

            try:
                req = json.loads(line.decode("utf-8"))
            except json.JSONDecodeError as e:
                print(f"[GR00T] JSON decode error: {e}")
                continue

            images_b64 = normalize_images_b64(req)
            if not images_b64:
                raise KeyError("image")
            images_np = decode_images_b64(images_b64)
            update_frame_buffers(frame_buffers, images_np, video_keys, video_horizon)

            episode_id = req.get("episode_id")
            if episode_id is not None and episode_id != active_episode_id:
                active_episode_id = episode_id
                policy.reset()
                step = 0
                frame_buffers = defaultdict(lambda: deque(maxlen=video_horizon))
                update_frame_buffers(frame_buffers, images_np, video_keys, video_horizon)
                print(f"[GR00T] new episode_id={active_episode_id}; policy reset")

            prompt_state = get_prompt_state()
            if prompt_state["revision"] != active_prompt_revision:
                active_prompt_revision = prompt_state["revision"]
                policy.reset()
                step = 0
                print(
                    f"[GR00T] prompt target switched: prompt_v{prompt_state['prompt_version']} "
                    f"AREA_{prompt_state['target_area']}; policy reset"
                )

            speed = float(req.get("cmd_linear", 0.0))
            model_input_timestamp = time.time()
            camera_to_model_input_ms, per_view_camera_latency_ms = compute_camera_latency_ms(
                req,
                model_input_timestamp,
                video_keys,
            )
            if step < WARMUP_STEPS:
                resp = {"linear": 0.0, "angular": 0.0, "action_step": action_step}
                conn.sendall((json.dumps(resp) + "\n").encode())
                _push_dashboard(
                    images_b64,
                    {
                        "episode_id": active_episode_id,
                        "prompt_version": prompt_state["prompt_version"],
                        "target_area": prompt_state["target_area"],
                        "vx": 0.0,
                        "wz": 0.0,
                        "speed": speed,
                        "step": step,
                        "action_step": action_step,
                        "action_horizon": 16,
                        "camera_to_model_input_ms": camera_to_model_input_ms,
                        "per_view_camera_latency_ms": per_view_camera_latency_ms,
                    },
                )
                print(f"[{step:05d}] WARMUP ({step + 1}/{WARMUP_STEPS})")
                step += 1
                continue

            obs = build_observation(
                frame_buffers,
                speed,
                prompt_state["language"],
                video_keys,
                video_horizon,
            )
            action, _ = policy.get_action(obs)
            vx, wz, selected_step, action_horizon = select_vel_cmd_step(action["vel_cmd"], action_step)
            resp = {
                "linear": vx,
                "angular": wz,
                "action_step": selected_step,
                "action_horizon": action_horizon,
            }
            conn.sendall((json.dumps(resp) + "\n").encode())

            _push_dashboard(
                images_b64,
                {
                    "episode_id": active_episode_id,
                    "prompt_version": prompt_state["prompt_version"],
                    "target_area": prompt_state["target_area"],
                    "vx": vx,
                    "wz": wz,
                    "speed": speed,
                    "step": step,
                    "action_step": selected_step,
                    "action_horizon": action_horizon,
                    "camera_to_model_input_ms": camera_to_model_input_ms,
                    "per_view_camera_latency_ms": per_view_camera_latency_ms,
                },
            )
            latency_text = (
                f"latency={camera_to_model_input_ms:.1f}ms"
                if camera_to_model_input_ms is not None
                else "latency=NA"
            )
            print(
                f"[{step:05d}] action_step={selected_step}/{action_horizon - 1} "
                f"v={vx:+.3f} w={wz:+.3f} "
                f"speed={speed:.3f} prompt_v{prompt_state['prompt_version']} "
                f"AREA_{prompt_state['target_area']} {latency_text}"
            )
            step += 1

    except (BrokenPipeError, ConnectionResetError, OSError) as e:
        print(f"[GR00T] client {addr} disconnected: {e}")
    finally:
        conn.close()
        print(f"[GR00T] connection closed: {addr}")


def main():
    parser = argparse.ArgumentParser(description="GR00T SignNav inference server")
    parser.add_argument("--model-path", required=True, help="Path to fine-tuned GR00T checkpoint")
    parser.add_argument("--port", type=int, default=LISTEN_PORT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--web-port", type=int, default=WEB_PORT)
    parser.add_argument("--prompt-version", type=int, choices=sorted(PROMPT_TEMPLATES), default=PROMPT_VERSION)
    parser.add_argument("--target-area", type=int, choices=range(1, 13), default=DEFAULT_TARGET_AREA)
    parser.add_argument("--action-step", type=int, default=ACTION_STEP_IDX,
                        help="Which step of the 16-step action horizon to execute (0-based)")
    parser.add_argument("--modality-config-path", default=str(DEFAULT_MODALITY_CONFIG))
    args = parser.parse_args()

    with _prompt_lock:
        _prompt_state["prompt_version"] = normalize_prompt_version(args.prompt_version)
        _prompt_state["target_area"] = normalize_target_area(args.target_area)
        _prompt_state["revision"] = 0
    print_prompt_state("[PROMPT INIT]")

    load_modality_config(args.modality_config_path)
    print(f"[GR00T] Loading model from {args.model_path} ...")
    policy = Gr00tPolicy(
        embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
        model_path=args.model_path,
        device=args.device,
        strict=False,
    )
    print(
        f"[GR00T] Model loaded. video_keys={get_policy_video_keys(policy)} "
        f"video_horizon={get_video_horizon(policy)} action_step={args.action_step}"
    )

    web_thread = threading.Thread(target=_run_web, args=(args.web_port,), daemon=True)
    web_thread.start()
    print(f"[GR00T] Dashboard at http://0.0.0.0:{args.web_port}")

    kb_thread = threading.Thread(target=_keyboard_listener, daemon=True)
    kb_thread.start()

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("0.0.0.0", args.port))
    server.listen(1)
    print(f"[GR00T] Listening on port {args.port} ...")

    try:
        while True:
            conn, addr = server.accept()
            handle_client(conn, addr, policy, args.action_step)
    except KeyboardInterrupt:
        print("\n[GR00T] Shutting down.")
    finally:
        server.close()


if __name__ == "__main__":
    main()
