"""
GR00T Real-Robot / Isaac-Sim Inference Server

Pairs with the C++ client (uon_amr_gr00t).
For Isaac Sim, script/1_start_isaacsim_and_client.sh starts an Isaac Sim
observation server and a lightweight local client. The client sends observations
to this server and forwards returned velocity commands to the ROS2 bridge.

Protocol (newline-delimited JSON over TCP):
  Request  (client → Python):
    { "image":        "<base64 JPEG>",
      "images":       {"ego_view": "...", "left_view": "...", "right_view": "..."},
      "amcl_x":       <float>,
      "amcl_y":       <float>,
      "amcl_yaw":     <float>,
      "odom_vlin":    <float>,
      "odom_vang":    <float>,
      "cmd_linear":   <float>,
      "cmd_angular":  <float> }

  Response (Python → C++):
    { "linear": <float>, "angular": <float> }

Usage:
  python gr00t_inference_server.py \
      --model-path /path/to/checkpoint \
      [--port 5000] [--device cuda:0] [--action-step 1] [--web-port 8080]

  Runtime: press 1 + Enter or 2 + Enter to switch task (goal + language)
"""

import argparse
import base64
import io
import json
import math
import queue
import socket
import threading

import numpy as np
from PIL import Image

from gr00t.policy.gr00t_policy import Gr00tPolicy
from gr00t.data.embodiment_tags import EmbodimentTag


# ── Configuration defaults ────────────────────────────────────────────────────

LISTEN_PORT      = 5000
WEB_PORT         = 8080
INFERENCE_HZ     = 10          # matches C++ INFER_PERIOD_MS = 100
ACTION_STEP_IDX  = 1
N_ROUTE_SEGMENTS = 10          # route = 10 segments × 4 values = 40 floats

ARRIVAL_DIST_M   = 0.20        # stop when closer than this
ARRIVAL_STOP_STEPS = 10        # consecutive steps to confirm arrival

TASKS = {
    1: {
        "language": "Go to marker_1",
        "goal_x":   4.29,
        "goal_y":   0.35,
    },
    2: {
        "language": "Go to marker_5",
        "goal_x":   4.03,
        "goal_y":   11.29,
    },
}

WARMUP_STEPS         = 12      # Isaac Sim client와 동일 (6Hz × 2s)
SPEED_SMOOTHING_ALPHA = 0.3    # EMA factor (Isaac Sim client와 동일)
CMD_SMOOTHING_ALPHA   = 0.4    # EMA for output vx/wz (낮을수록 더 smooth)


# ── Shared task state ────────────────────────────────────────────────────────

_task_lock    = threading.Lock()
_current_task = [1]   # mutable list so threads can update in-place


# ── Shared dashboard state ────────────────────────────────────────────────────

_dash_lock  = threading.Lock()
_dash_state = {
    "image_b64": None,   # latest JPEG as base64 string
    "images_b64": {},    # latest multiview JPEGs keyed by modality name
    "telemetry": {},     # latest telemetry dict
}
_sse_subscribers: list[queue.Queue] = []
_sse_lock = threading.Lock()
DASHBOARD_VIEW_ORDER = ["left_view", "ego_view", "right_view"]


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


# ── Web dashboard ─────────────────────────────────────────────────────────────

_DASHBOARD_HTML = """\
<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<title>GR00T Dashboard</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: #0f0f13; color: #e0e0e0; font-family: 'Courier New', monospace;
         display: flex; flex-direction: column; align-items: center; padding: 10px; gap: 10px; overflow-x: hidden; }
  h1 { color: #76c442; font-size: 1.18rem; letter-spacing: 2px; line-height: 1.1; }
  .grid { display: flex; flex-direction: column; gap: 10px; width: 100%; max-width: none; }
  .camera-strip { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px; align-items: start; width: 100%; }
  .camera-strip.single { grid-template-columns: minmax(0, min(100%, 1500px)); justify-content: center; }
  .camera-strip.single .side-card { display: none; }
  .camera-card { background: #111; border: 1px solid #333; border-radius: 8px; overflow: hidden; align-self: start; }
  .camera-card img { width: 100%; height: auto; object-fit: contain; display: block; background: #050507; }
  .camera-strip.single .camera-card img { max-height: calc(100vh - 255px); }
  .cam-label { color: #9a9a9a; font-size: 0.68rem; text-transform: uppercase; padding: 6px 8px; border-bottom: 1px solid #25252c; }
  .telemetry { display: grid; grid-template-columns: 1fr 1fr 1.25fr auto; gap: 10px; align-items: stretch; }
  .panel { background: #1a1a22; border-radius: 8px; border: 1px solid #333; padding: 12px;
           display: flex; flex-direction: column; gap: 8px; }
  .row { display: flex; justify-content: space-between; align-items: center; }
  .label { color: #888; font-size: 0.75rem; text-transform: uppercase; }
  .value { font-size: 1rem; font-weight: bold; color: #fff; }
  .value.green { color: #76c442; }
  .value.red   { color: #e05252; }
  .bar-wrap { background: #111; border-radius: 4px; height: 9px; overflow: hidden; position: relative; }
  .bar       { height: 100%; border-radius: 4px; transition: width 0.1s; }
  .bar.pos   { background: #76c442; }
  .bar.neg   { background: #e05252; position: absolute; right: 0; }
  .bar-center { position: absolute; left: 50%; width: 2px; height: 100%; background: #555; transform: translateX(-50%); }
  .viz-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
  .viz-card { background: #111; border: 1px solid #333; border-radius: 8px; padding: 8px; min-width: 0; }
  .viz-card canvas { width: 100%; aspect-ratio: 1 / 1; display: block; border: 0; border-radius: 6px; background: #111; }
  .status { font-size: 0.8rem; color: #555; }
  .status.connected { color: #76c442; }
  @media (max-width: 1200px) {
    .telemetry { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  }
  @media (max-width: 860px) {
    .camera-strip { grid-template-columns: 1fr; }
    .camera-strip.multiview .side-card { display: block; }
    .telemetry { grid-template-columns: 1fr; }
  }
</style>
</head>
<body>
<h1>&#9632; GR00T VLA Dashboard</h1>
<div class="grid">
  <div class="camera-strip multiview" id="camera_strip">
    <div class="camera-card side-card" id="card_left">
      <div class="cam-label">Left View</div>
      <img id="cam_left" src="/image/left_view" alt="left view">
    </div>
    <div class="camera-card ego-card" id="card_ego">
      <div class="cam-label">Ego View</div>
      <img id="cam_ego" src="/image/ego_view" alt="ego view">
    </div>
    <div class="camera-card side-card" id="card_right">
      <div class="cam-label">Right View</div>
      <img id="cam_right" src="/image/right_view" alt="right view">
    </div>
  </div>
  <div class="telemetry">
    <div class="panel">
      <div class="row"><span class="label">AMCL X</span><span class="value" id="px">—</span></div>
      <div class="row"><span class="label">AMCL Y</span><span class="value" id="py">—</span></div>
      <div class="row"><span class="label">Yaw</span>  <span class="value" id="yaw">—</span></div>
      <div class="row"><span class="label">Dist to Goal</span><span class="value green" id="dist">—</span></div>
      <div class="row"><span class="label">Cam→Input</span><span class="value" id="latency">—</span></div>
    </div>
    <div class="panel">
      <div class="label">Linear cmd (m/s)</div>
      <div class="row"><span class="value" id="vx">—</span></div>
      <div class="bar-wrap"><div class="bar-center"></div><div class="bar" id="vx_bar"></div></div>
      <div class="label" style="margin-top:6px">Angular cmd (rad/s)</div>
      <div class="row"><span class="value" id="wz">—</span></div>
      <div class="bar-wrap"><div class="bar-center"></div><div class="bar" id="wz_bar"></div></div>
    </div>
    <div class="panel">
      <div class="row"><span class="label">Speed</span><span class="value" id="spd">—</span></div>
      <div class="row"><span class="label">Step</span> <span class="value" id="step">—</span></div>
      <div class="row"><span class="label">Goal</span> <span class="value green" id="goal">—</span></div>
      <div class="row"><span class="label">Goal Heading</span><span class="value" id="gh_deg">—</span></div>
    </div>
    <div class="panel">
      <div class="viz-grid">
        <div class="viz-card"><canvas id="compass" width="160" height="160"></canvas></div>
        <div class="viz-card"><canvas id="map" width="160" height="160"></canvas></div>
      </div>
    </div>
    <div class="row" style="align-self:center"><span class="status" id="conn_status">● waiting…</span></div>
  </div>
</div>
<script>
const MAX_TRAIL = 300;
let trail = [];
let goal  = null;
let cameraMode = "multiview";

function setCameraMode(mode, views) {
  const hasMultiview = mode === "multiview" || (
    Array.isArray(views) &&
    views.includes("left_view") &&
    views.includes("right_view")
  );
  cameraMode = hasMultiview ? "multiview" : "single";
  document.getElementById("camera_strip").className = "camera-strip " + cameraMode;
}

function barUpdate(id, val, maxVal) {
  const bar = document.getElementById(id);
  const pct = Math.min(Math.abs(val) / maxVal * 50, 50);
  bar.style.width  = pct + "%";
  if (val >= 0) {
    bar.className = "bar pos";
    bar.style.left  = "50%";
    bar.style.right = "";
  } else {
    bar.className = "bar neg";
    bar.style.right = "50%";
    bar.style.left  = "";
  }
}

function drawCompass(cos_h, sin_h) {
  const canvas = document.getElementById("compass");
  const ctx = canvas.getContext("2d");
  const cx = canvas.width / 2, cy = canvas.height / 2;
  const r = Math.min(canvas.width, canvas.height) * 0.42;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  // circle
  ctx.strokeStyle = "#333"; ctx.lineWidth = 1.5;
  ctx.beginPath(); ctx.arc(cx, cy, r, 0, 2*Math.PI); ctx.stroke();
  // cardinal labels
  ctx.fillStyle = "#555"; ctx.font = "16px Courier New"; ctx.textAlign = "center";
  ctx.fillText("F", cx, cy - r + 18);
  ctx.fillText("B", cx, cy + r - 6);
  ctx.fillText("L", cx - r + 14, cy + 5);
  ctx.fillText("R", cx + r - 14, cy + 5);
  // arrow (robot frame: x=forward, y=left → canvas: up=forward, right=right)
  // cos_h = cos(angle_to_goal), sin_h = sin(angle_to_goal) in robot frame
  // angle_to_goal>0 means goal is to the left
  const ax =  sin_h * r * 0.85;   // left in robot = left on canvas (negate for screen x→right)
  const ay = -cos_h * r * 0.85;   // forward in robot = up on screen
  ctx.strokeStyle = "#76c442"; ctx.lineWidth = 4;
  ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(cx - ax, cy + ay); ctx.stroke();
  // arrowhead
  const angle = Math.atan2(ay, -ax);
  ctx.beginPath();
  ctx.moveTo(cx - ax, cy + ay);
  ctx.lineTo(cx - ax + Math.cos(angle - 2.5) * 16, cy + ay + Math.sin(angle - 2.5) * 16);
  ctx.lineTo(cx - ax + Math.cos(angle + 2.5) * 16, cy + ay + Math.sin(angle + 2.5) * 16);
  ctx.closePath(); ctx.fillStyle = "#76c442"; ctx.fill();
}

function drawMap() {
  const canvas = document.getElementById("map");
  const ctx    = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = "#111";
  ctx.fillRect(0, 0, W, H);
  if (trail.length < 2) return;

  // auto-scale
  const xs = trail.map(p => p[0]);
  const ys = trail.map(p => p[1]);
  if (goal) { xs.push(goal[0]); ys.push(goal[1]); }
  const xmin = Math.min(...xs), xmax = Math.max(...xs);
  const ymin = Math.min(...ys), ymax = Math.max(...ys);
  const pad  = 20;
  const scaleX = (W - 2*pad) / (xmax - xmin + 1e-6);
  const scaleY = (H - 2*pad) / (ymax - ymin + 1e-6);
  const sc = Math.min(scaleX, scaleY);
  const tx = p => pad + (p - xmin) * sc;
  const ty = p => H - pad - (p - ymin) * sc;

  // trail
  ctx.strokeStyle = "#445544";
  ctx.lineWidth   = 1.5;
  ctx.beginPath();
  trail.forEach((p, i) => i === 0 ? ctx.moveTo(tx(p[0]), ty(p[1])) : ctx.lineTo(tx(p[0]), ty(p[1])));
  ctx.stroke();

  // goal
  if (goal) {
    ctx.fillStyle = "#e05252";
    ctx.beginPath();
    ctx.arc(tx(goal[0]), ty(goal[1]), 6, 0, 2*Math.PI);
    ctx.fill();
    ctx.fillStyle = "#e05252";
    ctx.font = "10px Courier New";
    ctx.fillText("GOAL", tx(goal[0]) + 8, ty(goal[1]) + 4);
  }

  // robot (last position)
  const last = trail[trail.length - 1];
  ctx.fillStyle = "#76c442";
  ctx.beginPath();
  ctx.arc(tx(last[0]), ty(last[1]), 5, 0, 2*Math.PI);
  ctx.fill();
}

// SSE telemetry
const es = new EventSource("/stream");
es.onopen    = () => { document.getElementById("conn_status").textContent = "● connected"; document.getElementById("conn_status").className = "status connected"; };
es.onerror   = () => { document.getElementById("conn_status").textContent = "● disconnected"; document.getElementById("conn_status").className = "status"; };
es.onmessage = e => {
  const d = JSON.parse(e.data);
  setCameraMode(d.camera_mode, d.views);
  document.getElementById("px"  ).textContent = d.robot_x   !== undefined ? d.robot_x.toFixed(3)   : "—";
  document.getElementById("py"  ).textContent = d.robot_y   !== undefined ? d.robot_y.toFixed(3)   : "—";
  document.getElementById("yaw" ).textContent = d.robot_yaw !== undefined ? (d.robot_yaw * 180/Math.PI).toFixed(1) + "°" : "—";
  document.getElementById("dist").textContent = d.dist      !== undefined ? d.dist.toFixed(3) + " m" : "—";
  document.getElementById("latency").textContent = d.camera_to_model_input_ms !== undefined ? d.camera_to_model_input_ms.toFixed(1) + " ms" : "—";
  document.getElementById("vx"  ).textContent = d.vx        !== undefined ? d.vx.toFixed(4)  : "—";
  document.getElementById("wz"  ).textContent = d.wz        !== undefined ? d.wz.toFixed(4)  : "—";
  document.getElementById("spd" ).textContent = d.speed     !== undefined ? d.speed.toFixed(3) : "—";
  document.getElementById("step").textContent = d.step      !== undefined ? d.step  : "—";
  if (d.goal_x !== undefined) {
    document.getElementById("goal").textContent = "(" + d.goal_x.toFixed(2) + ", " + d.goal_y.toFixed(2) + ")";
    goal = [d.goal_x, d.goal_y];
  }
  if (d.vx !== undefined) barUpdate("vx_bar", d.vx, 1.0);
  if (d.wz !== undefined) barUpdate("wz_bar", d.wz, 1.5);
  if (d.gh_cos !== undefined) {
    const deg = Math.atan2(d.gh_sin, d.gh_cos) * 180 / Math.PI;
    document.getElementById("gh_deg").textContent = deg.toFixed(1) + "°";
    drawCompass(d.gh_cos, d.gh_sin);
  }
  if (d.robot_x !== undefined) {
    trail.push([d.robot_x, d.robot_y]);
    if (trail.length > MAX_TRAIL) trail.shift();
    drawMap();
  }
};

// image polling
function refreshImage() {
  const t = Date.now();
  document.getElementById("cam_ego").src = "/image/ego_view?" + t;
  if (cameraMode === "multiview") {
    document.getElementById("cam_left").src = "/image/left_view?" + t;
    document.getElementById("cam_right").src = "/image/right_view?" + t;
  }
}
setInterval(refreshImage, 100);
</script>
</body>
</html>
"""


def _make_flask_app():
    from flask import Flask, Response

    app = Flask(__name__)
    app.logger.disabled = True
    import logging
    log = logging.getLogger("werkzeug")
    log.setLevel(logging.ERROR)

    @app.route("/")
    def index():
        return _DASHBOARD_HTML, 200, {"Content-Type": "text/html; charset=utf-8"}

    def image_response_for_view(view_name: str = "ego_view"):
        with _dash_lock:
            images = dict(_dash_state.get("images_b64") or {})
            b64 = images.get(view_name) or _dash_state["image_b64"]
        if b64 is None:
            # return 1×1 grey placeholder
            placeholder = (
                b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
                b"\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t"
                b"\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a"
                b"\x1f\x1e\x1d\x1a\x1c\x1c $.' \",#\x1c\x1c(7),01444\x1f'9=82<.342\x1e"
                b"\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00\xff\xc4\x00"
                b"\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00\x00\x00\x00\x00"
                b"\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\xff\xda\x00"
                b"\x08\x01\x01\x00\x00?\x00\xf5\x07\xff\xd9"
            )
            return Response(placeholder, mimetype="image/jpeg")
        raw = base64.b64decode(b64)
        return Response(raw, mimetype="image/jpeg")

    @app.route("/image")
    def image():
        return image_response_for_view("ego_view")

    @app.route("/image/<view_name>")
    def image_view(view_name):
        return image_response_for_view(view_name)

    @app.route("/stream")
    def stream():
        q: queue.Queue = queue.Queue(maxsize=5)
        with _sse_lock:
            _sse_subscribers.append(q)

        def generate():
            try:
                # send current state immediately so page isn't blank
                with _dash_lock:
                    t = _dash_state["telemetry"]
                if t:
                    yield "data: " + json.dumps(t) + "\n\n"
                while True:
                    try:
                        yield q.get(timeout=30)
                    except queue.Empty:
                        yield ": ping\n\n"
            finally:
                with _sse_lock:
                    if q in _sse_subscribers:
                        _sse_subscribers.remove(q)

        return Response(generate(), mimetype="text/event-stream",
                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    return app


def _run_web(web_port: int):
    app = _make_flask_app()
    app.run(host="0.0.0.0", port=web_port, threaded=True, use_reloader=False)


def _keyboard_listener():
    """Read '1' or '2' from stdin and switch the active task."""
    print("[GR00T] Keyboard: press 1 or 2 + Enter to switch task")
    while True:
        try:
            key = input().strip()
        except EOFError:
            break
        if key in ("1", "2"):
            task_id = int(key)
            with _task_lock:
                _current_task[0] = task_id
            t = TASKS[task_id]
            print(
                f"\n[TASK SWITCH] → task {task_id}  "
                f"goal=({t['goal_x']:.2f},{t['goal_y']:.2f})  "
                f"lang={t['language'][:60]}...\n"
            )


# ── Geometry helpers ──────────────────────────────────────────────────────────

def _to_robot_frame(wx, wy, robot_x, robot_y, robot_yaw):
    cos_r = math.cos(robot_yaw)
    sin_r = math.sin(robot_yaw)
    dx, dy = wx - robot_x, wy - robot_y
    return cos_r * dx + sin_r * dy, -sin_r * dx + cos_r * dy


def compute_route_segments(robot_x, robot_y, robot_yaw, goal_x, goal_y):
    """Straight-line path → N_ROUTE_SEGMENTS in robot frame, shape (40,)."""
    segments = []
    for i in range(N_ROUTE_SEGMENTS):
        t_s = i       / N_ROUTE_SEGMENTS
        t_e = (i + 1) / N_ROUTE_SEGMENTS
        sx, sy = _to_robot_frame(
            robot_x + t_s * (goal_x - robot_x),
            robot_y + t_s * (goal_y - robot_y),
            robot_x, robot_y, robot_yaw,
        )
        ex, ey = _to_robot_frame(
            robot_x + t_e * (goal_x - robot_x),
            robot_y + t_e * (goal_y - robot_y),
            robot_x, robot_y, robot_yaw,
        )
        segments.extend([sx, sy, ex, ey])
    return np.array(segments, dtype=np.float32)


def compute_goal_heading(robot_x, robot_y, robot_yaw, goal_x, goal_y):
    """Direction robot→goal in robot frame as (cos θ, sin θ), shape (2,)."""
    angle_world = math.atan2(goal_y - robot_y, goal_x - robot_x)
    angle_local = math.atan2(
        math.sin(angle_world - robot_yaw),
        math.cos(angle_world - robot_yaw),
    )
    return np.array([math.cos(angle_local), math.sin(angle_local)], dtype=np.float32)


# ── Observation builder ───────────────────────────────────────────────────────

def normalize_images_b64(req: dict) -> dict:
    images = req.get("images")
    if not isinstance(images, dict):
        images = req.get("images_b64")
    normalized = {}
    if isinstance(images, dict):
        for key, value in images.items():
            if isinstance(value, str) and value:
                normalized[str(key)] = value

    # Accept training/export naming variants while keeping model modality keys.
    if "ego_view" not in normalized:
        for alias in ("front_view", "front"):
            if alias in normalized:
                normalized["ego_view"] = normalized[alias]
                break
    if "left_view" not in normalized and "left" in normalized:
        normalized["left_view"] = normalized["left"]
    if "right_view" not in normalized and "right" in normalized:
        normalized["right_view"] = normalized["right"]
    if "ego_view" not in normalized and isinstance(req.get("image"), str):
        normalized["ego_view"] = req["image"]
    return normalized


def decode_images_b64(images_b64: dict) -> dict:
    return {key: decode_image_b64(value) for key, value in images_b64.items()}


def build_observation(images_np, speed, route, goal_heading, language="", video_keys=None):
    """Pack into GR00T PointNav observation format."""
    if isinstance(images_np, np.ndarray):
        images = {"ego_view": images_np}
    else:
        images = dict(images_np)

    requested_video_keys = list(video_keys or images.keys() or ["ego_view"])
    video = {}
    fallback = images.get("ego_view")
    if fallback is None:
        fallback = next(iter(images.values()), None)
    if fallback is None:
        raise ValueError("no decoded RGB images available")

    for key in requested_video_keys:
        image = images.get(key)
        if image is None and key == "ego_view":
            image = images.get("front_view")
            if image is None:
                image = images.get("front")
        if image is None and key == "left_view":
            image = images.get("left")
        if image is None and key == "right_view":
            image = images.get("right")
        if image is None:
            image = fallback
        video[key] = image[np.newaxis, np.newaxis].astype(np.uint8)  # (1,1,H,W,3)

    return {
        "video": video,
        "state": {
            "speed":        np.array([[[speed]]], dtype=np.float32),         # (1,1,1)
            "route":        route[np.newaxis, np.newaxis],                   # (1,1,40)
            "goal_heading": goal_heading[np.newaxis, np.newaxis],            # (1,1,2)
        },
        "language": {
            "annotation.human.action.task_description": [[language]],
        },
    }


def decode_image_b64(image_b64):
    raw = base64.b64decode(image_b64)
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    return np.array(img, dtype=np.uint8)


def get_policy_video_keys(policy) -> list[str]:
    try:
        video_cfg = policy.modality_configs["video"]
        keys = list(video_cfg.modality_keys)
        if keys:
            return keys
    except Exception:
        pass
    return ["ego_view"]


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
        representative_ms = per_view_ms["ego_view"]
    elif per_view_ms:
        representative_ms = max(per_view_ms.values())
    else:
        representative_ms = None
    return representative_ms, per_view_ms


# ── Client handler ────────────────────────────────────────────────────────────

def handle_client(conn, addr, policy, action_step):
    print(f"[GR00T] connected from {addr}")

    policy.reset()

    buf             = b""
    step            = 0
    speed_ema       = 0.0
    # cmd_v_ema       = 0.0
    # cmd_w_ema       = 0.0
    prev_x          = None
    prev_y          = None
    prev_yaw        = None
    prev_time       = None
    arrival_counter = 0
    arrived         = False

    with _task_lock:
        prev_task_id = _current_task[0]
    task      = TASKS[prev_task_id]
    goal_x    = task["goal_x"]
    goal_y    = task["goal_y"]
    language  = task["language"]
    print(f"[GR00T] initial task={prev_task_id}  goal=({goal_x:.2f},{goal_y:.2f})")

    import time as _time

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
            video_keys = get_policy_video_keys(policy)
            now       = _time.time()
            robot_x   = req["amcl_x"]
            robot_y   = req["amcl_y"]
            robot_yaw = req["amcl_yaw"]
            camera_mode = req.get("camera_mode", "multiview" if len(images_b64) > 1 else "single")
            camera_views = req.get("views", list(images_b64.keys()))
            model_input_timestamp = now
            camera_to_model_input_ms, per_view_camera_latency_ms = compute_camera_latency_ms(
                req,
                model_input_timestamp,
                video_keys,
            )

            # Task switch detection
            with _task_lock:
                cur_task_id = _current_task[0]
            if cur_task_id != prev_task_id:
                prev_task_id    = cur_task_id
                task            = TASKS[cur_task_id]
                goal_x          = task["goal_x"]
                goal_y          = task["goal_y"]
                language        = task["language"]
                arrived         = False
                arrival_counter = 0
                policy.reset()
                print(f"[GR00T] task switched → {cur_task_id}  goal=({goal_x:.2f},{goal_y:.2f})")

            # Speed: AMCL pose 차분 + EMA 스무딩 (Isaac Sim client와 동일)
            if prev_x is not None and prev_time is not None:
                dt = now - prev_time
                if dt > 1e-4:
                    dx_w = robot_x - prev_x
                    dy_w = robot_y - prev_y
                    cos_r = math.cos(prev_yaw)
                    sin_r = math.sin(prev_yaw)
                    fwd = cos_r * dx_w + sin_r * dy_w
                    raw_speed = fwd / dt
                    speed_ema = (
                        SPEED_SMOOTHING_ALPHA * raw_speed
                        + (1.0 - SPEED_SMOOTHING_ALPHA) * speed_ema
                    )
            prev_x, prev_y, prev_yaw, prev_time = robot_x, robot_y, robot_yaw, now

            goal_distance = math.hypot(goal_x - robot_x, goal_y - robot_y)

            # Warmup: 처음 N스텝은 stop 명령만 (Isaac Sim client와 동일)
            if step < WARMUP_STEPS:
                resp = {"linear": 0.0, "angular": 0.0}
                conn.sendall((json.dumps(resp) + "\n").encode())
                goal_heading = compute_goal_heading(robot_x, robot_y, robot_yaw, goal_x, goal_y)
                _push_dashboard(images_b64, {
                    "robot_x": robot_x, "robot_y": robot_y, "robot_yaw": robot_yaw,
                    "dist": goal_distance, "vx": 0.0, "wz": 0.0,
                    "speed": speed_ema, "step": step, "goal_x": goal_x, "goal_y": goal_y,
                    "gh_cos": float(goal_heading[0]), "gh_sin": float(goal_heading[1]),
                    "camera_mode": camera_mode, "views": camera_views,
                    "camera_to_model_input_ms": camera_to_model_input_ms,
                    "per_view_camera_latency_ms": per_view_camera_latency_ms,
                })
                print(f"[{step:05d}] WARMUP ({step+1}/{WARMUP_STEPS})")
                step += 1
                continue

            route        = compute_route_segments(robot_x, robot_y, robot_yaw, goal_x, goal_y)
            goal_heading = compute_goal_heading(robot_x, robot_y, robot_yaw, goal_x, goal_y)

            model_input_timestamp = _time.time()
            camera_to_model_input_ms, per_view_camera_latency_ms = compute_camera_latency_ms(
                req,
                model_input_timestamp,
                video_keys,
            )
            obs = build_observation(
                images_np,
                speed_ema,
                route,
                goal_heading,
                language,
                video_keys=video_keys,
            )

            action, _ = policy.get_action(obs)
            vel_cmd = action["vel_cmd"][0, action_step]   # (3,) [vx, vy, wz]

            vx = float(vel_cmd[0])
            wz = float(vel_cmd[2])

            # Arrival detection: distance + model stop + consecutive counter
            if not arrived:
                model_stop = abs(vx) < 0.03 and abs(wz) < 0.03
                if goal_distance < ARRIVAL_DIST_M and model_stop:
                    arrival_counter += 1
                else:
                    arrival_counter = 0
                if arrival_counter >= ARRIVAL_STOP_STEPS:
                    arrived = True
                    print(f"\n{'='*60}")
                    print(f"  ARRIVED  dist={goal_distance:.3f}m  step={step:05d}")
                    print(f"{'='*60}\n")

            if arrived:
                resp = {"linear": 0.0, "angular": 0.0}
                conn.sendall((json.dumps(resp) + "\n").encode())
                _push_dashboard(images_b64, {
                    "robot_x": robot_x, "robot_y": robot_y, "robot_yaw": robot_yaw,
                    "dist": goal_distance, "vx": 0.0, "wz": 0.0,
                    "speed": speed_ema, "step": step, "goal_x": goal_x, "goal_y": goal_y,
                    "gh_cos": float(goal_heading[0]), "gh_sin": float(goal_heading[1]),
                    "camera_mode": camera_mode, "views": camera_views,
                    "camera_to_model_input_ms": camera_to_model_input_ms,
                    "per_view_camera_latency_ms": per_view_camera_latency_ms,
                })
                step += 1
                continue

            # cmd_v_ema = CMD_SMOOTHING_ALPHA * vx + (1.0 - CMD_SMOOTHING_ALPHA) * cmd_v_ema
            # cmd_w_ema = CMD_SMOOTHING_ALPHA * wz + (1.0 - CMD_SMOOTHING_ALPHA) * cmd_w_ema

            resp = {"linear": vx, "angular": wz}
            conn.sendall((json.dumps(resp) + "\n").encode())

            _push_dashboard(images_b64, {
                "robot_x": robot_x, "robot_y": robot_y, "robot_yaw": robot_yaw,
                "dist": goal_distance, "vx": vx, "wz": wz,
                "speed": speed_ema, "step": step, "goal_x": goal_x, "goal_y": goal_y,
                "gh_cos": float(goal_heading[0]), "gh_sin": float(goal_heading[1]),
                "camera_mode": camera_mode, "views": camera_views,
                "camera_to_model_input_ms": camera_to_model_input_ms,
                "per_view_camera_latency_ms": per_view_camera_latency_ms,
            })

            latency_text = (
                f"latency={camera_to_model_input_ms:.1f}ms"
                if camera_to_model_input_ms is not None
                else "latency=NA"
            )
            print(
                f"[{step:05d}] dist={goal_distance:.2f}m  "
                f"v={vx:+.3f}  w={wz:+.3f}  "
                f"raw=({vx:+.3f},{wz:+.3f})  speed_ema={speed_ema:.3f}  "
                f"{latency_text}"
            )
            step += 1

    except (BrokenPipeError, ConnectionResetError, OSError) as e:
        print(f"[GR00T] client {addr} disconnected: {e}")
    finally:
        conn.close()
        print(f"[GR00T] connection closed: {addr}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GR00T real-robot inference server")
    parser.add_argument("--model-path",  required=True,
                        help="Path to fine-tuned GR00T checkpoint")
    parser.add_argument("--port",       type=int,   default=LISTEN_PORT)
    parser.add_argument("--device",     default="cuda:0")
    parser.add_argument("--action-step", type=int,  default=ACTION_STEP_IDX,
                        help="Which step of the action horizon to execute (0-based)")
    parser.add_argument("--web-port",   type=int,   default=WEB_PORT,
                        help="Port for the web dashboard (default: 8080)")
    args = parser.parse_args()

    print(f"[GR00T] Loading model from {args.model_path} ...")
    policy = Gr00tPolicy(
        embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
        model_path=args.model_path,
        device=args.device,
        strict=False,
    )
    t0 = TASKS[1]
    print(f"[GR00T] Model loaded. Starting task=1  goal=({t0['goal_x']:.2f},{t0['goal_y']:.2f})")

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


'''
uv run python gr00t_inference_server.py \
    --model-path /home/lds/model/gr00t-finetune+real_v2_lerobot--20260507/checkpoint-60000 \
    --port 5000 --device cuda:0 --action-step 1 --web-port 9090
'''
