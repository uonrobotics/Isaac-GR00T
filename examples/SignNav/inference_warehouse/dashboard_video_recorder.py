"""Server-side video recording for the SignNav GR00T dashboard.

The inference servers already have every camera image and telemetry value used
by the browser dashboard.  This module renders those values into a fixed-size
RGB frame and streams the frames to an ffmpeg child process.  Finished videos
remain temporary until the dashboard explicitly saves or discards them.
"""

from __future__ import annotations

import atexit
import base64
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
import io
import json
import math
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import time
from typing import Any
import uuid

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720
DEFAULT_FPS = 5.0
DEFAULT_STOP_LINEAR_THRESHOLD = 0.03
DEFAULT_STOP_ANGULAR_THRESHOLD = 0.03
DEFAULT_STOP_HOLD_SECONDS = 1.5

_FFMPEG_CLOSE_TIMEOUT_SECONDS = 30.0
_MODULE_DIR = Path(__file__).resolve().parent

_BACKGROUND = (16, 17, 20)
_CARD_BACKGROUND = (25, 26, 32)
_CAMERA_BACKGROUND = (5, 5, 7)
_BORDER = (48, 48, 56)
_MUTED = (141, 144, 152)
_FOREGROUND = (232, 232, 232)
_ACCENT = (127, 211, 107)
_NEGATIVE = (227, 96, 96)

_VIEW_LABELS = {
    "ego_view": "EGO VIEW / MODEL INPUT",
    "sign_grounding": "SIGN GROUNDING BBOX / E1-C",
    "target_bbox": "SELECTED BBOX / SAM3 + QWEN3",
    "sign_crop": "SIGN CROP / GR00T INPUT",
    "segmented_ego_view": "SEGMENTED EGO VIEW / SAM3",
}
_VIEW_ORDER = tuple(_VIEW_LABELS)

_MEASURED_SCALAR_PAIRS = (
    ("measured_linear_speed", "measured_angular_speed"),
    ("robot_linear_speed", "robot_angular_speed"),
    ("actual_linear_speed", "actual_angular_speed"),
    ("measured_vx", "measured_wz"),
)
_MEASURED_VECTOR_PAIRS = (
    ("robot_linear_velocity", "robot_angular_velocity"),
    ("measured_linear_velocity", "measured_angular_velocity"),
)


def _load_font(size: int, bold: bool = False):
    names = (
        "DejaVuSansMono-Bold.ttf" if bold else "DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    )
    for name in names:
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _vector_linear_speed(value: Any) -> float | None:
    try:
        values = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if values.size == 0 or not np.all(np.isfinite(values)):
        return None
    if values.size == 1:
        return abs(float(values[0]))
    return float(np.linalg.norm(values[:2]))


def _vector_angular_speed(value: Any) -> float | None:
    try:
        values = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if values.size == 0 or not np.all(np.isfinite(values)):
        return None
    if values.size >= 3:
        return abs(float(values[2]))
    return abs(float(values[-1]))


def _select_motion_speeds(telemetry: Mapping[str, Any]) -> tuple[float, float, str]:
    """Return linear/angular speed, preferring a complete measured pair."""

    for linear_key, angular_key in _MEASURED_SCALAR_PAIRS:
        linear = _finite_float(telemetry.get(linear_key))
        angular = _finite_float(telemetry.get(angular_key))
        if linear is not None and angular is not None:
            return linear, angular, "measured"

    for linear_key, angular_key in _MEASURED_VECTOR_PAIRS:
        linear = _vector_linear_speed(telemetry.get(linear_key))
        angular = _vector_angular_speed(telemetry.get(angular_key))
        if linear is not None and angular is not None:
            return linear, angular, "measured"

    for container_key in ("measured_velocity", "robot_velocity"):
        velocity = telemetry.get(container_key)
        if not isinstance(velocity, Mapping):
            continue
        linear = _finite_float(velocity.get("linear"))
        angular = _finite_float(velocity.get("angular"))
        if linear is not None and angular is not None:
            return linear, angular, "measured"
        linear = _vector_linear_speed(velocity.get("linear"))
        angular = _vector_angular_speed(velocity.get("angular"))
        if linear is not None and angular is not None:
            return linear, angular, "measured"

    linear = _finite_float(telemetry.get("vx"))
    angular = _finite_float(telemetry.get("wz"))
    return linear or 0.0, angular or 0.0, "command"


def _episode_key(episode_id: Any, session_id: Any = None) -> str:
    try:
        return json.dumps(
            {"session_id": session_id, "episode_id": episode_id},
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError):
        return repr((session_id, episode_id))


def _json_safe_episode_id(episode_id: Any) -> Any:
    try:
        json.dumps(episode_id)
    except (TypeError, ValueError):
        return str(episode_id)
    return episode_id


def _safe_filename_part(value: Any) -> str:
    text = str(value)
    safe = "".join(char if char.isalnum() or char in "-_" else "-" for char in text)
    return safe.strip("-")[:64] or "unknown"


class _RawRGBFFmpegWriter:
    """Write fixed-shape RGB frames into a temporary H.264 MP4."""

    def __init__(
        self,
        path: Path,
        width: int,
        height: int,
        fps: float,
        ffmpeg_binary: str,
    ):
        self.path = path
        self.width = width
        self.height = height
        self.fps = fps
        self.ffmpeg_binary = ffmpeg_binary
        self._process: subprocess.Popen | None = None

    def _open(self) -> None:
        command = [
            self.ffmpeg_binary,
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s:v",
            f"{self.width}x{self.height}",
            "-r",
            f"{self.fps:g}",
            "-i",
            "-",
            "-an",
            "-vcodec",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(self.path),
        ]
        try:
            self._process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(f"ffmpeg executable not found: {self.ffmpeg_binary}") from exc

    def write(self, frame: np.ndarray) -> None:
        expected_shape = (self.height, self.width, 3)
        if frame.shape != expected_shape or frame.dtype != np.uint8:
            raise ValueError(
                f"dashboard frame must be uint8 RGB {expected_shape}, got {frame.dtype} {frame.shape}"
            )
        if self._process is None:
            self._open()
        assert self._process is not None
        assert self._process.stdin is not None
        try:
            self._process.stdin.write(np.ascontiguousarray(frame).tobytes())
        except BrokenPipeError as exc:
            raise RuntimeError("ffmpeg stopped while writing dashboard video") from exc

    def close(self) -> None:
        process = self._process
        self._process = None
        if process is None:
            return
        try:
            try:
                _, stderr = process.communicate(timeout=_FFMPEG_CLOSE_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired as exc:
                process.kill()
                _, stderr = process.communicate()
                raise RuntimeError(
                    "ffmpeg did not finish dashboard video within "
                    f"{_FFMPEG_CLOSE_TIMEOUT_SECONDS:.0f}s and was killed"
                ) from exc
            if process.returncode != 0:
                message = (stderr or b"").decode("utf-8", errors="replace").strip()
                raise RuntimeError(f"ffmpeg failed to encode dashboard video: {message}")
        finally:
            self._process = None

    def abort(self) -> None:
        process = self._process
        self._process = None
        if process is None:
            return
        process.kill()
        process.communicate()


@dataclass
class _ActiveRecording:
    episode_id: Any
    session_id: Any
    episode_key: str
    recording_id: str
    temp_path: Path
    writer: Any
    started_monotonic: float
    started_wall_time: float
    frames: int = 0
    last_frame: np.ndarray | None = None
    next_frame_monotonic: float | None = None
    motion_seen: bool = False
    stationary_since: float | None = None
    linear_speed: float = 0.0
    angular_speed: float = 0.0
    speed_source: str = "command"


@dataclass
class _PendingRecording:
    episode_id: Any
    session_id: Any
    episode_key: str
    recording_id: str
    temp_path: Path
    started_monotonic: float
    ended_monotonic: float
    started_wall_time: float
    ended_wall_time: float
    frames: int
    reason: str


class DashboardVideoRecorder:
    """Compose, encode, and stage one dashboard recording per episode."""

    def __init__(
        self,
        experiment_name: str,
        *,
        recordings_dir: str | Path | None = None,
        temp_dir: str | Path | None = None,
        width: int = DEFAULT_WIDTH,
        height: int = DEFAULT_HEIGHT,
        fps: float = DEFAULT_FPS,
        stop_linear_threshold: float = DEFAULT_STOP_LINEAR_THRESHOLD,
        stop_angular_threshold: float = DEFAULT_STOP_ANGULAR_THRESHOLD,
        stop_hold_seconds: float = DEFAULT_STOP_HOLD_SECONDS,
        ffmpeg_binary: str = "ffmpeg",
        enabled: bool = True,
        clock: Callable[[], float] | None = None,
        wall_clock: Callable[[], float] | None = None,
        writer_factory: Callable[..., Any] | None = None,
    ):
        if width <= 0 or height <= 0 or width % 2 or height % 2:
            raise ValueError("dashboard video width and height must be positive even integers")
        if fps <= 0.0:
            raise ValueError("dashboard video fps must be positive")
        if stop_linear_threshold <= 0.0 or stop_angular_threshold <= 0.0:
            raise ValueError("dashboard video stop thresholds must be positive")
        if stop_hold_seconds < 0.0:
            raise ValueError("dashboard video stop hold time must not be negative")

        self.experiment_name = _safe_filename_part(experiment_name)
        if recordings_dir is None:
            recordings_dir = _MODULE_DIR / self.experiment_name / "recordings"
        self.recordings_dir = Path(recordings_dir).expanduser().absolute()
        if temp_dir is None:
            temp_dir = Path(tempfile.gettempdir()) / "signnav_dashboard_video"
        self.temp_dir = Path(temp_dir).expanduser().absolute()
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        self.stop_linear_threshold = float(stop_linear_threshold)
        self.stop_angular_threshold = float(stop_angular_threshold)
        self.stop_hold_seconds = float(stop_hold_seconds)
        self.ffmpeg_binary = ffmpeg_binary
        self.enabled = bool(enabled)
        self._clock = clock or time.monotonic
        self._wall_clock = wall_clock or time.time
        self._writer_factory = writer_factory or _RawRGBFFmpegWriter

        self._lock = threading.RLock()
        self._active: _ActiveRecording | None = None
        self._pending: dict[str, _PendingRecording] = {}
        self._finished_episode_keys: set[str] = set()
        self._last_event = "ready" if self.enabled else "disabled"
        self._last_error: str | None = None
        self._last_resolution: dict[str, Any] | None = None

        self._title_font = _load_font(18, bold=True)
        self._section_font = _load_font(13, bold=True)
        self._label_font = _load_font(12)
        self._value_font = _load_font(14, bold=True)
        self._small_font = _load_font(11)

    def capture(
        self,
        images_b64: Mapping[str, str] | str | None,
        telemetry: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Record one dashboard frame and advance the episode stop state."""

        with self._lock:
            if not self.enabled:
                return self.status()
            episode_id = telemetry.get("episode_id")
            if episode_id is None:
                self._last_event = "waiting_for_episode"
                return self.status()
            session_id = telemetry.get("session_id")

            now = self._clock()
            episode_key = _episode_key(episode_id, session_id)
            if self._active is not None and self._active.episode_key != episode_key:
                self._finish_active_locked("episode_changed", finish_time=now)

            if self._active is None:
                if episode_key in self._finished_episode_keys:
                    self._last_event = "episode_already_finished"
                    return self.status()
                try:
                    self._start_episode_locked(episode_id, session_id, episode_key, now)
                except Exception as exc:
                    self._last_error = str(exc)
                    self._last_event = "start_failed"
                    return self.status()

            assert self._active is not None
            active = self._active
            linear, angular, source = _select_motion_speeds(telemetry)
            active.linear_speed = linear
            active.angular_speed = angular
            active.speed_source = source

            stopped = (
                abs(linear) < self.stop_linear_threshold
                and abs(angular) < self.stop_angular_threshold
            )
            if not stopped:
                active.motion_seen = True
                active.stationary_since = None
            elif active.motion_seen and active.stationary_since is None:
                active.stationary_since = now

            try:
                frame = self.compose_frame(images_b64, telemetry)
                if active.last_frame is None:
                    active.writer.write(frame)
                    active.frames += 1
                    active.next_frame_monotonic = now + 1.0 / self.fps
                else:
                    self._flush_active_until_locked(active, now)
                active.last_frame = frame
                self._last_error = None
                self._last_event = "recording"
            except Exception as exc:
                self._last_error = str(exc)
                self._last_event = "capture_failed"
                self._abort_active_locked()
                return self.status()

            if (
                active.motion_seen
                and active.stationary_since is not None
                and now - active.stationary_since >= self.stop_hold_seconds
            ):
                self._finish_active_locked("robot_stopped", finish_time=now)
            return self.status()

    def compose_frame(
        self,
        images_b64: Mapping[str, str] | str | None,
        telemetry: Mapping[str, Any],
    ) -> np.ndarray:
        """Render dashboard inputs as a fixed-size uint8 RGB array."""

        if isinstance(images_b64, Mapping):
            images = {str(key): value for key, value in images_b64.items() if value}
        elif isinstance(images_b64, str) and images_b64:
            images = {"ego_view": images_b64}
        else:
            images = {}

        canvas = Image.new("RGB", (self.width, self.height), _BACKGROUND)
        draw = ImageDraw.Draw(canvas)
        margin = 12
        header_height = 42
        panel_width = min(330, max(280, self.width // 4))
        camera_width = self.width - panel_width - margin * 3
        content_top = margin + header_height
        content_height = self.height - content_top - margin

        draw.text(
            (margin, 12),
            "SignNav GR00T Dashboard",
            fill=_ACCENT,
            font=self._title_font,
        )
        self._draw_camera_grid(
            canvas,
            draw,
            images,
            (margin, content_top, camera_width, content_height),
        )
        self._draw_telemetry_panel(
            draw,
            telemetry,
            (camera_width + margin * 2, content_top, panel_width, content_height),
        )
        return np.asarray(canvas, dtype=np.uint8)

    def resolve(
        self,
        save: bool,
        episode_id: Any | None = None,
        *,
        session_id: Any | None = None,
        recording_id: str | None = None,
    ) -> dict[str, Any]:
        """Save or discard a completed temporary recording."""

        with self._lock:
            if self._active is not None:
                active_matches = recording_id is None or self._active.recording_id == recording_id
                if episode_id is not None:
                    active_matches = active_matches and self._active.episode_id == episode_id
                if session_id is not None:
                    active_matches = active_matches and self._active.session_id == session_id
                if active_matches:
                    self._finish_active_locked("user_decision", finish_time=self._clock())

            matches = list(self._pending.values())
            if recording_id is not None:
                matches = [pending for pending in matches if pending.recording_id == recording_id]
            if episode_id is not None:
                matches = [pending for pending in matches if pending.episode_id == episode_id]
            if session_id is not None:
                matches = [pending for pending in matches if pending.session_id == session_id]

            if not matches:
                return self._resolution_error_locked("no matching pending dashboard recording")
            if len(matches) > 1:
                return self._resolution_error_locked(
                    "multiple recordings match; pass episode_id or recording_id"
                )

            pending = matches[0]
            try:
                if save:
                    destination = self._recording_destination_locked(pending)
                    self.recordings_dir.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(pending.temp_path), str(destination))
                    action = "saved"
                    path: str | None = str(destination)
                else:
                    pending.temp_path.unlink(missing_ok=True)
                    action = "discarded"
                    path = None
            except OSError as exc:
                return self._resolution_error_locked(str(exc))

            self._pending.pop(pending.recording_id, None)
            self._last_error = None
            self._last_event = action
            self._last_resolution = {
                "action": action,
                "episode_id": _json_safe_episode_id(pending.episode_id),
                "session_id": _json_safe_episode_id(pending.session_id),
                "recording_id": pending.recording_id,
                "path": path,
            }
            return {"ok": True, **self._last_resolution, "status": self.status()}

    def status(self) -> dict[str, Any]:
        """Return a JSON-serializable snapshot of recorder state."""

        with self._lock:
            now = self._clock()
            active_status = None
            if self._active is not None:
                active = self._active
                stationary_seconds = (
                    max(0.0, now - active.stationary_since)
                    if active.stationary_since is not None
                    else 0.0
                )
                active_status = {
                    "episode_id": _json_safe_episode_id(active.episode_id),
                    "session_id": _json_safe_episode_id(active.session_id),
                    "recording_id": active.recording_id,
                    "frames": active.frames,
                    "duration_seconds": max(0.0, now - active.started_monotonic),
                    "encoded_duration_seconds": active.frames / self.fps,
                    "motion_seen": active.motion_seen,
                    "stationary_seconds": stationary_seconds,
                    "linear_speed": active.linear_speed,
                    "angular_speed": active.angular_speed,
                    "speed_source": active.speed_source,
                }

            pending_status = [
                {
                    "episode_id": _json_safe_episode_id(pending.episode_id),
                    "session_id": _json_safe_episode_id(pending.session_id),
                    "recording_id": pending.recording_id,
                    "frames": pending.frames,
                    "duration_seconds": max(
                        0.0, pending.ended_monotonic - pending.started_monotonic
                    ),
                    "encoded_duration_seconds": pending.frames / self.fps,
                    "reason": pending.reason,
                    "temp_path": str(pending.temp_path),
                }
                for pending in self._pending.values()
            ]
            state = (
                "recording"
                if active_status is not None
                else "pending"
                if pending_status
                else "idle"
            )
            if not self.enabled:
                state = "disabled"
            return {
                "enabled": self.enabled,
                "state": state,
                "experiment": self.experiment_name,
                "active": active_status,
                "pending": pending_status,
                "pending_count": len(pending_status),
                "last_event": self._last_event,
                "last_error": self._last_error,
                "last_resolution": dict(self._last_resolution) if self._last_resolution else None,
                "config": {
                    "width": self.width,
                    "height": self.height,
                    "fps": self.fps,
                    "stop_linear_threshold": self.stop_linear_threshold,
                    "stop_angular_threshold": self.stop_angular_threshold,
                    "stop_hold_seconds": self.stop_hold_seconds,
                    "recordings_dir": str(self.recordings_dir),
                },
            }

    def cleanup(self) -> dict[str, Any]:
        """Reap the encoder and remove every unresolved temporary video."""

        with self._lock:
            if self._active is not None:
                self._abort_active_locked()
            for pending in list(self._pending.values()):
                try:
                    pending.temp_path.unlink(missing_ok=True)
                except OSError as exc:
                    self._last_error = str(exc)
            self._pending.clear()
            self._finished_episode_keys.clear()
            self._last_event = "cleaned_up"
            return self.status()

    close = cleanup

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.cleanup()

    def _start_episode_locked(
        self,
        episode_id: Any,
        session_id: Any,
        episode_key: str,
        now: float,
    ) -> None:
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        recording_id = uuid.uuid4().hex
        timestamp = datetime.fromtimestamp(self._wall_clock()).strftime("%Y%m%d_%H%M%S_%f")
        episode_part = _safe_filename_part(episode_id)
        temp_path = self.temp_dir / (
            f"signnav_{self.experiment_name}_episode-{episode_part}_{timestamp}_"
            f"{recording_id[:8]}.tmp.mp4"
        )
        writer = self._writer_factory(
            path=temp_path,
            width=self.width,
            height=self.height,
            fps=self.fps,
            ffmpeg_binary=self.ffmpeg_binary,
        )
        self._active = _ActiveRecording(
            episode_id=episode_id,
            session_id=session_id,
            episode_key=episode_key,
            recording_id=recording_id,
            temp_path=temp_path,
            writer=writer,
            started_monotonic=now,
            started_wall_time=self._wall_clock(),
        )
        self._last_error = None
        self._last_event = "started"

    def _flush_active_until_locked(
        self,
        active: _ActiveRecording,
        finish_time: float,
    ) -> None:
        """Fill the fixed-fps timeline with the last visible dashboard frame."""

        if active.last_frame is None or active.next_frame_monotonic is None:
            return
        frame_period = 1.0 / self.fps
        epsilon = frame_period * 1e-6
        while active.next_frame_monotonic < finish_time - epsilon:
            active.writer.write(active.last_frame)
            active.frames += 1
            active.next_frame_monotonic += frame_period

    def _finish_active_locked(self, reason: str, finish_time: float | None = None) -> None:
        active = self._active
        if active is None:
            return
        self._active = None
        finish_time = self._clock() if finish_time is None else finish_time
        try:
            self._flush_active_until_locked(active, finish_time)
            active.writer.close()
            if active.frames <= 0 or not active.temp_path.is_file():
                raise RuntimeError("ffmpeg did not produce a dashboard video")
        except Exception as exc:
            try:
                active.writer.abort()
            except Exception:
                pass
            active.temp_path.unlink(missing_ok=True)
            self._finished_episode_keys.add(active.episode_key)
            self._last_error = str(exc)
            self._last_event = "finalize_failed"
            return

        pending = _PendingRecording(
            episode_id=active.episode_id,
            session_id=active.session_id,
            episode_key=active.episode_key,
            recording_id=active.recording_id,
            temp_path=active.temp_path,
            started_monotonic=active.started_monotonic,
            ended_monotonic=finish_time,
            started_wall_time=active.started_wall_time,
            ended_wall_time=self._wall_clock(),
            frames=active.frames,
            reason=reason,
        )
        self._pending[pending.recording_id] = pending
        self._finished_episode_keys.add(active.episode_key)
        self._last_error = None
        self._last_event = "pending"

    def _abort_active_locked(self) -> None:
        active = self._active
        if active is None:
            return
        self._active = None
        try:
            active.writer.abort()
        except Exception:
            pass
        try:
            active.temp_path.unlink(missing_ok=True)
        except OSError as exc:
            self._last_error = str(exc)
        self._finished_episode_keys.add(active.episode_key)

    def _recording_destination_locked(self, pending: _PendingRecording) -> Path:
        timestamp = datetime.fromtimestamp(pending.started_wall_time).strftime("%Y%m%d_%H%M%S")
        episode_part = _safe_filename_part(pending.episode_id)
        session_part = _safe_filename_part(pending.session_id)[:8]
        stem = (
            f"signnav_{self.experiment_name}_session-{session_part}_"
            f"episode-{episode_part}_{timestamp}"
        )
        destination = self.recordings_dir / f"{stem}.mp4"
        if destination.exists():
            destination = self.recordings_dir / f"{stem}_{pending.recording_id[:8]}.mp4"
        return destination

    def _resolution_error_locked(self, message: str) -> dict[str, Any]:
        self._last_error = message
        self._last_event = "resolve_failed"
        return {"ok": False, "error": message, "status": self.status()}

    def _draw_camera_grid(
        self,
        canvas: Image.Image,
        draw: ImageDraw.ImageDraw,
        images: Mapping[str, str],
        bounds: tuple[int, int, int, int],
    ) -> None:
        x, y, width, height = bounds
        ordered_keys = [key for key in _VIEW_ORDER if key in images]
        ordered_keys.extend(sorted(key for key in images if key not in ordered_keys))
        ordered_keys = ordered_keys[:4] or ["ego_view"]
        columns = 1 if len(ordered_keys) == 1 else 2
        rows = math.ceil(len(ordered_keys) / columns)
        gap = 10
        cell_width = (width - gap * (columns - 1)) // columns
        cell_height = (height - gap * (rows - 1)) // rows
        for index, key in enumerate(ordered_keys):
            row, column = divmod(index, columns)
            cell_x = x + column * (cell_width + gap)
            cell_y = y + row * (cell_height + gap)
            self._draw_camera_card(
                canvas,
                draw,
                key,
                images.get(key),
                (cell_x, cell_y, cell_width, cell_height),
            )

    def _draw_camera_card(
        self,
        canvas: Image.Image,
        draw: ImageDraw.ImageDraw,
        key: str,
        image_b64: str | None,
        bounds: tuple[int, int, int, int],
    ) -> None:
        x, y, width, height = bounds
        title_height = 28
        draw.rounded_rectangle(
            (x, y, x + width - 1, y + height - 1),
            radius=7,
            fill=_CAMERA_BACKGROUND,
            outline=_BORDER,
            width=1,
        )
        draw.rectangle((x + 1, y + 1, x + width - 2, y + title_height), fill=_CARD_BACKGROUND)
        label = _VIEW_LABELS.get(key, key.replace("_", " ").upper())
        draw.text((x + 8, y + 7), label, fill=(174, 179, 189), font=self._small_font)

        image_bounds = (x + 2, y + title_height + 2, width - 4, height - title_height - 4)
        image = self._decode_image(image_b64)
        if image is None:
            message = "WAITING FOR IMAGE" if image_b64 is None else "INVALID IMAGE"
            message_box = draw.textbbox((0, 0), message, font=self._section_font)
            message_width = message_box[2] - message_box[0]
            message_height = message_box[3] - message_box[1]
            image_x, image_y, image_width, image_height = image_bounds
            draw.text(
                (
                    image_x + (image_width - message_width) // 2,
                    image_y + (image_height - message_height) // 2,
                ),
                message,
                fill=(102, 102, 102),
                font=self._section_font,
            )
            return

        image_x, image_y, image_width, image_height = image_bounds
        contained = ImageOps.contain(image, (image_width, image_height), Image.Resampling.LANCZOS)
        paste_x = image_x + (image_width - contained.width) // 2
        paste_y = image_y + (image_height - contained.height) // 2
        canvas.paste(contained, (paste_x, paste_y))

    @staticmethod
    def _decode_image(image_b64: str | None) -> Image.Image | None:
        if not image_b64:
            return None
        try:
            raw = base64.b64decode(image_b64, validate=True)
            with Image.open(io.BytesIO(raw)) as image:
                return ImageOps.exif_transpose(image).convert("RGB")
        except (ValueError, OSError):
            return None

    def _draw_telemetry_panel(
        self,
        draw: ImageDraw.ImageDraw,
        telemetry: Mapping[str, Any],
        bounds: tuple[int, int, int, int],
    ) -> None:
        x, y, width, height = bounds
        draw.rounded_rectangle(
            (x, y, x + width - 1, y + height - 1),
            radius=7,
            fill=_CARD_BACKGROUND,
            outline=_BORDER,
            width=1,
        )
        cursor_y = y + 14
        rows = [
            ("STEP", telemetry.get("step", "-")),
            ("EPISODE", telemetry.get("episode_id", "-")),
            ("TARGET AREA", self._area_text(telemetry.get("target_area"))),
            ("PROMPT", self._prompt_text(telemetry.get("prompt_version"))),
            ("SPEED", self._number_text(telemetry.get("speed"), 3)),
            ("ROBOT V / W", self._motion_text(telemetry)),
            ("ACTION STEP", self._action_step_text(telemetry)),
            ("CAM TO INPUT", self._latency_text(telemetry.get("camera_to_model_input_ms"))),
        ]
        for label, key in (
            ("ACTION INFER", "action_inference_ms"),
            ("GROUND INFER", "grounding_inference_ms"),
            ("MODEL TOTAL", "model_total_ms"),
        ):
            if telemetry.get(key) is not None:
                rows.append((label, self._latency_text(telemetry.get(key))))

        grounding = telemetry.get("sign_grounding")
        if isinstance(grounding, Mapping) and grounding.get("ok"):
            probability = _finite_float(grounding.get("found_probability"))
            probability_text = f"{probability:.3f}" if probability is not None else "-"
            rows.append(
                ("GROUNDING", f"S{grounding.get('pred_status', '-')} / P {probability_text}")
            )

        sam3 = telemetry.get("sam3_timing_ms")
        if isinstance(sam3, Mapping):
            matched_text = sam3.get("matched_sign_text")
            if matched_text:
                rows.append(
                    ("TARGET MARKER", f"{matched_text} / {sam3.get('arrow_direction', '-')}")
                )
            bbox = sam3.get("bbox_state")
            if isinstance(bbox, Mapping):
                bbox_values = [
                    bbox.get("bbox_status"),
                    bbox.get("bbox_x1"),
                    bbox.get("bbox_y1"),
                    bbox.get("bbox_x2"),
                    bbox.get("bbox_y2"),
                ]
                rows.append(
                    (
                        "BBOX STATE",
                        ", ".join(self._number_text(value, 2) for value in bbox_values),
                    )
                )
            rows.append(("SAM3", self._sam3_text(sam3)))

        for label, value in rows:
            draw.text((x + 12, cursor_y), label, fill=_MUTED, font=self._label_font)
            value_text = str(value)[:25]
            value_box = draw.textbbox((0, 0), value_text, font=self._value_font)
            value_width = value_box[2] - value_box[0]
            draw.text(
                (max(x + 118, x + width - value_width - 12), cursor_y - 1),
                value_text,
                fill=_FOREGROUND,
                font=self._value_font,
            )
            cursor_y += 31

        cursor_y += 3
        vx = _finite_float(telemetry.get("vx")) or 0.0
        wz = _finite_float(telemetry.get("wz")) or 0.0
        cursor_y = self._draw_command_bar(draw, x, cursor_y, width, "LINEAR CMD", vx, 1.0)
        cursor_y = self._draw_command_bar(draw, x, cursor_y, width, "ANGULAR CMD", wz, 1.5)

        recording_text = "REC"
        recording_color = _NEGATIVE
        if self._active is not None:
            elapsed = max(0.0, self._clock() - self._active.started_monotonic)
            recording_text = f"REC  EP {self._active.episode_id}  {elapsed:.1f}s"
        elif self._pending:
            recording_text = f"PENDING  {len(self._pending)} VIDEO(S)"
            recording_color = _ACCENT
        draw.ellipse((x + 12, cursor_y + 3, x + 22, cursor_y + 13), fill=recording_color)
        draw.text((x + 30, cursor_y), recording_text, fill=recording_color, font=self._section_font)

        footer = f"{self.experiment_name}  |  {self.width}x{self.height} @ {self.fps:g} FPS"
        draw.text((x + 12, y + height - 24), footer, fill=(105, 108, 116), font=self._small_font)

    def _draw_command_bar(
        self,
        draw: ImageDraw.ImageDraw,
        x: int,
        y: int,
        width: int,
        label: str,
        value: float,
        maximum: float,
    ) -> int:
        draw.text((x + 12, y), label, fill=_MUTED, font=self._label_font)
        value_text = f"{value:+.4f}"
        value_box = draw.textbbox((0, 0), value_text, font=self._value_font)
        value_width = value_box[2] - value_box[0]
        draw.text(
            (x + width - value_width - 12, y - 1),
            value_text,
            fill=_FOREGROUND,
            font=self._value_font,
        )

        bar_left = x + 12
        bar_right = x + width - 12
        bar_top = y + 24
        bar_bottom = bar_top + 10
        center = (bar_left + bar_right) // 2
        draw.rounded_rectangle(
            (bar_left, bar_top, bar_right, bar_bottom), radius=4, fill=(11, 12, 16)
        )
        draw.line((center, bar_top, center, bar_bottom), fill=(87, 90, 99), width=2)
        length = int(min(abs(value) / maximum, 1.0) * (bar_right - bar_left) / 2)
        if value >= 0.0:
            draw.rectangle((center, bar_top, center + length, bar_bottom), fill=_ACCENT)
        else:
            draw.rectangle((center - length, bar_top, center, bar_bottom), fill=_NEGATIVE)
        return y + 54

    @staticmethod
    def _number_text(value: Any, digits: int) -> str:
        number = _finite_float(value)
        return f"{number:.{digits}f}" if number is not None else "-"

    @staticmethod
    def _area_text(value: Any) -> str:
        return f"AREA_{value}" if value is not None else "-"

    @staticmethod
    def _prompt_text(value: Any) -> str:
        return f"v{value}" if value is not None else "-"

    @staticmethod
    def _action_step_text(telemetry: Mapping[str, Any]) -> str:
        step = telemetry.get("action_step")
        horizon = telemetry.get("action_horizon")
        if step is None:
            return "-"
        return f"{step} / {int(horizon) - 1}" if horizon is not None else str(step)

    @staticmethod
    def _latency_text(value: Any) -> str:
        latency = _finite_float(value)
        return f"{latency:.1f} ms" if latency is not None else "-"

    @staticmethod
    def _motion_text(telemetry: Mapping[str, Any]) -> str:
        linear, angular, source = _select_motion_speeds(telemetry)
        suffix = "" if source == "measured" else " CMD"
        return f"{linear:.3f} / {angular:.3f}{suffix}"

    @staticmethod
    def _sam3_text(value: Any) -> str:
        if not isinstance(value, Mapping):
            return "-"
        latency = _finite_float(value.get("wall_ms", value.get("lookup_ms")))
        return f"{latency:.1f} ms" if latency is not None else "-"


_singleton_lock = threading.Lock()
_singleton: DashboardVideoRecorder | None = None


def configure_dashboard_video_recorder(
    experiment_name: str,
    *,
    recordings_dir: str | Path | None = None,
    temp_dir: str | Path | None = None,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    fps: float = DEFAULT_FPS,
    stop_linear_threshold: float = DEFAULT_STOP_LINEAR_THRESHOLD,
    stop_angular_threshold: float = DEFAULT_STOP_ANGULAR_THRESHOLD,
    stop_hold_seconds: float = DEFAULT_STOP_HOLD_SECONDS,
    ffmpeg_binary: str = "ffmpeg",
    enabled: bool = True,
) -> DashboardVideoRecorder:
    """Replace and return the process-wide dashboard recorder."""

    global _singleton
    with _singleton_lock:
        if _singleton is not None:
            _singleton.cleanup()
        _singleton = DashboardVideoRecorder(
            experiment_name,
            recordings_dir=recordings_dir,
            temp_dir=temp_dir,
            width=width,
            height=height,
            fps=fps,
            stop_linear_threshold=stop_linear_threshold,
            stop_angular_threshold=stop_angular_threshold,
            stop_hold_seconds=stop_hold_seconds,
            ffmpeg_binary=ffmpeg_binary,
            enabled=enabled,
        )
        return _singleton


def capture_dashboard_video(
    images_b64: Mapping[str, str] | str | None,
    telemetry: Mapping[str, Any],
) -> dict[str, Any]:
    """Capture through the configured singleton, creating a default one lazily."""

    global _singleton
    with _singleton_lock:
        if _singleton is None:
            _singleton = DashboardVideoRecorder("default")
        recorder = _singleton
    return recorder.capture(images_b64, telemetry)


def resolve_dashboard_video(
    save: bool,
    episode_id: Any | None = None,
    *,
    session_id: Any | None = None,
    recording_id: str | None = None,
) -> dict[str, Any]:
    with _singleton_lock:
        recorder = _singleton
    if recorder is None:
        return {
            "ok": False,
            "error": "dashboard video recorder is not configured",
            "status": get_dashboard_video_status(),
        }
    return recorder.resolve(
        save,
        episode_id,
        session_id=session_id,
        recording_id=recording_id,
    )


def get_dashboard_video_status() -> dict[str, Any]:
    with _singleton_lock:
        recorder = _singleton
    if recorder is not None:
        return recorder.status()
    return {
        "enabled": False,
        "state": "unconfigured",
        "experiment": None,
        "active": None,
        "pending": [],
        "pending_count": 0,
        "last_event": "unconfigured",
        "last_error": None,
        "last_resolution": None,
        "config": None,
    }


def cleanup_dashboard_video_recorder() -> dict[str, Any]:
    global _singleton
    with _singleton_lock:
        recorder = _singleton
        _singleton = None
    if recorder is not None:
        return recorder.cleanup()
    return get_dashboard_video_status()


atexit.register(cleanup_dashboard_video_recorder)
