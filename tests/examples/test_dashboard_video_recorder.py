from __future__ import annotations

import base64
import io
from pathlib import Path
import shutil
import sys

import numpy as np
from PIL import Image
import pytest


INFERENCE_DIR = Path(__file__).resolve().parents[2] / "examples" / "SignNav" / "inference_warehouse"
sys.path.insert(0, str(INFERENCE_DIR))
import dashboard_video_recorder as recorder_module  # noqa: E402


class FakeClock:
    def __init__(self, value: float = 100.0):
        self.value = value

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


class FakeWriter:
    instances = []

    def __init__(self, **kwargs):
        self.path = Path(kwargs["path"])
        self.frames = []
        self.closed = False
        self.aborted = False
        self.instances.append(self)

    def write(self, frame):
        self.frames.append(frame.copy())

    def close(self):
        self.closed = True
        self.path.write_bytes(b"fake mp4")

    def abort(self):
        self.aborted = True


@pytest.fixture(autouse=True)
def _clear_fake_writers():
    FakeWriter.instances.clear()


def _image_b64(color=(20, 100, 200), size=(64, 40)) -> str:
    image = Image.new("RGB", size, color)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _make_recorder(tmp_path, clock, **kwargs):
    return recorder_module.DashboardVideoRecorder(
        "E0_vanilla",
        recordings_dir=tmp_path / "recordings",
        temp_dir=tmp_path / "temp",
        width=640,
        height=360,
        fps=5,
        clock=clock,
        wall_clock=lambda: 1_700_000_000.0,
        writer_factory=FakeWriter,
        **kwargs,
    )


def test_compositor_always_returns_fixed_rgb_frame(tmp_path):
    recorder = _make_recorder(tmp_path, FakeClock())
    frame = recorder.compose_frame(
        {
            "ego_view": _image_b64(),
            "sign_crop": _image_b64((200, 50, 10), (20, 60)),
            "broken": "not base64",
        },
        {
            "episode_id": 4,
            "step": 12,
            "target_area": 7,
            "vx": 0.2,
            "wz": -0.1,
        },
    )

    assert frame.shape == (360, 640, 3)
    assert frame.dtype == np.uint8
    assert frame.flags.c_contiguous
    assert np.any(frame != frame[0, 0])


def test_stop_is_armed_only_after_motion_and_requires_sustained_stop(tmp_path):
    clock = FakeClock()
    recorder = _make_recorder(tmp_path, clock)
    image = _image_b64()

    recorder.capture(image, {"episode_id": 1, "vx": 0.0, "wz": 0.0})
    clock.advance(5.0)
    status = recorder.capture(image, {"episode_id": 1, "vx": 0.0, "wz": 0.0})
    assert status["state"] == "recording"
    assert status["active"]["motion_seen"] is False

    recorder.capture(image, {"episode_id": 1, "vx": 0.04, "wz": 0.0})
    clock.advance(0.1)
    recorder.capture(image, {"episode_id": 1, "vx": 0.0, "wz": 0.0})
    clock.advance(1.4)
    status = recorder.capture(image, {"episode_id": 1, "vx": 0.0, "wz": 0.0})
    assert status["state"] == "recording"

    clock.advance(0.1)
    status = recorder.capture(image, {"episode_id": 1, "vx": 0.0, "wz": 0.0})
    assert status["state"] == "pending"
    assert status["pending_count"] == 1
    assert status["pending"][0]["reason"] == "robot_stopped"
    assert FakeWriter.instances[0].closed is True

    status = recorder.capture(image, {"episode_id": 1, "vx": 0.2, "wz": 0.0})
    assert status["state"] == "pending"
    assert len(FakeWriter.instances) == 1


def test_measured_velocity_takes_priority_over_zero_command(tmp_path):
    clock = FakeClock()
    recorder = _make_recorder(tmp_path, clock, stop_hold_seconds=0.5)
    image = _image_b64()

    status = recorder.capture(
        image,
        {
            "episode_id": 3,
            "vx": 0.0,
            "wz": 0.0,
            "robot_linear_speed": 0.05,
            "robot_angular_speed": 0.0,
        },
    )
    assert status["active"]["motion_seen"] is True
    assert status["active"]["speed_source"] == "measured"

    clock.advance(0.1)
    recorder.capture(
        image,
        {
            "episode_id": 3,
            "vx": 0.5,
            "wz": 0.5,
            "measured_linear_speed": 0.0,
            "measured_angular_speed": 0.0,
        },
    )
    clock.advance(0.5)
    status = recorder.capture(
        image,
        {
            "episode_id": 3,
            "vx": 0.5,
            "wz": 0.5,
            "measured_linear_speed": 0.0,
            "measured_angular_speed": 0.0,
        },
    )
    assert status["state"] == "pending"


def test_multiple_pending_recordings_can_be_saved_or_discarded_independently(tmp_path):
    clock = FakeClock()
    recorder = _make_recorder(tmp_path, clock)
    image = _image_b64()

    recorder.capture(image, {"episode_id": 1, "vx": 0.2, "wz": 0.0})
    recorder.capture(image, {"episode_id": 2, "vx": 0.2, "wz": 0.0})
    recorder.capture(image, {"episode_id": 3, "vx": 0.2, "wz": 0.0})
    status = recorder.status()
    assert [item["episode_id"] for item in status["pending"]] == [1, 2]
    assert status["active"]["episode_id"] == 3

    saved = recorder.resolve(True, episode_id=2)
    assert saved["ok"] is True
    assert saved["action"] == "saved"
    assert Path(saved["path"]).is_file()
    assert [item["episode_id"] for item in recorder.status()["pending"]] == [1]

    discarded_temp_path = Path(recorder.status()["pending"][0]["temp_path"])
    discarded = recorder.resolve(False, episode_id=1)
    assert discarded["ok"] is True
    assert discarded["action"] == "discarded"
    assert not discarded_temp_path.exists()
    assert recorder.status()["pending_count"] == 0


def test_resolve_finalizes_matching_active_recording_first(tmp_path):
    clock = FakeClock()
    recorder = _make_recorder(tmp_path, clock)
    recorder.capture(_image_b64(), {"episode_id": 9, "vx": 0.2, "wz": 0.0})

    result = recorder.resolve(True, episode_id=9)

    assert result["ok"] is True
    assert result["action"] == "saved"
    assert Path(result["path"]).is_file()
    assert FakeWriter.instances[0].closed is True
    assert recorder.status()["state"] == "idle"


def test_same_episode_number_records_again_in_a_new_client_session(tmp_path):
    clock = FakeClock()
    recorder = _make_recorder(tmp_path, clock)
    image = _image_b64()

    recorder.capture(
        image,
        {"session_id": "client-a", "episode_id": 1, "vx": 0.2, "wz": 0.0},
    )
    first = recorder.resolve(False, episode_id=1, session_id="client-a")
    status = recorder.capture(
        image,
        {"session_id": "client-b", "episode_id": 1, "vx": 0.2, "wz": 0.0},
    )

    assert first["ok"] is True
    assert status["state"] == "recording"
    assert status["active"]["session_id"] == "client-b"
    assert len(FakeWriter.instances) == 2


def test_slow_capture_and_resolve_duplicate_last_frame_to_match_wall_time(tmp_path):
    clock = FakeClock()
    recorder = _make_recorder(tmp_path, clock)
    image = _image_b64()

    recorder.capture(image, {"episode_id": 5, "step": 0, "vx": 0.2, "wz": 0.0})
    first_frame = FakeWriter.instances[0].frames[0]
    clock.advance(2.0)
    status = recorder.capture(image, {"episode_id": 5, "step": 1, "vx": 0.2, "wz": 0.0})

    assert len(FakeWriter.instances[0].frames) == 10
    assert status["active"]["duration_seconds"] == pytest.approx(2.0)
    assert status["active"]["encoded_duration_seconds"] == pytest.approx(2.0)
    assert all(np.array_equal(frame, first_frame) for frame in FakeWriter.instances[0].frames)

    clock.advance(1.0)
    recorder.resolve(False, episode_id=5)
    assert len(FakeWriter.instances[0].frames) == 15
    assert not np.array_equal(FakeWriter.instances[0].frames[-1], first_frame)


def test_cleanup_aborts_active_and_removes_pending_temps(tmp_path):
    clock = FakeClock()
    recorder = _make_recorder(tmp_path, clock)
    image = _image_b64()

    recorder.capture(image, {"episode_id": 1, "vx": 0.2, "wz": 0.0})
    recorder.capture(image, {"episode_id": 2, "vx": 0.2, "wz": 0.0})
    pending_path = Path(recorder.status()["pending"][0]["temp_path"])
    assert pending_path.exists()

    status = recorder.cleanup()
    assert status["state"] == "idle"
    assert status["pending_count"] == 0
    assert FakeWriter.instances[-1].aborted is True
    assert not pending_path.exists()


def test_ffmpeg_writer_uses_raw_rgb_and_libx264(monkeypatch, tmp_path):
    class FakeProcess:
        def __init__(self):
            self.stdin = io.BytesIO()
            self.returncode = 0

        def communicate(self, timeout=None):
            return b"", b""

    calls = []
    process = FakeProcess()
    monkeypatch.setattr(
        recorder_module.subprocess,
        "Popen",
        lambda command, **kwargs: calls.append((command, kwargs)) or process,
    )
    writer = recorder_module._RawRGBFFmpegWriter(
        tmp_path / "video.tmp.mp4",
        width=4,
        height=2,
        fps=5,
        ffmpeg_binary="ffmpeg",
    )
    frame = np.zeros((2, 4, 3), dtype=np.uint8)

    writer.write(frame)
    writer.close()

    command, kwargs = calls[0]
    assert command[command.index("-f") + 1] == "rawvideo"
    assert command[command.index("-s:v") + 1] == "4x2"
    assert command[command.index("-r") + 1] == "5"
    assert command[command.index("-vcodec", command.index("-an")) + 1] == "libx264"
    assert command[-1].endswith(".tmp.mp4")
    assert kwargs["stdin"] == -1
    assert kwargs["stderr"] == -1
    assert process.stdin.getvalue() == frame.tobytes()


def test_real_ffmpeg_produces_mp4(tmp_path):
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is not installed")
    path = tmp_path / "video.tmp.mp4"
    writer = recorder_module._RawRGBFFmpegWriter(
        path,
        width=64,
        height=40,
        fps=5,
        ffmpeg_binary="ffmpeg",
    )
    writer.write(np.zeros((40, 64, 3), dtype=np.uint8))
    writer.write(np.full((40, 64, 3), 255, dtype=np.uint8))
    writer.close()

    assert path.is_file()
    assert path.stat().st_size > 0
