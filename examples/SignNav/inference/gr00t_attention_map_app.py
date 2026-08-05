"""Desktop SignNav GR00T attention-map viewer.

This app opens a native desktop window. Pick a GR00T checkpoint directory,
pick one RGB image, run SignNav inference, and inspect the action-head
cross-attention overlay.

The heatmap summarizes action-token -> image-token cross-attention by averaging
selected image-attention blocks, heads, and action horizon steps.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile

import cv2
import numpy as np
from PIL import Image
from PyQt5 import QtCore, QtGui, QtWidgets
import torch
from diffusers.models.attention_processor import AttnProcessor

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import MessageType
from gr00t.policy.gr00t_policy import Gr00tPolicy, _rec_to_dtype

from gr00t_inference_server import (
    DEFAULT_MODALITY_CONFIG,
    build_prompt,
    normalize_prompt_version,
    normalize_target_area,
)


TARGET_BLOCKS = [2, 6, 10, 14, 18, 22, 26, 30]
MAX_PREVIEW_SIZE = (720, 540)
DEFAULT_MODEL_ROOT_CANDIDATES = [
    Path("/nas/sujinkim/model/SignNav"),
    Path("/nas/sujinkim/mode/SignNav"),
]
DEFAULT_MODEL_ROOT = next(
    (path for path in DEFAULT_MODEL_ROOT_CANDIDATES if path.exists()),
    DEFAULT_MODEL_ROOT_CANDIDATES[0],
)
DEFAULT_IMAGE_DIR = Path("~/Pictures").expanduser()


class CaptureCrossAttnProcessor(AttnProcessor):
    """Attention processor that stores cross-attention probabilities."""

    def __init__(self, name: str, attn_store: dict):
        super().__init__()
        self.name = name
        self.attn_store = attn_store

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        *args,
        **kwargs,
    ):
        residual = hidden_states
        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size = hidden_states.shape[0]
        query_len = hidden_states.shape[1]
        is_cross_attention = encoder_hidden_states is not None

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key_len = encoder_hidden_states.shape[1]
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, key_len, batch_size)
            if attention_mask.dtype == torch.bool:
                attention_mask = torch.zeros_like(
                    attention_mask, dtype=hidden_states.dtype
                ).masked_fill(~attention_mask, -10000.0)
            else:
                attention_mask = attention_mask.to(dtype=hidden_states.dtype)
            if attention_mask.ndim == 2:
                attention_mask = attention_mask.unsqueeze(1)
            if attention_mask.shape[1] == 1:
                attention_mask = attention_mask.expand(-1, query_len, -1)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)
        attention_probs = attn.get_attention_scores(query, key, attention_mask)

        if is_cross_attention:
            self.attn_store.setdefault(self.name, []).append(
                attention_probs.detach()
                .float()
                .cpu()
                .view(batch_size, attn.heads, query_len, key_len)
            )

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        return hidden_states / attn.rescale_output_factor


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


def install_action_cross_attention_capture(model, attn_store: dict) -> list[str]:
    modules = dict(model.named_modules())
    installed = []
    for idx in TARGET_BLOCKS:
        name = f"action_head.model.transformer_blocks.{idx}.attn1"
        module = modules.get(name)
        if module is None:
            continue
        module.set_processor(CaptureCrossAttnProcessor(name, attn_store))
        installed.append(name)
    return installed


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


def build_observation(image: np.ndarray, speed: float, language: str, policy):
    video = {}
    video_horizon = get_video_horizon(policy)
    for key in get_policy_video_keys(policy):
        frames = np.stack([image] * video_horizon, axis=0)
        video[key] = frames[np.newaxis].astype(np.uint8)

    return {
        "video": video,
        "state": {"speed": np.array([[[float(speed)]]], dtype=np.float32)},
        "language": {"annotation.human.action.task_description": [[language]]},
    }


def get_image_mask(policy, obs) -> torch.Tensor:
    unbatched_obs = policy._unbatch_observation(obs)
    vla_step_data = policy._to_vla_step_data(unbatched_obs[0])
    processed_inputs = [
        policy.processor([{"type": MessageType.EPISODE_STEP.value, "content": vla_step_data}])
    ]
    collated_inputs = policy.collate_fn(processed_inputs)
    collated_inputs = _rec_to_dtype(collated_inputs, dtype=torch.bfloat16)

    with torch.inference_mode():
        backbone_inputs, _ = policy.model.prepare_input(collated_inputs["inputs"])
        backbone_outputs = policy.model.backbone(backbone_inputs)

    return backbone_outputs.image_mask.detach().cpu()[0].bool()


def best_grid(num_tokens: int) -> tuple[int, int]:
    rows = int(math.sqrt(num_tokens))
    while rows > 1 and num_tokens % rows != 0:
        rows -= 1
    cols = math.ceil(num_tokens / rows)
    return rows, cols


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    scores = scores.astype(np.float32)
    scores = scores - float(scores.min())
    return scores / (float(scores.max()) + 1e-8)


def attention_heatmap(policy, attn_store: dict, obs) -> tuple[np.ndarray, list[dict]]:
    if not attn_store:
        raise RuntimeError("no cross-attention maps were captured")

    image_mask = get_image_mask(policy, obs)
    action_horizon = policy.model.action_head.action_horizon
    block_heatmaps = []
    summaries = []

    for name, maps in attn_store.items():
        if not maps:
            continue
        attn = maps[-1]
        action_to_vl = attn[:, :, -action_horizon:, :].mean(dim=(1, 2))[0]
        image_scores = action_to_vl[image_mask].numpy()
        if image_scores.size == 0:
            continue
        image_scores = normalize_scores(image_scores)
        rows, cols = best_grid(image_scores.size)
        padded = np.zeros(rows * cols, dtype=np.float32)
        padded[: image_scores.size] = image_scores
        block_heatmaps.append(padded.reshape(rows, cols))
        summaries.append({"block": name.split(".")[-3], "tokens": int(image_scores.size), "grid": f"{rows}x{cols}"})

    if not block_heatmaps:
        raise RuntimeError("captured attention did not contain image tokens")

    resized = [
        cv2.resize(hm, (512, 512), interpolation=cv2.INTER_LINEAR) for hm in block_heatmaps
    ]
    return np.mean(np.stack(resized, axis=0), axis=0), summaries


def overlay_heatmap(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    heatmap = normalize_scores(heatmap)
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
    color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(image.astype(np.uint8), 1.0 - alpha, color, alpha, 0)


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


def checkpoint_step(path: Path) -> int:
    name = path.name
    if name.startswith("checkpoint-"):
        try:
            return int(name.split("-", 1)[1])
        except ValueError:
            return -1
    return -1


def resolve_checkpoint_path(path: str | Path) -> Path:
    path = Path(path).expanduser().resolve()
    if path.is_file() and path.name == "config.json":
        path = path.parent
    if (path / "config.json").exists():
        return path
    raise ValueError(f"checkpoint config.json을 찾을 수 없습니다:\n{path}")


def find_checkpoint_candidates(path: str | Path) -> list[Path]:
    path = Path(path).expanduser().resolve()
    candidates = [candidate.parent for candidate in path.glob("checkpoint-*/config.json")]
    if not candidates:
        candidates = [candidate.parent for candidate in path.rglob("checkpoint-*/config.json")]
    return sorted(set(candidates), key=lambda item: (checkpoint_step(item), str(item)))


def default_initial_dir(path_value: str, fallback: Path) -> str:
    if path_value:
        path = Path(path_value).expanduser()
        if path.is_file():
            path = path.parent
        if path.exists():
            return str(path)
        if path.parent.exists():
            return str(path.parent)
    if fallback.exists():
        return str(fallback)
    return str(Path.home())


def run_analysis_once(args) -> int:
    print("[ANALYZE] RGB 이미지 디코딩 중...", flush=True)
    image = Image.open(args.analysis_image_path).convert("RGB")
    image_np = np.array(image, dtype=np.uint8)
    original_preview_path = Path(args.analysis_overlay_path).with_name("original_preview.png")
    original_preview = image.copy()
    original_preview.thumbnail(MAX_PREVIEW_SIZE, Image.Resampling.LANCZOS)
    original_preview.save(original_preview_path)

    print("[ANALYZE] 모델 로딩 중...", flush=True)
    cache = PolicyCache(device=args.device)
    try:
        policy = cache.get(args.analysis_model_path)
        policy.reset()
        attn_store = {}
        installed = install_action_cross_attention_capture(policy.model, attn_store)
        if not installed:
            raise RuntimeError("no action-head attention blocks were found on this model")

        print("[ANALYZE] 추론 실행 중...", flush=True)
        language = build_prompt(
            normalize_prompt_version(args.analysis_prompt_version),
            normalize_target_area(args.analysis_target_area),
        )
        obs = build_observation(image_np, args.analysis_speed, language, policy)
        with torch.inference_mode():
            action, _ = policy.get_action(obs)

        print("[ANALYZE] attention heatmap 생성 중...", flush=True)
        vx, wz, selected_step, action_horizon = select_vel_cmd_step(
            action["vel_cmd"], args.analysis_action_step
        )
        heatmap, summaries = attention_heatmap(policy, attn_store, obs)
        overlay = overlay_heatmap(image_np, heatmap, alpha=0.45)

        overlay_path = Path(args.analysis_overlay_path)
        Image.fromarray(overlay).save(overlay_path)
        overlay_preview_path = overlay_path.with_name("overlay_preview.png")
        overlay_preview = Image.fromarray(overlay)
        overlay_preview.thumbnail(MAX_PREVIEW_SIZE, Image.Resampling.LANCZOS)
        overlay_preview.save(overlay_preview_path)
        result = {
            "linear": vx,
            "angular": wz,
            "action_step": selected_step,
            "action_horizon": action_horizon,
            "blocks": summaries,
            "overlay_path": str(overlay_path),
            "overlay_preview_path": str(overlay_preview_path),
            "original_preview_path": str(original_preview_path),
        }
        Path(args.analysis_result_path).write_text(json.dumps(result), encoding="utf-8")
        print("[ANALYZE] 완료", flush=True)
        return 0
    finally:
        cache.close()


class PolicyCache:
    def __init__(self, device: str):
        self.device = device
        self.model_path = None
        self.policy = None

    def get(self, model_path: str):
        if self.policy is not None and self.model_path == model_path:
            return self.policy

        self.close()
        print(f"[GR00T] Loading model from {model_path} ...")
        self.policy = Gr00tPolicy(
            embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
            model_path=model_path,
            device=self.device,
            strict=False,
        )
        self.model_path = model_path
        print(
            f"[GR00T] Model loaded. video_keys={get_policy_video_keys(self.policy)} "
            f"video_horizon={get_video_horizon(self.policy)}"
        )
        return self.policy

    def close(self):
        if self.policy is not None:
            del self.policy
        self.policy = None
        self.model_path = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class ImagePanel(QtWidgets.QFrame):
    def __init__(self, title: str, placeholder: str):
        super().__init__()
        self.pixmap = None
        self.setObjectName("ImagePanel")
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 14)
        layout.setSpacing(10)

        self.title = QtWidgets.QLabel(title)
        self.title.setObjectName("PanelTitle")
        self.canvas = QtWidgets.QLabel(placeholder)
        self.canvas.setObjectName("ImageCanvas")
        self.canvas.setAlignment(QtCore.Qt.AlignCenter)
        self.canvas.setMinimumSize(420, 420)
        self.canvas.setWordWrap(True)

        layout.addWidget(self.title)
        layout.addWidget(self.canvas, 1)

    def set_image(self, path: str | Path):
        pixmap = QtGui.QPixmap(str(path))
        if pixmap.isNull():
            self.clear("Failed to load image")
            return
        self.pixmap = pixmap
        self._fit_pixmap()

    def clear(self, text: str):
        self.pixmap = None
        self.canvas.clear()
        self.canvas.setText(text)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._fit_pixmap()

    def _fit_pixmap(self):
        if self.pixmap is None:
            return
        scaled = self.pixmap.scaled(
            self.canvas.size(),
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation,
        )
        self.canvas.setPixmap(scaled)
        self.canvas.setText("")


class AttentionMapWindow(QtWidgets.QMainWindow):
    def __init__(self, device: str, modality_config_path: str):
        super().__init__()
        self.device = device
        self.modality_config_path = modality_config_path
        self.analysis_proc = None
        self.analysis_result_path = None
        self.analysis_overlay_path = None
        self.analysis_log_path = None
        self.analysis_log_pos = 0
        self.analysis_log_tail = ""

        self.setWindowTitle("SignNav GR00T Attention Map")
        self.resize(1500, 900)
        self.setMinimumSize(1180, 720)
        self._build_ui()
        self._apply_style()
        self._update_run_state()

        self.poll_timer = QtCore.QTimer(self)
        self.poll_timer.setInterval(500)
        self.poll_timer.timeout.connect(self._poll_analysis_process)

    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QHBoxLayout(central)
        root.setContentsMargins(18, 18, 18, 18)
        root.setSpacing(18)

        sidebar = QtWidgets.QFrame()
        sidebar.setObjectName("Sidebar")
        sidebar.setFixedWidth(440)
        side = QtWidgets.QVBoxLayout(sidebar)
        side.setContentsMargins(18, 18, 18, 18)
        side.setSpacing(14)
        root.addWidget(sidebar)

        title = QtWidgets.QLabel("SignNav Attention Map")
        title.setObjectName("AppTitle")
        subtitle = QtWidgets.QLabel("Choose a checkpoint and one RGB frame, then inspect action-head visual attention.")
        subtitle.setObjectName("Subtitle")
        subtitle.setWordWrap(True)
        side.addWidget(title)
        side.addWidget(subtitle)

        inputs = self._card("Inputs")
        self.checkpoint_edit = self._path_control(
            inputs.layout(),
            "Checkpoint",
            "checkpoint folder or config.json",
            [("Folder", self.select_checkpoint_folder), ("config.json", self.select_checkpoint_config)],
        )
        self.image_edit = self._path_control(
            inputs.layout(),
            "RGB image",
            "image file",
            [("Image file", self.select_image)],
        )
        side.addWidget(inputs)

        inference = self._card("Inference")
        form = QtWidgets.QFormLayout()
        form.setHorizontalSpacing(18)
        form.setVerticalSpacing(10)
        form.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
        inference.layout().addLayout(form)

        self.prompt_combo = QtWidgets.QComboBox()
        self.prompt_combo.addItems(["1", "2", "3"])
        self.area_spin = QtWidgets.QSpinBox()
        self.area_spin.setRange(1, 12)
        self.speed_spin = QtWidgets.QDoubleSpinBox()
        self.speed_spin.setRange(-10.0, 10.0)
        self.speed_spin.setDecimals(3)
        self.speed_spin.setSingleStep(0.05)
        self.action_step_spin = QtWidgets.QSpinBox()
        self.action_step_spin.setRange(0, 15)
        self.action_step_spin.setValue(1)
        form.addRow("Prompt version", self.prompt_combo)
        form.addRow("Target area", self.area_spin)
        form.addRow("Speed", self.speed_spin)
        form.addRow("Action step", self.action_step_spin)
        side.addWidget(inference)

        self.run_button = QtWidgets.QPushButton("Run inference")
        self.run_button.setObjectName("RunButton")
        self.run_button.clicked.connect(self.run_inference)
        side.addWidget(self.run_button)

        status = self._card("Status")
        self.status_label = QtWidgets.QLabel("Select a checkpoint and an RGB image.")
        self.status_label.setObjectName("StatusLabel")
        self.status_label.setWordWrap(True)
        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.hide()
        self.log_text = QtWidgets.QPlainTextEdit()
        self.log_text.setObjectName("LogText")
        self.log_text.setReadOnly(True)
        self.log_text.setMinimumHeight(180)
        status.layout().addWidget(self.status_label)
        status.layout().addWidget(self.progress)
        status.layout().addWidget(self.log_text, 1)
        side.addWidget(status, 1)

        viewer = QtWidgets.QWidget()
        viewer_layout = QtWidgets.QGridLayout(viewer)
        viewer_layout.setContentsMargins(0, 0, 0, 0)
        viewer_layout.setSpacing(18)
        viewer_layout.setColumnStretch(0, 1)
        viewer_layout.setColumnStretch(1, 1)
        self.original_panel = ImagePanel("Original RGB", "No image selected")
        self.overlay_panel = ImagePanel("Attention Overlay", "Run inference to create overlay")
        viewer_layout.addWidget(self.original_panel, 0, 0)
        viewer_layout.addWidget(self.overlay_panel, 0, 1)
        root.addWidget(viewer, 1)

    def _card(self, title: str):
        frame = QtWidgets.QFrame()
        frame.setObjectName("Card")
        layout = QtWidgets.QVBoxLayout(frame)
        layout.setContentsMargins(14, 12, 14, 14)
        layout.setSpacing(10)
        label = QtWidgets.QLabel(title)
        label.setObjectName("SectionTitle")
        layout.addWidget(label)
        return frame

    def _path_control(self, layout, label: str, placeholder: str, buttons: list[tuple[str, object]]):
        field_label = QtWidgets.QLabel(label)
        field_label.setObjectName("FieldLabel")
        edit = QtWidgets.QLineEdit()
        edit.setPlaceholderText(placeholder)
        edit.textChanged.connect(self._update_run_state)
        button_row = QtWidgets.QHBoxLayout()
        button_row.setSpacing(8)
        for text, callback in buttons:
            button = QtWidgets.QPushButton(text)
            button.clicked.connect(callback)
            button_row.addWidget(button)
        layout.addWidget(field_label)
        layout.addWidget(edit)
        layout.addLayout(button_row)
        return edit

    def _apply_style(self):
        self.setStyleSheet(
            """
            QMainWindow { background: #edf0f3; }
            #Sidebar, #Card, #ImagePanel { background: #ffffff; border: 1px solid #d6dbe1; border-radius: 8px; }
            #AppTitle { font-size: 22px; font-weight: 700; color: #20242a; }
            #Subtitle { color: #66707a; font-size: 12px; }
            #SectionTitle, #PanelTitle { font-size: 13px; font-weight: 700; color: #252a31; }
            #FieldLabel { color: #59616b; font-size: 12px; font-weight: 600; }
            QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
                min-height: 30px; padding: 4px 8px; border: 1px solid #c4cbd4; border-radius: 5px; background: #fbfcfd;
            }
            QPushButton { min-height: 30px; padding: 5px 10px; border: 1px solid #b9c1cc; border-radius: 5px; background: #f7f8fa; }
            QPushButton:hover { background: #edf1f5; }
            QPushButton:disabled { color: #a9b0b8; background: #f2f3f5; }
            #RunButton { min-height: 42px; background: #235f46; color: white; font-weight: 700; border: 1px solid #235f46; }
            #RunButton:hover { background: #2f7658; }
            #StatusLabel { color: #25303b; font-size: 12px; }
            #LogText { background: #111820; color: #d6dde6; border: 1px solid #26313c; border-radius: 6px; font-family: monospace; font-size: 11px; }
            #ImageCanvas { background: #f8f9fb; border: 1px solid #d9dee5; border-radius: 6px; color: #8b949e; font-size: 14px; }
            """
        )

    def select_checkpoint_folder(self):
        selected = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Select checkpoint or experiment folder",
            default_initial_dir(self.checkpoint_edit.text(), DEFAULT_MODEL_ROOT),
        )
        if selected:
            self._set_checkpoint_from_path(selected)

    def select_checkpoint_config(self):
        selected, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select checkpoint config.json",
            default_initial_dir(self.checkpoint_edit.text(), DEFAULT_MODEL_ROOT),
            "config.json (config.json);;JSON files (*.json);;All files (*)",
        )
        if selected:
            self._set_checkpoint_from_path(selected)

    def _set_checkpoint_from_path(self, selected: str):
        try:
            path = resolve_checkpoint_path(selected)
        except ValueError:
            candidates = find_checkpoint_candidates(selected)
            if not candidates:
                self._error("Invalid checkpoint", f"No checkpoint config.json found under:\n{selected}")
                return
            path = self.choose_checkpoint(candidates)
            if path is None:
                return
        self.checkpoint_edit.setText(str(path))
        self.status_label.setText("Checkpoint selected.")
        self._update_run_state()

    def choose_checkpoint(self, candidates: list[Path]) -> Path | None:
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Choose checkpoint")
        dialog.resize(860, 480)
        layout = QtWidgets.QVBoxLayout(dialog)
        label = QtWidgets.QLabel("Select one checkpoint. Nothing is chosen automatically.")
        label.setObjectName("SectionTitle")
        layout.addWidget(label)
        list_widget = QtWidgets.QListWidget()
        for candidate in candidates:
            list_widget.addItem(f"{candidate.name}    {candidate}")
        list_widget.setCurrentRow(0)
        layout.addWidget(list_widget, 1)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        list_widget.itemDoubleClicked.connect(lambda _item: dialog.accept())
        layout.addWidget(buttons)
        if dialog.exec_() != QtWidgets.QDialog.Accepted or list_widget.currentRow() < 0:
            return None
        return candidates[list_widget.currentRow()]

    def select_image(self):
        selected, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select RGB image",
            default_initial_dir(self.image_edit.text(), DEFAULT_IMAGE_DIR),
            "Image files (*.png *.jpg *.jpeg *.bmp *.webp);;All files (*)",
        )
        if not selected:
            return
        self.image_edit.setText(str(Path(selected).expanduser().resolve()))
        self.original_panel.clear("Image selected. Preview appears after analysis.")
        self.overlay_panel.clear("Run inference to create overlay.")
        self.status_label.setText("Ready. Run inference when you want to analyze this frame.")
        self._update_run_state()

    def _update_run_state(self):
        has_inputs = bool(self.checkpoint_edit.text().strip()) and bool(self.image_edit.text().strip())
        self.run_button.setEnabled(has_inputs and self.analysis_proc is None)

    def run_inference(self):
        if self.analysis_proc is not None:
            return
        try:
            checkpoint_path = resolve_checkpoint_path(self.checkpoint_edit.text())
            prompt_version = normalize_prompt_version(int(self.prompt_combo.currentText()))
            target_area = normalize_target_area(self.area_spin.value())
            speed = float(self.speed_spin.value())
            action_step = int(self.action_step_spin.value())
        except Exception as exc:
            self._error("Invalid input", str(exc))
            return
        if not Path(self.image_edit.text()).expanduser().exists():
            self._error("Invalid image", f"Image file does not exist:\n{self.image_edit.text()}")
            return

        self.checkpoint_edit.setText(str(checkpoint_path))
        self.run_button.setEnabled(False)
        self.progress.show()
        self.log_text.clear()
        self.status_label.setText("Starting analysis subprocess...")
        self.original_panel.clear("Preparing image...")
        self.overlay_panel.clear("Waiting for attention overlay...")

        tmpdir = Path(tempfile.mkdtemp(prefix="signnav_attention_"))
        self.analysis_result_path = tmpdir / "result.json"
        self.analysis_overlay_path = tmpdir / "overlay.png"
        self.analysis_log_path = tmpdir / "analysis.log"
        self.analysis_log_pos = 0
        self.analysis_log_tail = ""

        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--analyze-once",
            "--analysis-model-path", str(checkpoint_path),
            "--analysis-image-path", self.image_edit.text(),
            "--analysis-prompt-version", str(prompt_version),
            "--analysis-target-area", str(target_area),
            "--analysis-speed", str(speed),
            "--analysis-action-step", str(action_step),
            "--analysis-result-path", str(self.analysis_result_path),
            "--analysis-overlay-path", str(self.analysis_overlay_path),
            "--device", self.device,
            "--modality-config-path", self.modality_config_path,
        ]
        log_file = self.analysis_log_path.open("w", encoding="utf-8")
        try:
            self.analysis_proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, text=True)
        finally:
            log_file.close()
        self.poll_timer.start()

    def _poll_analysis_process(self):
        self._refresh_analysis_log_status()
        if self.analysis_proc is None:
            self.poll_timer.stop()
            return
        returncode = self.analysis_proc.poll()
        if returncode is None:
            return
        self.poll_timer.stop()
        if returncode == 0:
            self._finish_success()
        else:
            self._finish_error(f"analysis subprocess failed with code {returncode}\n{self.analysis_log_tail}")

    def _refresh_analysis_log_status(self):
        if not self.analysis_log_path or not self.analysis_log_path.exists():
            return
        with self.analysis_log_path.open("r", encoding="utf-8", errors="replace") as f:
            f.seek(self.analysis_log_pos)
            chunk = f.read()
            self.analysis_log_pos = f.tell()
        if not chunk:
            return
        self.log_text.insertPlainText(chunk)
        self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())
        lines = [line.strip() for line in chunk.splitlines() if line.strip()]
        if lines:
            self.analysis_log_tail = "\n".join(lines[-8:])
            self.status_label.setText(lines[-1])

    def _finish_success(self):
        try:
            result = json.loads(self.analysis_result_path.read_text(encoding="utf-8"))
            self.original_panel.set_image(result["original_preview_path"])
            self.overlay_panel.set_image(result["overlay_preview_path"])
            blocks = ", ".join(
                f"{item['block']} ({item['tokens']}, {item['grid']})" for item in result["blocks"]
            )
            self.status_label.setText(
                f"Done. linear={result['linear']:+.4f}, angular={result['angular']:+.4f}, "
                f"action_step={result['action_step']}/{result['action_horizon'] - 1}\n"
                f"blocks: {blocks}"
            )
        except Exception as exc:
            self._finish_error(str(exc))
            return
        self.analysis_proc = None
        self.progress.hide()
        self._update_run_state()

    def _finish_error(self, message: str):
        self.analysis_proc = None
        self.progress.hide()
        self._update_run_state()
        self.status_label.setText(f"Error: {message}")
        self._error("Attention analysis failed", message)

    def _error(self, title: str, message: str):
        QtWidgets.QMessageBox.critical(self, title, message)

    def closeEvent(self, event):
        if self.analysis_proc is not None and self.analysis_proc.poll() is None:
            self.analysis_proc.terminate()
        event.accept()

def main():
    parser = argparse.ArgumentParser(description="Desktop SignNav attention-map viewer")
    parser.add_argument("--analyze-once", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--analysis-model-path", default="", help=argparse.SUPPRESS)
    parser.add_argument("--analysis-image-path", default="", help=argparse.SUPPRESS)
    parser.add_argument("--analysis-prompt-version", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--analysis-target-area", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--analysis-speed", type=float, default=0.0, help=argparse.SUPPRESS)
    parser.add_argument("--analysis-action-step", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--analysis-result-path", default="", help=argparse.SUPPRESS)
    parser.add_argument("--analysis-overlay-path", default="", help=argparse.SUPPRESS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--modality-config-path", default=str(DEFAULT_MODALITY_CONFIG))
    args = parser.parse_args()

    load_modality_config(args.modality_config_path)
    if args.analyze_once:
        return run_analysis_once(args)

    app = QtWidgets.QApplication(sys.argv)
    window = AttentionMapWindow(device=args.device, modality_config_path=args.modality_config_path)
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
