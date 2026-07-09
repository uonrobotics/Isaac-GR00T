"""GR00T N1.6 action-head cross-attention 시각화 스크립트.

이 스크립트의 목적:
1. action head의 image cross-attention block에 hook을 건다.
2. action/proprio token(query)이 vision-language token(key/value)을 얼마나 참조했는지 저장한다.
3. 그중 image token에 해당하는 attention만 골라 원본 이미지 위에 heatmap으로 표시한다.

해석할 때 주의할 점:
- 여기서 보는 값은 "action을 만들 때 visual token 중 어디를 참조했는가"에 가장 가까운 신호다.
- block별 head별 action-token attention을 평균낸 요약 heatmap이다.
- image token의 실제 2D 위치는 Eagle processor의 세부 packing 방식에 의존한다. 현재 시각화는
  image token 개수를 가능한 직사각형 grid로 펴는 근사 시각화다.
"""

import base64
import io
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from diffusers.models.attention_processor import AttnProcessor

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import MessageType
from gr00t.policy.gr00t_policy import Gr00tPolicy, _rec_to_dtype


# MODEL_PATH = "/nas/sujinkim/model/following_lane/gr00t/gr00t-finetune+real_v1_lerobot/gr00t-finetune+real_v1_lerobot--20260615/checkpoint-60000/"
# MODEL_PATH = "/nas/sujinkim/model/following_lane/gr00t/gr00t-finetune+real_v1_lerobot_tophalf_masked/gr00t-finetune+real_v1_lerobot_tophalf_masked--20260623/checkpoint-60000/"
# MODEL_PATH = "/nas/sujinkim/model/following_lane/gr00t/gr00t-finetune+real_v1_lerobot_rule-ifthen/gr00t-finetune+real_v1_lerobot_rule-ifthen--20260618/checkpoint-60000/"
MODEL_PATH = "/nas/sujinkim/model/following_lane/gr00t/gr00t-finetune+real_v1_lerobot_visual-bluelane/gr00t-finetune+real_v1_lerobot_visual-bluelane--20260618/checkpoint-60000/"

# ### road 1 #### 
# GOAL_X = 5.17
# GOAL_Y = 6.6
# GOAL_YAW = -1.57

# INPUT_IMAGE_PATH = "/nas/sujinkim/data/following_lane/real_v1/rgb/road1/0007/rgb_0036.png"

# ROBOT_X = 5.19
# ROBOT_Y = 5.05
# ROBOT_YAW = 1.6

# #### road 2 ####
# GOAL_X = 2.89
# GOAL_Y = 5.76
# GOAL_YAW = 3.14

# # INPUT_IMAGE_PATH = "/nas/sujinkim/data/following_lane/real_v1/rgb/road2/0008/rgb_0000.png"
# # INPUT_IMAGE_PATH = "/nas/sujinkim/data/following_lane/real_v1/rgb/road2/0052/rgb_0035.png"
# INPUT_IMAGE_PATH = "/nas/sujinkim/data/following_lane/real_v1/rgb/road2/0066/rgb_0072.png"

# ROBOT_X = 7.42
# ROBOT_Y = 5.82
# ROBOT_YAW = -3.14

# #### unseen road (new start --> road2 goal) ####
# GOAL_X = 2.89
# GOAL_Y = 5.76
# GOAL_YAW = 3.14

# INPUT_IMAGE_PATH = "/nas/sujinkim/data/following_lane/unseen_road_test/rgb/road1/0000/rgb_0013.png"

# ROBOT_X = 2.96
# ROBOT_Y = 4.24
# ROBOT_YAW = 1.58

#### unseen road ### 
GOAL_X = 7.47
GOAL_Y = 4.97
GOAL_YAW = -0.04
INPUT_IMAGE_PATH = "/nas/sujinkim/data/following_lane/unseen_road_test/rgb/road1/0001/rgb_0000.png"
# INPUT_IMAGE_PATH = "/nas/sujinkim/data/following_lane/unseen_road_test/rgb/road1/0001/rgb_0038.png"

ROBOT_X = 5.2
ROBOT_Y = 5.04
ROBOT_YAW = -0.02


LAN_PROMPT = (
    "Navigate to the target safely. "
    "Follow the path whenever possible. " # TODO: Stay inside the blue lane whenever possible.
    "Turn in place toward the target when necessary. "
    "If a static obstacle blocks the path, "
    "detour around it and rejoin the original path. "
    "If a dynamic obstacle is approaching, "
    "stop and wait until it moves away before continuing."
)

N_ROUTE_SEGMENTS = 10
MASK_TOP_HALF_IMAGE = False
MASK_FILL_VALUE = 0


class CaptureCrossAttnProcessor(AttnProcessor):
    """Diffusers Attention processor를 교체해서 cross-attention probability를 저장한다.

    GR00T action head의 transformer block은 Diffusers `Attention`을 사용한다. 기본 processor는
    attention probability를 밖으로 반환하지 않으므로, 같은 연산을 수행하되 softmax(QK^T) 결과만
    `attn_store`에 복사해 둔다.
    """

    def __init__(self, name, attn_store):
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

        # Diffusers Attention은 4D 입력도 받을 수 있어 3D token sequence로 맞춘다.
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size = hidden_states.shape[0]
        query_len = hidden_states.shape[1]
        is_cross_attention = encoder_hidden_states is not None

        # encoder_hidden_states가 있으면 action/proprio token -> V/L token cross-attention이다.
        # 없으면 hidden_states 내부 self-attention이므로 저장 대상이 아니다.
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key_len = encoder_hidden_states.shape[1]

        # AlternateVLDiT는 image token만 attend하도록 boolean mask를 넘긴다.
        # Diffusers의 get_attention_scores는 additive mask [B*heads, query_len, key_len]를 기대하므로
        # True는 0, False는 큰 음수로 바꿔 softmax에서 제외되게 만든다.
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, key_len, batch_size)
            if attention_mask.dtype == torch.bool:
                attention_mask = torch.zeros_like(attention_mask, dtype=hidden_states.dtype).masked_fill(
                    ~attention_mask, -10000.0
                )
            else:
                attention_mask = attention_mask.to(dtype=hidden_states.dtype)
            if attention_mask.ndim == 2:
                attention_mask = attention_mask.unsqueeze(1)
            if attention_mask.shape[1] == 1:
                attention_mask = attention_mask.expand(-1, query_len, -1)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        # Q는 action/state token에서 나오고, K/V는 image/text token에서 나온다.
        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)

        if is_cross_attention:
            # 저장 shape: [batch, heads, query_tokens(state+action), key_tokens(V/L)]
            self.attn_store.setdefault(self.name, []).append(
                attention_probs.detach()
                .float()
                .cpu()
                .view(batch_size, attn.heads, query_len, key_len)
            )

        # 여기부터는 원래 AttnProcessor와 같은 출력 계산이다. 모델 동작을 바꾸지 않기 위해 필요하다.
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


def install_action_cross_attention_capture(model, attn_store): 
    """action head의 image cross-attention block에 capture processor를 설치한다.

    GR00T N1.6 기본 설정은 AlternateVLDiT + interleave_self_attention이다.
    - 홀수 block: self-attention
    - 짝수 block: cross-attention
    - block 0, 4, 8, ...: non-image/text token 쪽
    - block 2, 6, 10, ...: image token 쪽

    따라서 "action을 만들 때 visual token 중 어디를 봤는가"를 보려면
    2, 6, 10, ... block을 우선 확인한다.
    """

    modules = dict(model.named_modules())
    target_blocks = [2, 6, 10, 14, 18, 22, 26, 30]
    installed = []

    for idx in target_blocks:
        name = f"action_head.model.transformer_blocks.{idx}.attn1"
        module = modules.get(name)
        if module is None:
            continue
        module.set_processor(CaptureCrossAttnProcessor(name, attn_store))
        installed.append(name)

    print(f"[ATTN] Installed action-head image cross-attention capture on {len(installed)} blocks")
    for name in installed:
        print(f"[ATTN]   {name}")


def print_action_attention_summary(policy, attn_store):
    """캡처된 attention tensor의 shape와 기본 통계를 출력한다."""

    action_horizon = policy.model.action_head.action_horizon

    if not attn_store:
        print("[ATTN] No cross-attention maps captured.")
        return

    for name, maps in attn_store.items():
        attn = maps[-1]  # [B, heads, state+action tokens, VL tokens]

        # state token을 제외하고 마지막 action_horizon개 query token만 사용한다.
        # heads와 action timestep을 평균내면 [B, VL tokens]가 된다.
        action_to_vl = attn[:, :, -action_horizon:, :].mean(dim=(1, 2))
        print(
            f"[ATTN] {name}: calls={len(maps)} "
            f"last_shape={tuple(attn.shape)} "
            f"action_to_vl_shape={tuple(action_to_vl.shape)} "
            f"sum={float(action_to_vl[0].sum()):.6f} "
            f"max={float(action_to_vl[0].max()):.6f}"
        )


def _best_grid(num_tokens):
    """image token 개수를 가능한 직사각형 grid로 펴기 위한 행/열을 고른다."""

    rows = int(math.sqrt(num_tokens))
    while rows > 1 and num_tokens % rows != 0:
        rows -= 1
    cols = math.ceil(num_tokens / rows)
    return rows, cols


def show_action_attention_overlays(policy, attn_store, obs):
    """image-token cross-attention을 원본 이미지 위에 heatmap으로 띄운다."""

    if not attn_store:
        print("[ATTN] No cross-attention maps to visualize.")
        return

    # 시각화 단계에서 image_mask가 필요하다. image_mask는 backbone output에 들어 있으므로
    # 같은 observation을 한 번 더 processor/backbone에 통과시켜 image token 위치를 얻는다.
    unbatched_obs = policy._unbatch_observation(obs)
    vla_step_data = policy._to_vla_step_data(unbatched_obs[0])
    processed_inputs = [
        policy.processor(
            [{"type": MessageType.EPISODE_STEP.value, "content": vla_step_data}]
        )
    ]
    collated_inputs = policy.collate_fn(processed_inputs)
    collated_inputs = _rec_to_dtype(collated_inputs, dtype=torch.bfloat16)

    with torch.inference_mode():
        backbone_inputs, _ = policy.model.prepare_input(collated_inputs["inputs"])
        backbone_outputs = policy.model.backbone(backbone_inputs)

    image_mask = backbone_outputs.image_mask.detach().cpu()[0].bool()
    image = next(iter(obs["video"].values()))[0, 0].astype(np.uint8)
    action_horizon = policy.model.action_head.action_horizon

    overlays = []
    titles = []

    for name, maps in attn_store.items():
        if not maps:
            continue

        attn = maps[-1]  # [B, heads, state+action tokens, VL tokens]

        # action query만 사용하고 head/action timestep 평균을 내서 V/L token별 score를 만든다.
        action_to_vl = attn[:, :, -action_horizon:, :].mean(dim=(1, 2))[0]

        # V/L token 중 image token만 골라 "action -> image" score로 만든다.
        image_scores = action_to_vl[image_mask].numpy()

        if image_scores.size == 0:
            print(f"[ATTN] {name}: no image tokens to visualize")
            continue

        # block별 상대적 분포를 보기 위해 0~1로 정규화한다.
        image_scores = image_scores - image_scores.min()
        image_scores = image_scores / (image_scores.max() + 1e-8)

        rows, cols = _best_grid(image_scores.size)
        padded = np.zeros(rows * cols, dtype=np.float32)
        padded[: image_scores.size] = image_scores
        heatmap = padded.reshape(rows, cols)
        overlays.append((heatmap, rows, cols, image_scores.size))
        titles.append(name.split(".")[-3])

    if not overlays:
        print("[ATTN] No valid image-token attention maps to visualize.")
        return

    num_plots = len(overlays)
    ncols = min(4, num_plots)
    nrows = math.ceil(num_plots / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows))
    axes = np.atleast_1d(axes).reshape(-1)

    for ax, title, (heatmap, rows, cols, token_count) in zip(axes, titles, overlays):
        ax.imshow(image)
        ax.imshow(
            heatmap,
            cmap="jet",
            alpha=0.45,
            extent=(0, image.shape[1], image.shape[0], 0),
            interpolation="bilinear",
        )
        ax.set_title(f"{title} | {token_count} tokens | {rows}x{cols}")
        ax.axis("off")

    for ax in axes[num_plots:]:
        ax.axis("off")

    fig.suptitle("GR00T action-token to image-token cross-attention")
    fig.text(0.5, 0.92, "IMAGE_PATH=" + INPUT_IMAGE_PATH, fontsize=10, ha='center')
    fig.tight_layout()
    plt.show()


def build_observation(images_np, speed, route, goal_heading, language="", video_keys=None):
    """GR00T policy가 기대하는 video/state/language observation dict를 만든다."""

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
        video[key] = image[np.newaxis, np.newaxis].astype(np.uint8)  # (B=1, T=1, H, W, C)

    return {
        "video": video,
        "state": {
            "speed": np.array([[[speed]]], dtype=np.float32),  # (B=1, T=1, D=1)
            "route": route[np.newaxis, np.newaxis],  # (B=1, T=1, D=40)
            "goal_heading": goal_heading[np.newaxis, np.newaxis],  # (B=1, T=1, D=2)
        },
        "language": {
            "annotation.human.action.task_description": [[language]],
        },
    }


def _to_robot_frame(wx, wy, robot_x, robot_y, robot_yaw):
    """월드 좌표의 점을 로봇 기준 좌표로 변환한다."""

    cos_r = math.cos(robot_yaw)
    sin_r = math.sin(robot_yaw)
    dx, dy = wx - robot_x, wy - robot_y
    return cos_r * dx + sin_r * dy, -sin_r * dx + cos_r * dy


def compute_route_segments(robot_x, robot_y, robot_yaw, goal_x, goal_y):
    """현재 위치에서 목표까지 직선 경로를 N_ROUTE_SEGMENTS개 segment로 만든다."""

    segments = []
    for i in range(N_ROUTE_SEGMENTS):
        t_s = i / N_ROUTE_SEGMENTS
        t_e = (i + 1) / N_ROUTE_SEGMENTS
        sx, sy = _to_robot_frame(
            robot_x + t_s * (goal_x - robot_x),
            robot_y + t_s * (goal_y - robot_y),
            robot_x,
            robot_y,
            robot_yaw,
        )
        ex, ey = _to_robot_frame(
            robot_x + t_e * (goal_x - robot_x),
            robot_y + t_e * (goal_y - robot_y),
            robot_x,
            robot_y,
            robot_yaw,
        )
        segments.extend([sx, sy, ex, ey])
    return np.array(segments, dtype=np.float32)


def compute_goal_heading(robot_x, robot_y, robot_yaw, goal_x, goal_y):
    """로봇 기준 목표 방향을 (cos(theta), sin(theta))로 만든다."""

    angle_world = math.atan2(goal_y - robot_y, goal_x - robot_x)
    angle_local = math.atan2(
        math.sin(angle_world - robot_yaw),
        math.cos(angle_world - robot_yaw),
    )
    return np.array([math.cos(angle_local), math.sin(angle_local)], dtype=np.float32)


def get_policy_video_keys(policy):
    """checkpoint processor가 요구하는 video key를 읽는다."""

    try:
        video_cfg = policy.modality_configs["video"]
        keys = list(video_cfg.modality_keys)
        if keys:
            return keys
    except Exception:
        pass
    return ["ego_view"]


def load_input_image(image_source):
    """파일 경로 또는 base64 문자열에서 RGB 이미지를 읽는다."""

    if Path(image_source).exists():
        img = Image.open(image_source).convert("RGB")
    else:
        raw = base64.b64decode(image_source)
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    return np.array(img, dtype=np.uint8)


def mask_top_half_image(image, fill_value=0):
    """RGB 이미지의 위쪽 절반을 지정한 값으로 마스킹한다."""

    masked = image.copy()
    half_height = masked.shape[0] // 2
    masked[:half_height, :, :] = fill_value
    return masked


def main():
    print(f"[GR00T] Loading model from {MODEL_PATH} ...")
    policy = Gr00tPolicy(
        embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
        model_path=MODEL_PATH,
        device="cuda:0",
        strict=False,
    )

    # policy.get_action()이 호출되기 전에 action head attention processor를 교체해야 한다.
    attn_store = {}
    install_action_cross_attention_capture(policy.model, attn_store)

    print(f"[GR00T] Model loaded. Starting task: goal=({GOAL_X:.2f},{GOAL_Y:.2f})")

    images_np = load_input_image(INPUT_IMAGE_PATH)
    if MASK_TOP_HALF_IMAGE:
        images_np = mask_top_half_image(images_np, fill_value=MASK_FILL_VALUE)
        print(
            f"[GR00T] Masked top half of input image with fill_value={MASK_FILL_VALUE}"
        )

    route = compute_route_segments(ROBOT_X, ROBOT_Y, ROBOT_YAW, GOAL_X, GOAL_Y)
    goal_heading = compute_goal_heading(ROBOT_X, ROBOT_Y, ROBOT_YAW, GOAL_X, GOAL_Y)
    video_keys = get_policy_video_keys(policy)

    obs = build_observation(
        images_np,
        speed=0.0,
        route=route,
        goal_heading=goal_heading,
        language=LAN_PROMPT,
        video_keys=video_keys,
    )

    # 여기서 action head가 실제로 실행되고, CaptureCrossAttnProcessor가 attention map을 저장한다.
    action, _ = policy.get_action(obs)
    print(f"[GR00T] Predicted action keys: {list(action.keys())}")

    print_action_attention_summary(policy, attn_store)
    show_action_attention_overlays(policy, attn_store, obs)


if __name__ == "__main__":
    main()
