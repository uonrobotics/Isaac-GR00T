# ===============================================================
# [ ISAAC SIM SERVER ]
# 역할:
#   - stage/env/robot 초기화
#   - 랜덤 spawn reset
#   - TCP 8765로 reset/get_obs 제공
# ===============================================================

from isaacsim import SimulationApp

simulation_app = SimulationApp(
    {
        "headless": False,
        "fast_shutdown": True,
    }
)

import base64
import copy
import io
import json
import math
import os
import select
import socket
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
from PIL import Image
import carb
import omni.timeline
import omni.usd
import omni.kit.commands
import yaml

from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdSkel
from isaacsim.core.utils.stage import add_reference_to_stage, create_new_stage, is_stage_loading
from isaacsim.storage.native import get_assets_root_path
from isaacsim.sensors.camera import Camera
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.utils.extensions import enable_extension

# ENV_USD_PATH = "/nas/sujinkim/data/goto/sim/goto_warehouse.usd"
ENV_USD_PATH = "/nas/sujinkim/data/goto/sim_v2_env_settings/sim_v2_env.usd"
MAP_PNG_PATH = "/nas/sujinkim/data/goto/sim_v2_env_settings/sim_v2_env.png"
MAP_YAML_PATH = "/nas/sujinkim/data/goto/sim_v2_env_settings/sim_v2_env.yaml"
ROBOT_REL_PATH = "/Isaac/Samples/ROS2/Robots/Nova_Carter_ROS.usd"

ENV_PRIM_PATH = "/World/env"
ROBOT_ROOT_PRIM_PATH = "/World/Nova_Carter_ROS"
ROBOT_BODY_PRIM_PATH = "/World/Nova_Carter_ROS/chassis_link"
SPAWN_AREA_PRIM_PATH = "/World/env/spawn_area"
DYNAMIC_OBSTACLE_PATH = "/World/env/dynamic_obstacle/obs_001"
ROBOT_CAMERA_ROS_GRAPH_PATHS = [
    "/World/Nova_Carter_ROS/front_hawk",
    "/World/Nova_Carter_ROS/right_hawk",
    "/World/Nova_Carter_ROS/left_hawk",
    "/World/Nova_Carter_ROS/back_hawk",
    "/World/Nova_Carter_ROS/front_owl",
    "/World/Nova_Carter_ROS/back_owl",
    "/World/Nova_Carter_ROS/left_owl",
    "/World/Nova_Carter_ROS/right_owl",
]
DEFAULT_Z = 0.0
CAMERA_SETTLE_FRAMES = 6
CAMERA_MAX_WAIT_FRAMES = 45
CAMERA_MIN_VALID_MEAN = 1.0
DEFAULT_DYNAMIC_OBSTACLE_ANIMATION_SPEED = 2.0
DEFAULT_DYNAMIC_OBSTACLE_ANIMATION_FPS = 30.0
DEFAULT_DYNAMIC_OBSTACLES = [
    {
        "enabled": True,
        "name": "moving_person1",
        "prim_path": "/World/env/moving_person1",
        "animation_usd": "/nas/sujinkim/data/goto/sim_v2_env_settings/walk_1.skelanim.usd",
        "use_animation_graph": True,
        "clear_existing_animation_graph": True,
        "animation_speed": 2.0,
        "trajectory": "loop",
        "speed_mps": 0.8,
        "pause_sec": 1.0,
        "waypoints_local": [
            [0.0, 0.0, 0.0],
            [3.0, 6.0, 0.0],
            [5.0, 2.5, 0.0],
            [1.5, 2.5, 0.0],
        ],
    },
    {
        "enabled": True,
        "name": "moving_person2",
        "prim_path": "/World/env/moving_person2",
        "animation_usd": "/nas/sujinkim/data/goto/sim_v2_env_settings/walk_2.skelanim.usd",
        "use_animation_graph": True,
        "clear_existing_animation_graph": True,
        "animation_speed": 2.0,
        "trajectory": "pingpong_polyline",
        "speed_mps": 0.6,
        "pause_sec": 0.5,
        "yaw_offset_deg": -90.0,
        "waypoints_local": [
            [0.0, 0.0, 0.0],
            [-2.0, 1.0, 0.0],
            [-3.5, 2.5, 0.0],
            [-1.0, 4.0, 0.0],
        ],
    },
    {
        "enabled": True,
        "name": "moving_person3",
        "prim_path": "/World/env/moving_person3",
        "animation_usd": "/nas/sujinkim/data/goto/sim_v2_env_settings/walk_1.skelanim.usd",
        "use_animation_graph": True,
        "clear_existing_animation_graph": True,
        "animation_speed": 2.0,
        "trajectory": "loop",
        "speed_mps": 0.7,
        "pause_sec": 1.5,
        "waypoints_local": [
            [0.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
            [-1.5, 1.5, 0.0],
            [1.5, 1.5, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, -3.0, 0.0],
            [-1.5, -1.5, 0.0],
            [1.5, -1.5, 0.0],
        ],
    },
]

DEFAULT_CAMERA_PRESET = "gemini_336"
DEFAULT_CAMERA_LAYOUT = "default"
GEMINI336L_HIGH_DUAL_VIEW_LAYOUT = "gemini336l_high_dual_view"
GEMINI336L_HIGH_DUAL_VIEW_CONCAT_LAYOUT = "gemini336l_high_dual_view_concat"
DEFAULT_MULTIVIEW_YAW_STEP_DEG = 70.0
GEMINI336L_DRIVEWAY_MOUNT = {
    "center_camera_x": 0.20,
    "side_camera_x": 0.19,
    "camera_z": 0.33,
    "side_camera_y_offset": 0.12,
    "side_yaw_deg": 50.0,
    "center_down_tilt_deg": 10.0,
}
GEMINI336L_HIGH_DUAL_LANDSCAPE_MOUNT = {
    "left_camera_x": 0.11,
    "left_camera_y": 0.05,
    "right_camera_x": 0.11,
    "right_camera_y": -0.05,
    "camera_z": 1.40,
    "left_yaw_deg": 43.0,
    "right_yaw_deg": -43.0,
    "down_tilt_deg": 25.0,
}
GEMINI336L_HIGH_DUAL_PORTRAIT_MOUNT = {
    "left_camera_x": 0.11,
    "left_camera_y": 0.05,
    "right_camera_x": 0.11,
    "right_camera_y": -0.05,
    "camera_z": 1.40,
    "left_yaw_deg": 30.0,
    "right_yaw_deg": -30.0,
    "down_tilt_deg": 35.0,
}

CAMERA_PRESETS = {
    # Orbbec Gemini 336 RGB. Official RGB FOV H86 x V55 deg.
    "gemini_336": {
        "resolution": (640, 360),
        "fov_deg": 86.0,
        "vertical_fov_deg": 55.0,
        "multiview_yaw_step_deg": DEFAULT_MULTIVIEW_YAW_STEP_DEG,
        "offset_xyz": [0.20, 0.0, 0.33],
        "clipping_range": (0.01, 1000.0),
        "horizontal_aperture": 20.955,
    },
    # Orbbec Gemini 336L RGB. Official RGB FOV H94 x V68 deg.
    "gemini_336l": {
        "resolution": (640, 400),
        "fov_deg": 94.0,
        "vertical_fov_deg": 68.0,
        "offset_xyz": [0.20, 0.0, 0.33],
        "clipping_range": (0.01, 1000.0),
        "horizontal_aperture": 20.955,
    },
    # Orbbec Gemini 336L RGB, physically portrait-mounted. Projection axes are swapped: H68 x V94 deg.
    "gemini_336l_portrait": {
        "resolution": (400, 640),
        "fov_deg": 68.0,
        "vertical_fov_deg": 94.0,
        "offset_xyz": [0.20, 0.0, 0.33],
        "clipping_range": (0.01, 1000.0),
        "horizontal_aperture": None,
        "horizontal_aperture_from_landscape_preset": "gemini_336l",
    },
    # Orbbec Gemini 345Lg color camera. Official Color FOV H137 x V71 deg.
    "gemini_345lg": {
        "resolution": (640, 360),
        "fov_deg": 137.0,
        "vertical_fov_deg": 71.0,
        "offset_xyz": [0.20, 0.0, 0.33],
        "clipping_range": (0.01, 1000.0),
        "horizontal_aperture": 20.955,
    },
}


def vertical_aperture_for_camera_preset(cfg: dict) -> float:
    if cfg.get("vertical_fov_deg") is None:
        return cfg["horizontal_aperture"] * (cfg["resolution"][1] / cfg["resolution"][0])

    focal = cfg["horizontal_aperture"] / (
        2.0 * math.tan(math.radians(float(cfg["fov_deg"])) * 0.5)
    )
    return 2.0 * focal * math.tan(math.radians(float(cfg["vertical_fov_deg"])) * 0.5)


def camera_preset_config(preset_name: str) -> dict:
    preset = copy.deepcopy(CAMERA_PRESETS[preset_name])
    source_preset = preset.get("horizontal_aperture_from_landscape_preset")
    if source_preset:
        preset["horizontal_aperture"] = vertical_aperture_for_camera_preset(
            CAMERA_PRESETS[source_preset]
        )
    return preset


def build_camera_config(preset_name: str = DEFAULT_CAMERA_PRESET, yaw_deg: float = 0.0) -> dict:
    preset = camera_preset_config(preset_name)
    preset.update(
        {
            "name": f"cam_{preset_name}",
            "camera_prim_path": "/World/replay_camera/front_camera",
            "rot_xyz_deg": [90.0, -90.0, 0.0],
            "yaw_offset_deg": float(yaw_deg),
        }
    )
    return preset


VIEW_YAW_SIGNS = {
    "ego_view": 0.0,
    "left_view": 1.0,
    "right_view": -1.0,
}

VIEW_CAMERA_NAMES = {
    "ego_view": "front",
    "left_view": "left",
    "right_view": "right",
}


def build_multiview_camera_configs(
    preset_name: str = DEFAULT_CAMERA_PRESET,
    multiview_yaw_step_deg: float | None = None,
    layout: str = DEFAULT_CAMERA_LAYOUT,
) -> dict:
    if layout == "gemini336l_driveway_view":
        if preset_name != "gemini_336l":
            raise ValueError(f"{layout} only supports camera_preset='gemini_336l'")
        return build_gemini336l_driveway_camera_configs()
    if layout in {GEMINI336L_HIGH_DUAL_VIEW_LAYOUT, GEMINI336L_HIGH_DUAL_VIEW_CONCAT_LAYOUT}:
        if preset_name not in {"gemini_336l", "gemini_336l_portrait"}:
            raise ValueError(
                f"{layout} only supports camera_preset='gemini_336l' or "
                "camera_preset='gemini_336l_portrait'"
            )
        return build_gemini336l_high_dual_camera_configs(preset_name)
    if layout != DEFAULT_CAMERA_LAYOUT:
        raise ValueError(f"Unknown camera_layout: {layout}")

    preset = camera_preset_config(preset_name)
    yaw_step = (
        multiview_yaw_step_deg
        if multiview_yaw_step_deg is not None
        else preset.get("multiview_yaw_step_deg", DEFAULT_MULTIVIEW_YAW_STEP_DEG)
    )
    configs = {}
    for view_name, yaw_sign in VIEW_YAW_SIGNS.items():
        cfg = copy.deepcopy(preset)
        cfg.update(
            {
                "name": f"cam_{view_name}",
                "camera_prim_path": f"/World/replay_camera/{VIEW_CAMERA_NAMES[view_name]}_camera",
                # Match mobile_robot_datagen/src/rgb_generator.py:
                # Isaac camera orientation is fixed and horizontal pan is applied
                # as a yaw offset around the robot heading.
                "rot_xyz_deg": [90.0, -90.0, 0.0],
                "yaw_offset_deg": float(yaw_sign * yaw_step),
            }
        )
        configs[view_name] = cfg
    return configs


def build_gemini336l_driveway_camera_configs() -> dict:
    preset = camera_preset_config("gemini_336l")
    mount = GEMINI336L_DRIVEWAY_MOUNT
    forward_camera_quat = IsaacSimServer.euler_xyz_deg_to_quat_xyzw(90.0, -90.0, 0.0)
    center_camera_quat = IsaacSimServer.quat_multiply_xyzw(
        forward_camera_quat,
        IsaacSimServer.euler_xyz_deg_to_quat_xyzw(0.0, 0.0, -90.0),
    )
    center_camera_quat = IsaacSimServer.quat_multiply_xyzw(
        IsaacSimServer.euler_xyz_deg_to_quat_xyzw(
            0.0,
            mount["center_down_tilt_deg"],
            0.0,
        ),
        center_camera_quat,
    )

    mounts = {
        "left_view": {
            "camera_name": "left",
            "offset_xyz": [
                mount["side_camera_x"],
                mount["side_camera_y_offset"],
                mount["camera_z"],
            ],
            "yaw_offset_deg": mount["side_yaw_deg"],
        },
        "ego_view": {
            "camera_name": "center",
            "offset_xyz": [
                mount["center_camera_x"],
                0.0,
                mount["camera_z"],
            ],
            "yaw_offset_deg": 0.0,
            "rot_quat_xyzw": center_camera_quat,
        },
        "right_view": {
            "camera_name": "right",
            "offset_xyz": [
                mount["side_camera_x"],
                -mount["side_camera_y_offset"],
                mount["camera_z"],
            ],
            "yaw_offset_deg": -mount["side_yaw_deg"],
        },
    }

    configs = {}
    for view_name, camera_mount in mounts.items():
        cfg = copy.deepcopy(preset)
        cfg.update(
            {
                "name": f"cam_{camera_mount['camera_name']}",
                "camera_prim_path": (
                    f"/World/replay_camera/{camera_mount['camera_name']}_camera"
                ),
                "rot_xyz_deg": [90.0, -90.0, 0.0],
                **camera_mount,
            }
        )
        cfg.pop("camera_name")
        configs[view_name] = cfg
    return configs


def tilted_camera_quat_xyzw(down_tilt_deg: float):
    forward_camera_quat = IsaacSimServer.euler_xyz_deg_to_quat_xyzw(90.0, -90.0, 0.0)
    return IsaacSimServer.quat_multiply_xyzw(
        IsaacSimServer.euler_xyz_deg_to_quat_xyzw(0.0, down_tilt_deg, 0.0),
        forward_camera_quat,
    )


def build_gemini336l_high_dual_camera_configs(preset_name: str) -> dict:
    preset = camera_preset_config(preset_name)
    mount = (
        GEMINI336L_HIGH_DUAL_PORTRAIT_MOUNT
        if preset_name == "gemini_336l_portrait"
        else GEMINI336L_HIGH_DUAL_LANDSCAPE_MOUNT
    )
    camera_quat = tilted_camera_quat_xyzw(mount["down_tilt_deg"])
    mounts = {
        "left_view": {
            "camera_name": "left",
            "offset_xyz": [
                mount["left_camera_x"],
                mount["left_camera_y"],
                mount["camera_z"],
            ],
            "yaw_offset_deg": mount["left_yaw_deg"],
        },
        "right_view": {
            "camera_name": "right",
            "offset_xyz": [
                mount["right_camera_x"],
                mount["right_camera_y"],
                mount["camera_z"],
            ],
            "yaw_offset_deg": mount["right_yaw_deg"],
        },
    }

    configs = {}
    for view_name, camera_mount in mounts.items():
        cfg = copy.deepcopy(preset)
        cfg.update(
            {
                "name": f"cam_{camera_mount['camera_name']}",
                "camera_prim_path": (
                    f"/World/replay_camera/{camera_mount['camera_name']}_camera"
                ),
                "rot_xyz_deg": [90.0, -90.0, 0.0],
                "rot_quat_xyzw": camera_quat,
                **camera_mount,
            }
        )
        cfg.pop("camera_name")
        configs[view_name] = cfg
    return configs


def add_physics_scene(stage):
    scene_path = "/physicsScene"
    if stage.GetPrimAtPath(scene_path).IsValid():
        return
    scene = UsdPhysics.Scene.Define(stage, scene_path)
    scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
    scene.CreateGravityMagnitudeAttr().Set(9.81)
    physx_scene = PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath(scene_path))
    physx_scene.CreateEnableCCDAttr(True)
    physx_scene.CreateEnableGPUDynamicsAttr(False)
    physx_scene.CreateBroadphaseTypeAttr("MBP")


def add_dome_light(stage):
    dome_path = "/World/DomeLight"
    if stage.GetPrimAtPath(dome_path).IsValid():
        return
    dome = UsdLux.DomeLight.Define(stage, dome_path)
    dome.CreateIntensityAttr(1000)


def get_valid_prim(stage, prim_path: str, name: str):
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"{name} prim not found: {prim_path}")
    return prim


def get_world_xy_yaw(stage, prim_path: str):
    prim = get_valid_prim(stage, prim_path, prim_path)
    xform = UsdGeom.Xformable(prim)
    mat = xform.ComputeLocalToWorldTransform(0)
    pos = mat.ExtractTranslation()
    world_forward = mat.TransformDir(Gf.Vec3d(1.0, 0.0, 0.0))
    yaw = math.atan2(float(world_forward[1]), float(world_forward[0]))
    return float(pos[0]), float(pos[1]), float(yaw)


def get_world_bbox_xy(stage, prim_path: str):
    prim = get_valid_prim(stage, prim_path, prim_path)
    bbox_cache = UsdGeom.BBoxCache(0, ["default"])
    bound = bbox_cache.ComputeWorldBound(prim)
    box = bound.ComputeAlignedBox()
    min_pt = box.GetMin()
    max_pt = box.GetMax()
    return float(min_pt[0]), float(max_pt[0]), float(min_pt[1]), float(max_pt[1])


def quat_wxyz_from_yaw(yaw_rad: float):
    half = yaw_rad * 0.5
    return np.array([math.cos(half), 0.0, 0.0, math.sin(half)], dtype=np.float32)


def ensure_translate_op(prim):
    xformable = UsdGeom.Xformable(prim)
    for op in xformable.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            return op
    return xformable.AddTranslateOp()


def find_first_prim_of_type(root_prim, type_name: str):
    if root_prim.GetTypeName() == type_name:
        return root_prim
    for child in root_prim.GetChildren():
        found = find_first_prim_of_type(child, type_name)
        if found is not None:
            return found
    return None


def summarize_child_prim_types(root_prim, limit: int = 20):
    rows = []
    for prim in Usd.PrimRange(root_prim):
        rows.append(f"{prim.GetPath()}:{prim.GetTypeName() or 'untyped'}")
        if len(rows) >= limit:
            break
    return rows


def sample_xy_in_region(region, margin=0.3):
    xmin, xmax, ymin, ymax = region
    x = np.random.uniform(xmin + margin, xmax - margin)
    y = np.random.uniform(ymin + margin, ymax - margin)
    return float(x), float(y)


class OccupancyDistanceMap:
    def __init__(self, map_png_path, map_yaml_path):
        with open(map_yaml_path, "r") as f:
            cfg = yaml.safe_load(f)
        self.resolution = float(cfg["resolution"])
        self.origin_x = float(cfg["origin"][0])
        self.origin_y = float(cfg["origin"][1])

        img = Image.open(map_png_path).convert("L")
        self.map_img = np.array(img)
        self.occupied = self.map_img < 128
        self.height, self.width = self.occupied.shape

        from scipy.ndimage import distance_transform_edt
        free_mask = ~self.occupied
        dist_pixels = distance_transform_edt(free_mask)
        self.dist_meters = dist_pixels * self.resolution

    def world_to_map_rc(self, x, y):
        mx = int((x - self.origin_x) / self.resolution)
        my = int((y - self.origin_y) / self.resolution)
        row = self.height - 1 - my
        col = mx
        return row, col

    def is_inside(self, x, y):
        row, col = self.world_to_map_rc(x, y)
        return 0 <= row < self.height and 0 <= col < self.width

    def is_free(self, x, y):
        if not self.is_inside(x, y):
            return False
        row, col = self.world_to_map_rc(x, y)
        return not self.occupied[row, col]

    def clearance(self, x, y):
        if not self.is_inside(x, y):
            return 0.0
        row, col = self.world_to_map_rc(x, y)
        return float(self.dist_meters[row, col])


def sample_conditioned_spawn(region, occ_map, min_clearance=1.5, max_trials=50):
    for _ in range(max_trials):
        x, y = sample_xy_in_region(region, margin=0.3)
        if not occ_map.is_inside(x, y):
            continue
        if not occ_map.is_free(x, y):
            continue
        if occ_map.clearance(x, y) < min_clearance:
            continue
        yaw = np.random.uniform(-math.pi, math.pi)
        return x, y, float(yaw)
    raise RuntimeError("Failed to sample valid spawn pose.")


# ===============================================================
# JSON Socket Server
# ===============================================================
class JsonSocketServer:
    """
    Non-blocking TCP server that:
    - accepts one client at a time
    - reads newline-delimited JSON messages from the client
    - handles partial recv correctly by accumulating into self.buffer
    - auto-drops client on any socket error
    """

    def __init__(self, host="127.0.0.1", port=8765):
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind((host, port))
        self.server.listen(1)
        self.server.setblocking(False)
        self.client: Optional[socket.socket] = None
        self.buffer = b""

    def _drop_client(self, reason: str = ""):
        if self.client is not None:
            try:
                self.client.close()
            except Exception:
                pass
            self.client = None
            self.buffer = b""
            if reason:
                print(f"[SIM IPC] client dropped: {reason}")

    def poll_accept(self):
        if self.client is not None:
            return
        readable, _, _ = select.select([self.server], [], [], 0.0)
        if readable:
            try:
                conn, addr = self.server.accept()
                conn.setblocking(False)
                # Large receive buffer for image payloads
                conn.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
                self.client = conn
                self.buffer = b""
                print(f"[SIM IPC] client connected from {addr}")
            except OSError as e:
                print(f"[SIM IPC] accept error: {e}")

    def recv_message(self) -> Optional[dict]:
        self.poll_accept()
        if self.client is None:
            return None

        # Check readability without blocking
        try:
            readable, _, exceptional = select.select([self.client], [], [self.client], 0.0)
        except (OSError, ValueError) as e:
            self._drop_client(f"select error: {e}")
            return None

        if exceptional:
            self._drop_client("socket exception flag")
            return None

        if not readable:
            return None

        # Drain all available data in a loop to handle partial sends
        try:
            while True:
                chunk = self.client.recv(1 << 20)  # 1 MB chunks
                if not chunk:
                    self._drop_client("client closed connection")
                    return None
                self.buffer += chunk
                # Check if more data is immediately available
                r, _, _ = select.select([self.client], [], [], 0.0)
                if not r:
                    break
        except BlockingIOError:
            pass  # No more data right now — normal
        except (ConnectionResetError, BrokenPipeError, OSError) as e:
            self._drop_client(f"recv error: {e}")
            return None

        # Parse one complete newline-delimited JSON message
        if b"\n" not in self.buffer:
            return None

        line, self.buffer = self.buffer.split(b"\n", 1)
        line = line.strip()
        if not line:
            return None

        try:
            return json.loads(line.decode("utf-8"))
        except json.JSONDecodeError as e:
            print(f"[SIM IPC] JSON decode error: {e} | raw={line[:120]}")
            return None

    def send_message(self, payload: dict) -> bool:
        if self.client is None:
            return False
        try:
            data = (json.dumps(payload) + "\n").encode("utf-8")
            self.client.sendall(data)
            return True
        except (ConnectionResetError, BrokenPipeError, OSError) as e:
            self._drop_client(f"send error: {e}")
            return False

    def close(self):
        self._drop_client()
        try:
            self.server.close()
        except Exception:
            pass


# ===============================================================
# IsaacSim Server
# ===============================================================
class IsaacSimServer:
    def __init__(
        self,
        camera_mode: str = "single",
        camera_preset: str = DEFAULT_CAMERA_PRESET,
        camera_layout: str = DEFAULT_CAMERA_LAYOUT,
        camera_yaw_deg: float = 0.0,
        multiview_yaw_step_deg: float | None = None,
        enable_ros2_bridge: bool = False,
    ):
        self.stage = None
        self.timeline = None
        self.robot = None
        self.camera = None
        self.cameras = {}
        self.dynamic_obstacles = []
        self.dynamic_obstacles_running = True
        self.animation_time_range = None
        self.animation_time_scale = DEFAULT_DYNAMIC_OBSTACLE_ANIMATION_SPEED
        self.animation_time_code = None
        self.enable_ros2_bridge = enable_ros2_bridge
        self.camera_mode = camera_mode
        self.camera_preset = camera_preset
        self.camera_layout = camera_layout
        if camera_mode == "single":
            if camera_layout != DEFAULT_CAMERA_LAYOUT:
                raise ValueError("Non-default camera layouts require camera_mode='multiview'")
            self.camera_configs = {
                "ego_view": build_camera_config(camera_preset, camera_yaw_deg),
            }
        elif camera_mode == "multiview":
            self.camera_configs = build_multiview_camera_configs(
                camera_preset,
                multiview_yaw_step_deg,
                camera_layout,
            )
        else:
            raise ValueError(f"Unknown camera_mode: {camera_mode}")
        self.primary_camera_view = (
            "ego_view" if "ego_view" in self.camera_configs else next(iter(self.camera_configs))
        )
        self.camera_cfg = self.camera_configs[self.primary_camera_view]
        self.spawn_region = None
        self.occ_map = OccupancyDistanceMap(
            MAP_PNG_PATH,
            MAP_YAML_PATH,
        )

        self.obs_prim = None
        self.obs_translate_op = None
        self.obs_base_y = -5.2
        self.obs_base_z = 0.5
        self.obs_center_x = -12.0
        self.obs_amplitude = 4.0
        self.obs_speed = 0.6
        self.obs_phase = 0.0

    def setup(self):
        self._enable_extensions()
        self._disable_replicator_capture()
        self._create_stage_once()
        self._load_env_and_robot_once()
        self._disable_replicator_capture()
        self._setup_dynamic_obstacles_once()
        self._start_simulation_once()
        self._disable_replicator_capture()
        self._initialize_articulation_once()
        self._setup_camera_once()
        self._disable_replicator_capture()

    def _disable_replicator_capture(self):
        return

    def _enable_extensions(self):
        enable_extension("omni.physx")
        enable_extension("omni.physx.ui")
        enable_extension("omni.graph.nodes")
        enable_extension("isaacsim.core.nodes")
        if self.enable_ros2_bridge:
            enable_extension("isaacsim.ros2.bridge")
        enable_extension("isaacsim.sensors.rtx")
        enable_extension("omni.anim.timeline")
        enable_extension("omni.anim.graph.core")
        enable_extension("omni.anim.graph.bundle")
        enable_extension("omni.anim.retarget.core")
        simulation_app.update()
        simulation_app.update()

    def _create_stage_once(self):
        create_new_stage()
        self.stage = omni.usd.get_context().get_stage()
        UsdGeom.SetStageMetersPerUnit(self.stage, 1.0)
        add_physics_scene(self.stage)
        add_dome_light(self.stage)

    def _load_env_and_robot_once(self):
        assets_root = get_assets_root_path()
        robot_usd = assets_root + ROBOT_REL_PATH
        add_reference_to_stage(usd_path=ENV_USD_PATH, prim_path=ENV_PRIM_PATH)
        add_reference_to_stage(usd_path=robot_usd, prim_path=ROBOT_ROOT_PRIM_PATH)
        if self.enable_ros2_bridge:
            self._deactivate_robot_camera_ros_graphs()
            self._remove_stage_replicator_ros_writers()
        simulation_app.update()
        if self.enable_ros2_bridge:
            self._deactivate_robot_camera_ros_graphs()
            self._remove_stage_replicator_ros_writers()
        simulation_app.update()
        if not self.enable_ros2_bridge:
            self._deactivate_robot_ros_graphs()
        else:
            self._deactivate_robot_camera_ros_graphs()
            self._remove_stage_replicator_ros_writers()
        self.spawn_region = get_world_bbox_xy(self.stage, SPAWN_AREA_PRIM_PATH)

    def _deactivate_robot_camera_ros_graphs(self):
        disabled = []
        for prim_path in ROBOT_CAMERA_ROS_GRAPH_PATHS:
            prim = self.stage.GetPrimAtPath(prim_path)
            if prim.IsValid() and prim.IsActive():
                prim.SetActive(False)
                disabled.append(prim_path)
        if disabled:
            print(
                f"[SIM SERVER] ROS2 bridge enabled; deactivated {len(disabled)} "
                "built-in Carter camera ROS graphs"
            )

    def _remove_stage_replicator_ros_writers(self):
        if self.stage is None:
            return 0
        removed = []
        for prim in Usd.PrimRange(self.stage.GetPseudoRoot()):
            path = str(prim.GetPath())
            path_lower = path.lower()
            name_lower = prim.GetName().lower()
            if (
                "/render/postprocess/sdgpipeline" in path_lower
                and ("nodewriter" in name_lower or "writer" in name_lower)
            ):
                removed.append(path)
        for path in sorted(removed, key=len, reverse=True):
            self.stage.RemovePrim(path)
        if removed:
            print(
                f"[SIM SERVER] removed {len(removed)} global "
                "replicator ROS writer prims"
            )
        return len(removed)

    def _deactivate_robot_ros_graphs(self):
        robot_prim = self.stage.GetPrimAtPath(ROBOT_ROOT_PRIM_PATH)
        if not robot_prim.IsValid():
            return
        disabled = []
        for prim in Usd.PrimRange(robot_prim):
            if prim == robot_prim:
                continue
            name_lower = prim.GetName().lower()
            type_lower = prim.GetTypeName().lower()
            if (
                type_lower in {"omnigraph", "nodegraph"}
                or "actiongraph" in name_lower
                or "ros" in name_lower
                or "render_product" in name_lower
                or "isaac_read_imu" in name_lower
                or "compute_odometry" in name_lower
                or "differential_drive" in name_lower
                or name_lower.endswith("_hawk")
                or name_lower == "chassis_imu"
            ):
                prim.SetActive(False)
                disabled.append(str(prim.GetPath()))
        if disabled:
            print(
                f"[SIM SERVER] ROS2 bridge disabled; deactivated {len(disabled)} "
                "robot ROS/graph prims"
            )

    def _start_simulation_once(self):
        self.timeline = omni.timeline.get_timeline_interface()
        self._disable_replicator_capture()
        self.timeline.play()
        for _ in range(20):
            self._disable_replicator_capture()
            simulation_app.update()

    def _initialize_articulation_once(self):
        self.robot = Articulation(ROBOT_ROOT_PRIM_PATH)
        self.robot.initialize()
        for _ in range(10):
            simulation_app.update()

    def _setup_dynamic_obstacles_once(self):
        for cfg in DEFAULT_DYNAMIC_OBSTACLES:
            if not cfg.get("enabled", True):
                continue

            prim = self.stage.GetPrimAtPath(cfg["prim_path"])
            if not prim.IsValid():
                print(f"[WARN] Dynamic obstacle prim not found: {cfg['prim_path']}")
                continue

            UsdPhysics.CollisionAPI.Apply(prim)
            translate_op = ensure_translate_op(prim)
            start = self._get_dynamic_obstacle_center(prim, translate_op)
            state = {
                "cfg": cfg,
                "prim": prim,
                "translate_op": translate_op,
                "start": start,
                "base_position": start.copy(),
                "current_pos": start.copy(),
                "segment_idx": 0,
                "pause_left": 0.0,
                "direction": 1,
            }
            self._apply_dynamic_obstacle_animation(cfg, prim)
            self.dynamic_obstacles.append(state)
            self._set_dynamic_obstacle_position(state, start)
            print(
                f"[DYNAMIC_OBSTACLE] {cfg['name']} start={start.tolist()} "
                f"trajectory={cfg.get('trajectory')} speed={cfg.get('speed_mps')}"
            )

    def _get_dynamic_obstacle_center(self, prim, translate_op):
        value = translate_op.Get()
        if value is not None:
            return np.array([float(value[0]), float(value[1]), float(value[2])])
        xform = UsdGeom.Xformable(prim)
        pos = xform.ComputeLocalToWorldTransform(0).ExtractTranslation()
        return np.array([float(pos[0]), float(pos[1]), float(pos[2])])

    def _set_dynamic_obstacle_position(self, state: dict, xyz):
        state["translate_op"].Set(
            Gf.Vec3d(float(xyz[0]), float(xyz[1]), float(xyz[2]))
        )

    def _face_velocity(self, state: dict, velocity_xy):
        vx, vy = float(velocity_xy[0]), float(velocity_xy[1])
        if math.hypot(vx, vy) < 1e-4:
            return
        yaw = math.atan2(vy, vx)
        yaw += math.radians(float(state["cfg"].get("yaw_offset_deg", 0.0)))
        xform = UsdGeom.Xformable(state["prim"])
        rotate_op = None
        for op in xform.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeRotateZ:
                rotate_op = op
                break
        if rotate_op is None:
            rotate_op = xform.AddRotateZOp()
        rotate_op.Set(math.degrees(yaw))

    def _move_towards(self, state: dict, target, dt: float):
        cfg = state["cfg"]
        pos = state["current_pos"]
        target = np.asarray(target, dtype=float)
        speed = max(0.0, float(cfg.get("speed_mps", 0.7)))
        delta = target - pos
        dist = float(np.linalg.norm(delta[:2]))
        if dist < 1e-4 or speed <= 1e-6:
            state["current_pos"] = target.copy()
            self._set_dynamic_obstacle_position(state, state["current_pos"])
            return True
        step = speed * max(0.0, float(dt))
        if step >= dist:
            new_pos = target.copy()
            reached = True
        else:
            new_pos = pos + delta / max(dist, 1e-6) * step
            reached = False
        velocity = new_pos - pos
        state["current_pos"] = new_pos
        self._set_dynamic_obstacle_position(state, new_pos)
        self._face_velocity(state, velocity[:2])
        return reached

    def _resolve_waypoints_for_state(self, state: dict):
        cfg = state["cfg"]
        base = state["base_position"]
        if "waypoints_local" in cfg:
            return [base + np.asarray(offset, dtype=float) for offset in cfg["waypoints_local"]]
        return [np.asarray(p, dtype=float) for p in cfg.get("waypoints", [])]

    def update_dynamic_obstacles(self, dt: float):
        if not self.dynamic_obstacles_running:
            return
        for state in self.dynamic_obstacles:
            cfg = state["cfg"]
            if state["pause_left"] > 0.0:
                state["pause_left"] = max(0.0, state["pause_left"] - dt)
                continue
            traj = cfg.get("trajectory", "loop")
            waypoints = self._resolve_waypoints_for_state(state)
            if len(waypoints) < 2:
                continue
            idx = int(state["segment_idx"])
            reached = self._move_towards(state, waypoints[idx], dt)
            if not reached:
                continue
            state["pause_left"] = float(cfg.get("pause_sec", 0.0))
            if traj == "loop":
                state["segment_idx"] = (idx + 1) % len(waypoints)
            elif traj == "pingpong_polyline":
                direction = int(state.get("direction", 1))
                next_idx = idx + direction
                if next_idx >= len(waypoints):
                    direction = -1
                    next_idx = len(waypoints) - 2
                elif next_idx < 0:
                    direction = 1
                    next_idx = 1
                state["direction"] = direction
                state["segment_idx"] = next_idx

    def _resolve_asset_path(self, path: str) -> str:
        if path.startswith(("omniverse://", "http://", "https://")):
            return path
        if os.path.isabs(path) and os.path.exists(path):
            return path
        if path.startswith("/Isaac/"):
            assets_root = get_assets_root_path()
            if not assets_root:
                raise RuntimeError(f"Could not resolve Isaac assets root for {path}")
            return assets_root + path
        return path

    def _apply_dynamic_obstacle_animation(self, cfg: Dict[str, Any], prim):
        animation_usd = cfg.get("animation_usd")
        if not animation_usd:
            return
        if cfg.get("clear_existing_animation_graph", False):
            self._clear_dynamic_obstacle_animation_graph_targets(prim)
        skel_root = find_first_prim_of_type(prim, "SkelRoot")
        if skel_root is None:
            print(f"[WARN] Dynamic obstacle has no SkelRoot: {prim.GetPath()}")
            return
        animation_path = self._resolve_asset_path(str(animation_usd))
        anim_root_path = f"{skel_root.GetPath()}/DynamicObstacleAnimation"
        anim_root = self.stage.OverridePrim(anim_root_path)
        anim_root.GetReferences().ClearReferences()
        anim_root.GetReferences().AddReference(animation_path)
        for _ in range(240):
            simulation_app.update()
            if not is_stage_loading():
                break
        anim_graph = find_first_prim_of_type(anim_root, "AnimationGraph")
        if cfg.get("use_animation_graph", True) and anim_graph is not None:
            omni.kit.commands.execute(
                "ApplyAnimationGraphAPICommand",
                paths=[skel_root.GetPath()],
                animation_graph_path=anim_graph.GetPath(),
            )
            print(f"[DYNAMIC_OBSTACLE] animation graph applied: {animation_path}")
            return
        skel_anim = find_first_prim_of_type(anim_root, "SkelAnimation")
        if skel_anim is None and anim_root.GetAttribute("joints").IsValid():
            anim_root.SetTypeName("SkelAnimation")
            skel_anim = anim_root
        if skel_anim is None:
            print(
                f"[WARN] No SkelAnimation in {animation_path}: "
                f"{summarize_child_prim_types(anim_root)}"
            )
            return
        if cfg.get("use_animation_graph", True) and self._apply_skel_animation_to_idle_graph(
            skel_root, prim, skel_anim, cfg
        ):
            return
        skeleton = find_first_prim_of_type(prim, "Skeleton")
        if skeleton is None:
            print(f"[WARN] Dynamic obstacle has no Skeleton: {prim.GetPath()}")
            return
        self._bind_skel_animation(skel_root, skeleton, skel_anim)
        self._set_timeline_for_skel_animation(skel_anim, cfg)

    def _clear_dynamic_obstacle_animation_graph_targets(self, prim):
        for child in Usd.PrimRange(prim):
            for rel in child.GetRelationships():
                if rel.GetName().endswith("animationGraph"):
                    rel.SetTargets([])

    def _apply_skel_animation_to_idle_graph(self, skel_root, character_prim, skel_anim, cfg):
        anim_graph = find_first_prim_of_type(character_prim, "AnimationGraph")
        if anim_graph is None:
            return False
        state_machine = find_first_prim_of_type(anim_graph, "StateMachine")
        if state_machine is None:
            return False
        idle_state = self.stage.GetPrimAtPath(f"{state_machine.GetPath()}/Idle")
        if not idle_state.IsValid():
            return False
        clip_path = f"{idle_state.GetPath()}/DynamicObstacleWalkClip"
        clip_prim = self.stage.GetPrimAtPath(clip_path)
        if not clip_prim.IsValid():
            omni.kit.commands.execute(
                "CreatePrimCommand",
                prim_type="AnimationClip",
                prim_path=clip_path,
                select_new_prim=False,
            )
            clip_prim = self.stage.GetPrimAtPath(clip_path)
        if not clip_prim.IsValid():
            return False
        self._set_relationship_target(clip_prim, "inputs:animationSource", skel_anim.GetPath())
        start_time, end_time = self._get_skel_animation_time_range(skel_anim)
        clip_start_sec = float(cfg.get("animation_loop_start_sec", start_time / DEFAULT_DYNAMIC_OBSTACLE_ANIMATION_FPS))
        clip_end_sec = float(cfg.get("animation_loop_end_sec", max(clip_start_sec, (end_time - 1.0) / DEFAULT_DYNAMIC_OBSTACLE_ANIMATION_FPS)))
        self._set_attr(clip_prim, "inputs:startTime", clip_start_sec, Sdf.ValueTypeNames.Float)
        self._set_attr(clip_prim, "inputs:endTime", clip_end_sec, Sdf.ValueTypeNames.Float)
        self._set_attr(clip_prim, "inputs:loop", True, Sdf.ValueTypeNames.Bool)
        self._set_attr(clip_prim, "inputs:backwards", False, Sdf.ValueTypeNames.Bool)
        self._set_relationship_target(idle_state, "inputs:pose", clip_prim.GetPath())
        omni.kit.commands.execute(
            "ApplyAnimationGraphAPICommand",
            paths=[skel_root.GetPath()],
            animation_graph_path=anim_graph.GetPath(),
        )
        print(f"[DYNAMIC_OBSTACLE] idle animation clip applied: {skel_anim.GetPath()}")
        return True

    def _set_relationship_target(self, prim, rel_name: str, target_path):
        rel = prim.GetRelationship(rel_name)
        if not rel.IsValid():
            rel = prim.CreateRelationship(rel_name)
        rel.SetTargets([target_path])

    def _set_attr(self, prim, attr_name: str, value, value_type):
        attr = prim.GetAttribute(attr_name)
        if not attr.IsValid():
            attr = prim.CreateAttribute(attr_name, value_type)
        attr.Set(value)

    def _bind_skel_animation(self, skel_root, skeleton, skel_anim):
        skel_anim_path = skel_anim.GetPath()
        for target_prim in (skel_root, skeleton):
            binding_api = UsdSkel.BindingAPI.Apply(target_prim)
            binding_api.CreateAnimationSourceRel().SetTargets([skel_anim_path])

    def _get_skel_animation_time_range(self, skel_anim):
        time_samples = []
        for attr_name in ("rotations", "translations", "scales"):
            attr = skel_anim.GetAttribute(attr_name)
            if attr.IsValid():
                time_samples.extend(attr.GetTimeSamples())
        if not time_samples:
            return 0.0, 0.0
        return min(time_samples), max(time_samples)

    def _set_timeline_for_skel_animation(self, skel_anim, cfg: Dict[str, Any]):
        start_time, end_time = self._get_skel_animation_time_range(skel_anim)
        if end_time <= start_time:
            return
        self.animation_time_scale = max(
            self.animation_time_scale,
            float(cfg.get("animation_speed", DEFAULT_DYNAMIC_OBSTACLE_ANIMATION_SPEED)),
        )
        self.animation_time_range = (start_time, end_time)
        self.animation_time_code = start_time
        self.stage.SetStartTimeCode(start_time)
        self.stage.SetEndTimeCode(end_time)
        self.stage.SetTimeCodesPerSecond(DEFAULT_DYNAMIC_OBSTACLE_ANIMATION_FPS)

    def update_dynamic_obstacle_animation(self, dt: float):
        # Do not drive character animation by changing the global timeline here.
        # Carter's ROS differential drive graph depends on monotonically advancing
        # physics time; forcing current_time each frame can invalidate the PhysX
        # tensor view and make /cmd_vel stop moving the robot.
        return

    def _setup_camera_once(self):
        stage = self.stage
        if not stage.GetPrimAtPath("/World/replay_camera").IsValid():
            UsdGeom.Xform.Define(stage, "/World/replay_camera")

        for view_name, cfg in self.camera_configs.items():
            cam_prim_path = cfg["camera_prim_path"]
            if not stage.GetPrimAtPath(cam_prim_path).IsValid():
                UsdGeom.Camera.Define(stage, cam_prim_path)
            camera = Camera(
                prim_path=cam_prim_path,
                name=cfg["name"],
                frequency=30,
                resolution=cfg["resolution"],
            )
            camera.initialize()

            cam_prim = stage.GetPrimAtPath(cam_prim_path)
            cam_geom = UsdGeom.Camera(cam_prim)
            horizontal_aperture = cfg["horizontal_aperture"]
            focal_length = self.fov_to_focal_length(cfg["fov_deg"], horizontal_aperture)
            vertical_aperture = self.vertical_aperture_from_fov(
                cfg["vertical_fov_deg"],
                focal_length,
            )
            cam_geom.GetHorizontalApertureAttr().Set(horizontal_aperture)
            cam_geom.GetVerticalApertureAttr().Set(vertical_aperture)
            cam_geom.GetFocalLengthAttr().Set(focal_length)
            cam_geom.GetClippingRangeAttr().Set(Gf.Vec2f(*cfg["clipping_range"]))
            self.cameras[view_name] = camera

        self.camera = self.cameras.get("ego_view") or next(iter(self.cameras.values()))
        for _ in range(10):
            simulation_app.update()

        print(
            f"[SIM SERVER] camera_mode={self.camera_mode} camera_preset={self.camera_preset} "
            f"camera_layout={self.camera_layout} "
            f"resolution={self.camera_cfg['resolution']} "
            f"fov=H{self.camera_cfg['fov_deg']} V{self.camera_cfg['vertical_fov_deg']} "
            f"offset={self.camera_cfg['offset_xyz']} "
            f"views={list(self.camera_configs.keys())}"
        )

    @staticmethod
    def fov_to_focal_length(fov_deg: float, aperture: float = 20.955) -> float:
        fov_rad = math.radians(fov_deg)
        return aperture / (2.0 * math.tan(fov_rad / 2.0))

    @staticmethod
    def vertical_aperture_from_fov(vertical_fov_deg: float, focal_length: float) -> float:
        fov_rad = math.radians(vertical_fov_deg)
        return 2.0 * focal_length * math.tan(fov_rad / 2.0)

    @staticmethod
    def yaw_to_quat_xyzw(yaw: float):
        half = yaw * 0.5
        return [0.0, 0.0, math.sin(half), math.cos(half)]

    @staticmethod
    def quat_multiply_xyzw(q1, q2):
        x1, y1, z1, w1 = q1
        x2, y2, z2, w2 = q2
        return [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ]

    @staticmethod
    def euler_xyz_deg_to_quat_xyzw(rx_deg, ry_deg, rz_deg):
        rx, ry, rz = math.radians(rx_deg), math.radians(ry_deg), math.radians(rz_deg)
        cx, sx = math.cos(rx * 0.5), math.sin(rx * 0.5)
        cy, sy = math.cos(ry * 0.5), math.sin(ry * 0.5)
        cz, sz = math.cos(rz * 0.5), math.sin(rz * 0.5)
        return [
            sx * cy * cz + cx * sy * sz,
            cx * sy * cz - sx * cy * sz,
            cx * cy * sz + sx * sy * cz,
            cx * cy * cz - sx * sy * sz,
        ]

    @staticmethod
    def npquat_xyzw_to_gf(q):
        x, y, z, w = q
        return Gf.Quatd(float(w), Gf.Vec3d(float(x), float(y), float(z)))

    def set_xform_pose(self, prim, xyz, quat_xyzw):
        quatd = self.npquat_xyzw_to_gf(quat_xyzw)
        xformable = UsdGeom.Xformable(prim)
        translate_op = orient_op = None
        for op in xformable.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                translate_op = op
            elif op.GetOpType() == UsdGeom.XformOp.TypeOrient:
                orient_op = op
        if translate_op is None:
            translate_op = xformable.AddTranslateOp()
        if orient_op is None:
            orient_op = xformable.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble)
        translate_op.Set(Gf.Vec3d(float(xyz[0]), float(xyz[1]), float(xyz[2])))
        orient_op.Set(quatd)
        orient_op.Set(quatd)

    def compose_camera_world_pose(self, robot_pose_world: Tuple[float, float, float], cfg: dict):
        base_x, base_y, base_yaw = robot_pose_world
        dx, dy, dz = cfg["offset_xyz"]
        camera_yaw = base_yaw + math.radians(float(cfg.get("yaw_offset_deg", 0.0)))
        cam_x = base_x + math.cos(base_yaw) * dx - math.sin(base_yaw) * dy
        cam_y = base_y + math.sin(base_yaw) * dx + math.cos(base_yaw) * dy
        cam_z = dz
        q_base = self.yaw_to_quat_xyzw(camera_yaw)
        if "rot_quat_xyzw" in cfg:
            q_cam_local = cfg["rot_quat_xyzw"]
        else:
            r_deg, p_deg, y_deg = cfg["rot_xyz_deg"]
            q_cam_local = self.euler_xyz_deg_to_quat_xyzw(r_deg, p_deg, y_deg)
        q_cam_world = self.quat_multiply_xyzw(q_base, q_cam_local)
        return [cam_x, cam_y, cam_z], q_cam_world

    def sync_camera_poses_to_robot(self) -> dict:
        pose = self.get_pose()
        robot_pose = (pose["x"], pose["y"], pose["yaw"])
        for cfg in self.camera_configs.values():
            cam_xyz, cam_quat = self.compose_camera_world_pose(robot_pose, cfg)
            cam_prim = self.stage.GetPrimAtPath(cfg["camera_prim_path"])
            self.set_xform_pose(cam_prim, cam_xyz, cam_quat)
        return pose

    @staticmethod
    def capture_camera_rgb(camera, max_wait_frames: int = CAMERA_MAX_WAIT_FRAMES) -> tuple[np.ndarray, float]:
        rgba = None
        capture_timestamp = time.time()
        for _ in range(max_wait_frames):
            rgba = camera.get_rgba()
            capture_timestamp = time.time()
            rgb_probe = np.asarray(rgba)
            if (
                rgb_probe.size > 0
                and rgb_probe.ndim >= 3
                and float(np.mean(rgb_probe[..., :3])) >= CAMERA_MIN_VALID_MEAN
            ):
                break
            simulation_app.update()
        rgb_probe = np.asarray(rgba)
        if rgb_probe.size == 0 or rgb_probe.ndim < 3:
            raise RuntimeError(
                f"camera returned empty image after {max_wait_frames} frames"
            )
        if float(np.mean(rgb_probe[..., :3])) < CAMERA_MIN_VALID_MEAN:
            raise RuntimeError(
                f"camera returned dark image after {max_wait_frames} frames"
            )
        rgb = np.asarray(rgba)[..., :3]
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        return rgb, capture_timestamp

    @staticmethod
    def encode_rgb_jpeg(rgb: np.ndarray) -> str:
        img = Image.fromarray(rgb)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    @classmethod
    def encode_camera_jpeg(cls, camera, max_wait_frames: int = CAMERA_MAX_WAIT_FRAMES) -> tuple[str, float]:
        rgb, capture_timestamp = cls.capture_camera_rgb(camera, max_wait_frames)
        return cls.encode_rgb_jpeg(rgb), capture_timestamp

    def get_obs(self) -> dict:
        for _ in range(CAMERA_SETTLE_FRAMES):
            self.sync_camera_poses_to_robot()
            simulation_app.update()

        pose = self.sync_camera_poses_to_robot()
        simulation_app.update()
        obs_timestamp = time.time()
        images_b64 = {}
        image_capture_timestamps = {}
        rgb_by_view = {}
        for view_name, camera in self.cameras.items():
            rgb, capture_timestamp = self.capture_camera_rgb(camera)
            rgb_by_view[view_name] = rgb
            image_b64 = self.encode_rgb_jpeg(rgb)
            images_b64[view_name] = image_b64
            image_capture_timestamps[view_name] = capture_timestamp
        response_camera_mode = self.camera_mode
        if self.camera_layout == GEMINI336L_HIGH_DUAL_VIEW_CONCAT_LAYOUT:
            left_rgb = rgb_by_view["left_view"]
            right_rgb = rgb_by_view["right_view"]
            if left_rgb.shape[0] != right_rgb.shape[0]:
                right_img = Image.fromarray(right_rgb)
                new_width = round(right_img.width * (left_rgb.shape[0] / right_img.height))
                right_rgb = np.asarray(
                    right_img.resize((new_width, left_rgb.shape[0]), Image.Resampling.LANCZOS)
                )
            ego_rgb = np.concatenate([left_rgb, right_rgb], axis=1)
            images_b64 = {"ego_view": self.encode_rgb_jpeg(ego_rgb)}
            image_capture_timestamps = {
                "ego_view": max(
                    image_capture_timestamps["left_view"],
                    image_capture_timestamps["right_view"],
                )
            }
            response_camera_mode = "single"
        primary_view = "ego_view" if "ego_view" in images_b64 else next(iter(images_b64))
        obs = {
            "image_b64": images_b64[primary_view],
            "image_capture_timestamp": image_capture_timestamps[primary_view],
            "camera_mode": response_camera_mode,
            "camera_preset": self.camera_preset,
            "camera_layout": self.camera_layout,
            "views": list(images_b64.keys()),
            "pose": pose,
            "timestamp": obs_timestamp,
            "image_capture_timestamps": image_capture_timestamps,
        }
        if response_camera_mode == "multiview":
            obs["images_b64"] = images_b64
        return obs

    def update_dynamic_obstacle(self, dt: float):
        if self.obs_translate_op is None:
            return
        self.obs_phase += self.obs_speed * dt
        x = self.obs_center_x + self.obs_amplitude * math.sin(self.obs_phase)
        self.obs_translate_op.Set(Gf.Vec3d(x, self.obs_base_y, self.obs_base_z))

    def get_pose(self):
        x, y, yaw = get_world_xy_yaw(self.stage, ROBOT_BODY_PRIM_PATH)
        return {"x": x, "y": y, "yaw": yaw}

    def reset_robot_pose(self, x: float, y: float, yaw: float, label: str = "requested"):
        quat_wxyz = quat_wxyz_from_yaw(yaw)
        try:
            self.robot.set_world_pose(
                position=np.array([x, y, DEFAULT_Z], dtype=np.float32),
                orientation=quat_wxyz,
            )
            self.robot.set_linear_velocity(np.zeros(3, dtype=np.float32))
            self.robot.set_angular_velocity(np.zeros(3, dtype=np.float32))
        except Exception as exc:
            print(f"[SIM SERVER] articulation reset failed, using xform fallback: {exc}")
            quat_xyzw = [0.0, 0.0, math.sin(yaw * 0.5), math.cos(yaw * 0.5)]
            robot_prim = self.stage.GetPrimAtPath(ROBOT_ROOT_PRIM_PATH)
            self.set_xform_pose(robot_prim, [x, y, DEFAULT_Z], quat_xyzw)
        for _ in range(120):
            simulation_app.update()
        pose = self.get_pose()
        print(
            f"[SIM SERVER] reset pose {label}=({x:.2f},{y:.2f},{yaw:.2f}) "
            f"actual=({pose['x']:.2f},{pose['y']:.2f},{pose['yaw']:.2f})"
        )
        return pose

    def reset_robot_random_pose(self):
        x, y, yaw = sample_conditioned_spawn(self.spawn_region, self.occ_map)
        return self.reset_robot_pose(x, y, yaw, label="random")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--camera-mode",
        default="single",
        choices=["single", "multiview"],
        help="Return either one RGB image or ego/left/right RGB images.",
    )
    parser.add_argument(
        "--camera-preset",
        default=DEFAULT_CAMERA_PRESET,
        choices=sorted(CAMERA_PRESETS.keys()),
        help="Camera preset used for Isaac Sim RGB observations.",
    )
    parser.add_argument(
        "--camera-layout",
        default=DEFAULT_CAMERA_LAYOUT,
        choices=[
            DEFAULT_CAMERA_LAYOUT,
            "gemini336l_driveway_view",
            GEMINI336L_HIGH_DUAL_VIEW_LAYOUT,
            GEMINI336L_HIGH_DUAL_VIEW_CONCAT_LAYOUT,
        ],
        help=(
            "Physical camera mount layout. Driveway view requires multiview gemini_336l; "
            "high-dual view requires multiview gemini_336l or gemini_336l_portrait. "
            "Use the concat layout to return left/right high-dual views as one ego_view image."
        ),
    )
    parser.add_argument(
        "--camera-yaw-deg",
        type=float,
        default=0.0,
        help="Yaw offset for the single RGB camera. Front view is 0.",
    )
    parser.add_argument(
        "--multiview-yaw-step-deg",
        type=float,
        default=None,
        help="Override the preset yaw step for left/right multiview cameras.",
    )
    parser.add_argument(
        "--enable-ros2-bridge",
        action="store_true",
        help="Enable ROS2 bridge and keep ROS graphs from the robot USD active.",
    )
    args = parser.parse_args()

    server = JsonSocketServer(host="0.0.0.0", port=8765)
    sim = IsaacSimServer(
        camera_mode=args.camera_mode,
        camera_preset=args.camera_preset,
        camera_layout=args.camera_layout,
        camera_yaw_deg=args.camera_yaw_deg,
        multiview_yaw_step_deg=args.multiview_yaw_step_deg,
        enable_ros2_bridge=args.enable_ros2_bridge,
    )
    sim.setup()

    print("[SIM OBS SERVER] ready, listening on 0.0.0.0:8765")
    last_update_time = time.perf_counter()

    try:
        while simulation_app.is_running():
            try:
                now = time.perf_counter()
                dt = min(now - last_update_time, 0.25)
                last_update_time = now
                sim._disable_replicator_capture()
                sim.update_dynamic_obstacles(dt)
                sim.update_dynamic_obstacle_animation(dt)
                simulation_app.update()

                msg = server.recv_message()
                if msg is None:
                    continue

                cmd = msg.get("cmd")

                if cmd == "ping":
                    server.send_message({"ok": True, "msg": "pong"})

                elif cmd == "reset":
                    pose = sim.reset_robot_random_pose()
                    ok = server.send_message({"ok": True, "pose": pose})
                    if not ok:
                        print("[SIM SERVER] send failed for reset response")

                elif cmd == "reset_to_pose":
                    pose = sim.reset_robot_pose(
                        float(msg["x"]),
                        float(msg["y"]),
                        float(msg["yaw"]),
                        label=str(msg.get("label", "fixed")),
                    )
                    ok = server.send_message({"ok": True, "pose": pose})
                    if not ok:
                        print("[SIM SERVER] send failed for fixed reset response")

                elif cmd == "get_obs":
                    obs = sim.get_obs()
                    server.send_message({"ok": True, **obs})

                else:
                    server.send_message({"ok": False, "error": f"unknown cmd: {cmd}"})

            except Exception as e:
                print(f"[SIM SERVER] loop error: {e}")
                try:
                    server.send_message({"ok": False, "error": str(e)})
                except Exception:
                    pass

    finally:
        if sim.timeline is not None:
            sim.timeline.stop()
        server.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
