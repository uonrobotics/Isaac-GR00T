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
        "renderer": "RaytracedLighting",
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
from typing import Optional, Tuple

import numpy as np
from PIL import Image
import carb
import omni.timeline
import omni.usd
import yaml

from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdLux, UsdPhysics
from isaacsim.core.utils.stage import add_reference_to_stage, is_stage_loading
from isaacsim.storage.native import get_assets_root_path
from isaacsim.sensors.camera import Camera
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.utils.extensions import enable_extension


def set_active_viewport_renderer(engine_name: str, render_mode: str, quiet: bool = False) -> bool:
    try:
        import omni.kit.viewport.utility as viewport_utility

        viewport = viewport_utility.get_active_viewport()
        if viewport is not None:
            viewport.set_hd_engine(engine_name, render_mode)
            actual_engine = str(getattr(viewport, "hydra_engine", ""))
            actual_mode = str(getattr(viewport, "render_mode", ""))
            if not quiet:
                print(
                    "[SIM SERVER] active viewport renderer set "
                    f"(engine={actual_engine}, mode={actual_mode})"
                )
            return actual_engine == engine_name and actual_mode == render_mode
    except Exception as exc:
        if not quiet:
            print(f"[SIM SERVER] active viewport renderer direct set failed: {exc}")
    return False


def set_rtx_render_mode(render_mode: str, spp: int | None = None) -> str | None:
    settings = carb.settings.get_settings()
    previous_mode = settings.get("/rtx/rendermode")
    settings.set("/rtx/rendermode", render_mode)
    if render_mode == "PathTracing" and spp is not None:
        settings.set("/rtx/pathtracing/spp", int(spp))
        settings.set("/rtx/pathtracing/totalSpp", int(spp))
        settings.set("/rtx/pathtracing/denoising/enabled", True)
        settings.set("/rtx/pathtracing/optixDenoiser/enabled", True)
        settings.set("/rtx/pathtracing/optixDenoiser/blendFactor", 0.0)
        settings.set("/rtx/post/aa/op", 3)
    elif render_mode == "RaytracedLighting":
        settings.set("/rtx/post/aa/op", 3)
    return previous_mode


def restore_rtx_realtime_viewport() -> None:
    set_rtx_render_mode("RaytracedLighting")
    set_active_viewport_renderer("rtx", "RaytracedLighting", quiet=True)


ENV_USD_PATH = "/nas/sujinkim/data/SignNav/_assets/hospital/hospital_with_signs_and_waypoints.usd"
MAP_PNG_PATH = "/nas/sujinkim/data/SignNav/_assets/hospital/hospital_occupancy_map.png"
MAP_YAML_PATH = ""
WAYPOINTS_NPY_PATH = "/nas/sujinkim/data/SignNav/_assets/hospital/hospital_waypoint_graphs/waypoints.npy"
ACTION_TRAJECTORY_ROOT = "/nas/sujinkim/data/SignNav/sim_v1/action"
ROBOT_REL_PATH = "/Isaac/Samples/ROS2/Robots/Nova_Carter_ROS.usd"

ROBOT_ROOT_PRIM_PATH = "/World/Nova_Carter_ROS"
ROBOT_BODY_PRIM_PATH = "/World/Nova_Carter_ROS/chassis_link"
SPAWN_AREA_PRIM_PATH = "/World/NavMeshFloorPreview"
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
CAMERA_SETTLE_FRAMES = 0
CAMERA_MAX_WAIT_FRAMES = 8
CAMERA_MIN_VALID_MEAN = 1.0

DEFAULT_CAMERA_PRESET = "gemini_336l"
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
        "offset_xyz": [0.12, 0.0, 0.85],
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


def try_get_world_bbox_xy(stage, prim_path: str):
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return None
    bbox_cache = UsdGeom.BBoxCache(0, ["default"])
    bound = bbox_cache.ComputeWorldBound(prim)
    box = bound.ComputeAlignedBox()
    min_pt = box.GetMin()
    max_pt = box.GetMax()
    return float(min_pt[0]), float(max_pt[0]), float(min_pt[1]), float(max_pt[1])


def quat_wxyz_from_yaw(yaw_rad: float):
    half = yaw_rad * 0.5
    return np.array([math.cos(half), 0.0, 0.0, math.sin(half)], dtype=np.float32)


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
    if region is None:
        raise RuntimeError("No spawn region is available.")
    for _ in range(max_trials):
        x, y = sample_xy_in_region(region, margin=0.3)
        if occ_map is None:
            yaw = np.random.uniform(-math.pi, math.pi)
            return x, y, float(yaw)
        if not occ_map.is_inside(x, y):
            continue
        if not occ_map.is_free(x, y):
            continue
        if occ_map.clearance(x, y) < min_clearance:
            continue
        yaw = np.random.uniform(-math.pi, math.pi)
        return x, y, float(yaw)
    raise RuntimeError("Failed to sample valid spawn pose.")


def load_spawn_waypoints(path: str) -> np.ndarray | None:
    if not path or not os.path.exists(path):
        print(f"[SIM SERVER] spawn waypoints unavailable: {path}")
        return None
    waypoints = np.load(path)
    if waypoints.ndim != 2 or waypoints.shape[1] < 2 or waypoints.shape[0] == 0:
        raise ValueError(f"invalid spawn waypoints shape: {waypoints.shape}")
    waypoints = np.asarray(waypoints[:, :2], dtype=np.float32)
    print(f"[SIM SERVER] loaded {len(waypoints)} spawn waypoints: {path}")
    return waypoints


def extract_first_trajectory_sim_pose(json_path: str) -> dict | None:
    """Extract trajectory[0].sim_pose without loading the full trajectory JSON."""
    in_trajectory = False
    collecting = False
    pose_lines = []
    brace_balance = 0

    with open(json_path, "r", encoding="utf-8") as f:
        for line in f:
            if not in_trajectory:
                if '"trajectory"' in line:
                    in_trajectory = True
                continue

            if not collecting:
                if '"sim_pose"' not in line:
                    continue
                if "null" in line:
                    return None
                start = line.find("{")
                if start < 0:
                    continue
                fragment = line[start:]
                pose_lines = [fragment]
                brace_balance = fragment.count("{") - fragment.count("}")
                collecting = True
                if brace_balance <= 0:
                    break
                continue

            pose_lines.append(line)
            brace_balance += line.count("{") - line.count("}")
            if brace_balance <= 0:
                break

    if not pose_lines:
        return None
    text = "".join(pose_lines)
    end = text.rfind("}")
    if end >= 0:
        text = text[: end + 1]
    try:
        pose = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not all(k in pose for k in ("x", "y", "yaw")):
        return None
    return {"x": float(pose["x"]), "y": float(pose["y"]), "yaw": float(pose["yaw"])}


def load_trajectory_spawn_poses(root: str) -> list[dict]:
    if not root or not os.path.isdir(root):
        print(f"[SIM SERVER] trajectory spawn root unavailable: {root}")
        return []

    spawns = []
    for goal_name in sorted(os.listdir(root)):
        goal_dir = os.path.join(root, goal_name)
        if not os.path.isdir(goal_dir):
            continue
        for file_name in sorted(os.listdir(goal_dir)):
            if not file_name.endswith(".json"):
                continue
            path = os.path.join(goal_dir, file_name)
            try:
                pose = extract_first_trajectory_sim_pose(path)
            except Exception as exc:
                print(f"[SIM SERVER] failed to read spawn pose: {path} ({exc})")
                continue
            if pose is None:
                print(f"[SIM SERVER] no trajectory sim_pose in: {path}")
                continue
            episode_id = os.path.splitext(file_name)[0]
            spawns.append(
                {
                    "goal_name": goal_name,
                    "episode_id": episode_id,
                    "path": path,
                    **pose,
                }
            )

    print(f"[SIM SERVER] loaded {len(spawns)} trajectory spawn poses from {root}")
    return spawns


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
        viewport_renderer: str = "rtx",
        camera_renderer: str = "pathtracing",
        camera_pathtracing_spp: int = 16,
    ):
        self.stage = None
        self.timeline = None
        self.robot = None
        self.camera = None
        self.cameras = {}
        self.enable_ros2_bridge = enable_ros2_bridge
        self.viewport_renderer = viewport_renderer
        self.camera_renderer = camera_renderer
        self.camera_pathtracing_spp = camera_pathtracing_spp
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
        self.trajectory_spawns = load_trajectory_spawn_poses(ACTION_TRAJECTORY_ROOT)
        self.spawn_waypoints = load_spawn_waypoints(WAYPOINTS_NPY_PATH)
        self.occ_map = None
        if MAP_PNG_PATH and MAP_YAML_PATH and os.path.exists(MAP_PNG_PATH) and os.path.exists(MAP_YAML_PATH):
            self.occ_map = OccupancyDistanceMap(
                MAP_PNG_PATH,
                MAP_YAML_PATH,
            )
        else:
            print("[SIM SERVER] occupancy yaml unavailable; random reset uses spawn region bbox only")

    def setup(self):
        self._enable_extensions()
        self._disable_replicator_capture()
        self._create_stage_once()
        self._load_env_and_robot_once()
        self._disable_replicator_capture()
        self._start_simulation_once()
        self._disable_replicator_capture()
        self._initialize_articulation_once()
        self._setup_camera_once()
        self._configure_interactive_viewport()
        self._disable_replicator_capture()

    def _disable_replicator_capture(self):
        return

    def _enable_extensions(self):
        enable_extension("omni.physx")
        enable_extension("omni.physx.ui")
        enable_extension("omni.graph.nodes")
        enable_extension("isaacsim.core.nodes")
        enable_extension("omni.kit.viewport.actions")
        if self.enable_ros2_bridge:
            enable_extension("isaacsim.ros2.bridge")
        enable_extension("isaacsim.sensors.rtx")
        simulation_app.update()
        simulation_app.update()

    def _configure_interactive_viewport(self, quiet: bool = False) -> None:
        if self.camera_renderer == "pathtracing":
            set_rtx_render_mode("PathTracing", spp=self.camera_pathtracing_spp)
        elif self.viewport_renderer == "rtx":
            set_rtx_render_mode("RaytracedLighting")
            set_active_viewport_renderer("rtx", "RaytracedLighting", quiet=quiet)

    def _create_stage_once(self):
        if not os.path.exists(ENV_USD_PATH):
            raise FileNotFoundError(f"Environment USD not found: {ENV_USD_PATH}")
        omni.usd.get_context().open_stage(ENV_USD_PATH)
        for _ in range(240):
            simulation_app.update()
            if not is_stage_loading():
                break
        self.stage = omni.usd.get_context().get_stage()
        UsdGeom.SetStageMetersPerUnit(self.stage, 1.0)
        add_physics_scene(self.stage)
        add_dome_light(self.stage)
        print(f"[SIM SERVER] opened env stage: {ENV_USD_PATH}")

    def _load_env_and_robot_once(self):
        assets_root = get_assets_root_path()
        robot_usd = assets_root + ROBOT_REL_PATH
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
        self.spawn_region = try_get_world_bbox_xy(self.stage, SPAWN_AREA_PRIM_PATH)
        if self.spawn_region is None:
            print(
                f"[SIM SERVER] spawn region prim not found: {SPAWN_AREA_PRIM_PATH}; "
                "random reset will use hospital waypoint graph"
            )
        else:
            print(f"[SIM SERVER] spawn_region={self.spawn_region}")

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

    @staticmethod
    def matrix3_to_quat_xyzw(m):
        trace = m[0][0] + m[1][1] + m[2][2]
        if trace > 0.0:
            s = math.sqrt(trace + 1.0) * 2.0
            w = 0.25 * s
            x = (m[2][1] - m[1][2]) / s
            y = (m[0][2] - m[2][0]) / s
            z = (m[1][0] - m[0][1]) / s
        elif m[0][0] > m[1][1] and m[0][0] > m[2][2]:
            s = math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2.0
            w = (m[2][1] - m[1][2]) / s
            x = 0.25 * s
            y = (m[0][1] + m[1][0]) / s
            z = (m[0][2] + m[2][0]) / s
        elif m[1][1] > m[2][2]:
            s = math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2.0
            w = (m[0][2] - m[2][0]) / s
            x = (m[0][1] + m[1][0]) / s
            y = 0.25 * s
            z = (m[1][2] + m[2][1]) / s
        else:
            s = math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2.0
            w = (m[1][0] - m[0][1]) / s
            x = (m[0][2] + m[2][0]) / s
            y = (m[1][2] + m[2][1]) / s
            z = 0.25 * s
        return [x, y, z, w]

    @classmethod
    def look_at_quat_xyzw(cls, eye, target):
        eye_v = np.array(eye, dtype=np.float64)
        target_v = np.array(target, dtype=np.float64)
        forward = target_v - eye_v
        forward_norm = np.linalg.norm(forward)
        if forward_norm < 1e-6:
            return cls.euler_xyz_deg_to_quat_xyzw(90.0, -90.0, 0.0)
        forward /= forward_norm
        world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        right = np.cross(forward, world_up)
        if np.linalg.norm(right) < 1e-6:
            world_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
            right = np.cross(forward, world_up)
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        backward = -forward
        matrix = [
            [right[0], up[0], backward[0]],
            [right[1], up[1], backward[1]],
            [right[2], up[2], backward[2]],
        ]
        return cls.matrix3_to_quat_xyzw(matrix)

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
        img.save(buf, format="JPEG", quality=95, optimize=True)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    @classmethod
    def encode_camera_jpeg(cls, camera, max_wait_frames: int = CAMERA_MAX_WAIT_FRAMES) -> tuple[str, float]:
        rgb, capture_timestamp = cls.capture_camera_rgb(camera, max_wait_frames)
        return cls.encode_rgb_jpeg(rgb), capture_timestamp

    def _prepare_camera_rendering(self) -> None:
        if self.camera_renderer == "pathtracing":
            set_rtx_render_mode("PathTracing", spp=self.camera_pathtracing_spp)

    def get_obs(self) -> dict:
        for _ in range(CAMERA_SETTLE_FRAMES):
            self.sync_camera_poses_to_robot()
            simulation_app.update()

        pose = self.sync_camera_poses_to_robot()
        obs_timestamp = time.time()
        images_b64 = {}
        image_capture_timestamps = {}
        rgb_by_view = {}
        self._prepare_camera_rendering()
        simulation_app.update()
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
        if self.trajectory_spawns:
            idx = int(np.random.randint(0, len(self.trajectory_spawns)))
            spawn = self.trajectory_spawns[idx]
            x = float(spawn["x"])
            y = float(spawn["y"])
            yaw = float(spawn["yaw"])
            print("\n" + "=" * 88)
            print("  SIGNNAV RANDOM RESPAWN FROM RECORDED TRAJECTORY")
            print(
                f"  source: {spawn['goal_name']}/{spawn['episode_id']} "
                f"({spawn['path']})"
            )
            print(f"  pose:   x={x:.3f} y={y:.3f} yaw={yaw:.3f} rad")
            print("=" * 88 + "\n")
        elif self.spawn_waypoints is not None and len(self.spawn_waypoints) > 0:
            idx = int(np.random.randint(0, len(self.spawn_waypoints)))
            x, y = [float(v) for v in self.spawn_waypoints[idx]]
            yaw = float(np.random.uniform(-math.pi, math.pi))
            print(f"[SIM SERVER] waypoint random spawn idx={idx} x={x:.2f} y={y:.2f} yaw={yaw:.2f}")
        else:
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
    parser.add_argument(
        "--viewport-renderer",
        choices=("storm", "rtx", "unchanged"),
        default="rtx",
        help=(
            "Renderer for the interactive Isaac Sim viewport. "
            "Camera sensor rendering for GR00T keeps the SimulationApp renderer."
        ),
    )
    parser.add_argument(
        "--camera-renderer",
        choices=("pathtracing", "realtime"),
        default="realtime",
        help="RTX render mode used for ego camera captures.",
    )
    parser.add_argument(
        "--camera-pathtracing-spp",
        type=int,
        default=16,
        help="PathTracing samples per pixel for camera captures.",
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
        viewport_renderer=args.viewport_renderer,
        camera_renderer=args.camera_renderer,
        camera_pathtracing_spp=args.camera_pathtracing_spp,
    )
    sim.setup()

    print("[SIM OBS SERVER] ready, listening on 0.0.0.0:8765")
    last_update_time = time.perf_counter()
    viewport_renderer_retry_frames = (
        0 if args.viewport_renderer == "unchanged" or args.camera_renderer == "pathtracing" else 120
    )

    try:
        while simulation_app.is_running():
            try:
                now = time.perf_counter()
                dt = min(now - last_update_time, 0.25)
                last_update_time = now
                sim._disable_replicator_capture()
                simulation_app.update()
                if viewport_renderer_retry_frames > 0:
                    sim._configure_interactive_viewport(quiet=True)
                    viewport_renderer_retry_frames -= 1

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
