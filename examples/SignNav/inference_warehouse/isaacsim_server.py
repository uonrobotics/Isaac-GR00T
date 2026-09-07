from isaacsim import SimulationApp

simulation_app = SimulationApp(
    {
        "headless": False,
        "renderer": "RaytracedLighting",
        "fast_shutdown": True,
    }
)

import argparse
import base64
import io
import json
import math
import os
import select
import socket
import time
from typing import Optional

import carb
import numpy as np
import omni.timeline
import omni.usd
from PIL import Image
from isaacsim.core.utils.stage import add_reference_to_stage, is_stage_loading
from isaacsim.sensors.camera import Camera
from isaacsim.storage.native import get_assets_root_path
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.utils.extensions import enable_extension
from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdLux, UsdPhysics


ROBOT_REL_PATH = "/Isaac/Samples/ROS2/Robots/Nova_Carter_ROS.usd"
ROBOT_ROOT_PRIM_PATH = "/World/Nova_Carter_ROS"
ROBOT_BODY_PRIM_PATH = "/World/Nova_Carter_ROS/chassis_link"
ENV_SCALE_PRIM_PATH = "/World/EnvScale"
ENV_ROOT_PRIM_PATH = "/World/EnvScale/Env"
CAMERA_PRIM_PATH = "/World/replay_camera/front_camera"
CAMERA_ROS_GRAPH_PATHS = [
    "/World/Nova_Carter_ROS/front_hawk",
    "/World/Nova_Carter_ROS/right_hawk",
    "/World/Nova_Carter_ROS/left_hawk",
    "/World/Nova_Carter_ROS/back_hawk",
    "/World/Nova_Carter_ROS/front_owl",
    "/World/Nova_Carter_ROS/back_owl",
    "/World/Nova_Carter_ROS/left_owl",
    "/World/Nova_Carter_ROS/right_owl",
]

CAMERA_PRESETS = {
    "gemini_336l": {
        "resolution": (640, 400),
        "fov_deg": 94.0,
        "vertical_fov_deg": 68.0,
        "offset_xyz": [0.12, 0.0, 0.85],
        "clipping_range": (0.01, 1000.0),
        "horizontal_aperture": 20.955,
    },
    "gemini_336": {
        "resolution": (640, 360),
        "fov_deg": 86.0,
        "vertical_fov_deg": 55.0,
        "offset_xyz": [0.20, 0.0, 0.33],
        "clipping_range": (0.01, 1000.0),
        "horizontal_aperture": 20.955,
    },
    "gemini_345lg": {
        "resolution": (640, 360),
        "fov_deg": 137.0,
        "vertical_fov_deg": 71.0,
        "offset_xyz": [0.20, 0.0, 0.33],
        "clipping_range": (0.01, 1000.0),
        "horizontal_aperture": 20.955,
    },
}


def set_realtime_renderer() -> None:
    settings = carb.settings.get_settings()
    settings.set("/rtx/rendermode", "RaytracedLighting")
    settings.set("/rtx/post/aa/op", 3)
    try:
        import omni.kit.viewport.utility as viewport_utility

        viewport = viewport_utility.get_active_viewport()
        if viewport is not None:
            viewport.set_hd_engine("rtx", "RaytracedLighting")
    except Exception as exc:
        print(f"[ISAACSIM] viewport renderer setup skipped: {exc}")


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
    path = "/World/DomeLight"
    if stage.GetPrimAtPath(path).IsValid():
        return
    dome = UsdLux.DomeLight.Define(stage, path)
    dome.CreateIntensityAttr(1000)


def add_collision_to_prim(stage, prim_path: str, approximation: str = "none") -> int:
    if not prim_path:
        return 0
    root = stage.GetPrimAtPath(prim_path)
    if not root.IsValid():
        print(f"[ISAACSIM] collision prim not found: {prim_path}")
        return 0
    count = 0
    for prim in Usd.PrimRange(root):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI.Apply(prim)
        if hasattr(UsdPhysics, "MeshCollisionAPI"):
            mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(prim)
            mesh_collision.CreateApproximationAttr().Set(approximation)
        count += 1
        print(f"[ISAACSIM] collision mesh: {prim.GetPath()}")
    print(f"[ISAACSIM] collision applied: root={prim_path} meshes={count}")
    return count


def source_stage_meters_per_unit(usd_path: str) -> float:
    source_stage = Usd.Stage.Open(usd_path)
    if source_stage is None:
        raise RuntimeError(f"failed to open source USD: {usd_path}")
    return float(UsdGeom.GetStageMetersPerUnit(source_stage))


def add_scaled_env_reference(stage, usd_path: str, meters_per_unit: float) -> None:
    scale_prim = UsdGeom.Xform.Define(stage, ENV_SCALE_PRIM_PATH).GetPrim()
    scale_xform = UsdGeom.Xformable(scale_prim)
    if not scale_xform.GetOrderedXformOps():
        scale_xform.AddScaleOp(opSuffix="sourceMetersPerUnit").Set(
            Gf.Vec3f(meters_per_unit, meters_per_unit, meters_per_unit)
        )

    env_prim = UsdGeom.Xform.Define(stage, ENV_ROOT_PRIM_PATH).GetPrim()
    env_prim.GetReferences().AddReference(usd_path, "/World")


def env_referenced_path(original_path: str) -> str:
    if not original_path:
        return ""
    if original_path == "/World":
        return ENV_ROOT_PRIM_PATH
    if original_path.startswith("/World/"):
        return ENV_ROOT_PRIM_PATH + original_path[len("/World") :]
    return original_path


def quat_wxyz_from_yaw(yaw: float):
    half = yaw * 0.5
    return np.array([math.cos(half), 0.0, 0.0, math.sin(half)], dtype=np.float32)


def quat_xyzw_from_yaw(yaw: float):
    half = yaw * 0.5
    return [0.0, 0.0, math.sin(half), math.cos(half)]


def npquat_xyzw_to_gf(q):
    x, y, z, w = q
    return Gf.Quatd(float(w), Gf.Vec3d(float(x), float(y), float(z)))


def set_xform_pose(prim, xyz, quat_xyzw):
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
    orient_op.Set(npquat_xyzw_to_gf(quat_xyzw))


def get_world_xy_yaw(stage, prim_path: str):
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return 0.0, 0.0, 0.0
    mat = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0)
    pos = mat.ExtractTranslation()
    forward = mat.TransformDir(Gf.Vec3d(1.0, 0.0, 0.0))
    yaw = math.atan2(float(forward[1]), float(forward[0]))
    return float(pos[0]), float(pos[1]), float(yaw)


def fov_to_focal_length(fov_deg: float, aperture: float) -> float:
    return aperture / (2.0 * math.tan(math.radians(fov_deg) * 0.5))


def vertical_aperture_from_fov(vertical_fov_deg: float, focal_length: float) -> float:
    return 2.0 * focal_length * math.tan(math.radians(vertical_fov_deg) * 0.5)


def quat_multiply_xyzw(q1, q2):
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return [
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ]


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


class JsonSocketServer:
    def __init__(self, host: str, port: int):
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
                print(f"[ISAACSIM IPC] client dropped: {reason}")

    def accept_if_needed(self):
        if self.client is not None:
            return
        readable, _, _ = select.select([self.server], [], [], 0.0)
        if not readable:
            return
        conn, addr = self.server.accept()
        conn.setblocking(False)
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.client = conn
        self.buffer = b""
        print(f"[ISAACSIM IPC] client connected from {addr}")

    def recv_message(self):
        self.accept_if_needed()
        if self.client is None:
            return None
        if b"\n" not in self.buffer:
            try:
                readable, _, err = select.select([self.client], [], [self.client], 0.0)
            except (OSError, ValueError):
                self._drop_client("select failed")
                return None
            if err:
                self._drop_client("socket exception")
                return None
            if readable:
                try:
                    data = self.client.recv(1 << 20)
                except BlockingIOError:
                    data = b""
                except OSError as exc:
                    self._drop_client(f"recv failed: {exc}")
                    return None
                if not data:
                    self._drop_client("closed")
                    return None
                self.buffer += data
        if b"\n" not in self.buffer:
            return None
        line, self.buffer = self.buffer.split(b"\n", 1)
        try:
            return json.loads(line.decode("utf-8"))
        except json.JSONDecodeError:
            return None

    def send_message(self, payload: dict) -> bool:
        if self.client is None:
            return False
        try:
            self.client.sendall((json.dumps(payload) + "\n").encode("utf-8"))
            return True
        except OSError as exc:
            self._drop_client(f"send failed: {exc}")
            return False

    def close(self):
        self._drop_client()
        self.server.close()


class IsaacSimServer:
    def __init__(self, args):
        self.args = args
        self.stage = None
        self.timeline = None
        self.robot = None
        self.camera = None
        self.camera_cfg = CAMERA_PRESETS[args.camera_preset]

    def setup(self):
        enable_extension("omni.physx")
        enable_extension("omni.graph.nodes")
        enable_extension("isaacsim.core.nodes")
        if self.args.enable_ros2_bridge:
            enable_extension("isaacsim.ros2.bridge")
        enable_extension("isaacsim.sensors.rtx")
        for _ in range(2):
            simulation_app.update()

        if not os.path.exists(self.args.env_usd_path):
            raise FileNotFoundError(self.args.env_usd_path)
        env_meters_per_unit = source_stage_meters_per_unit(self.args.env_usd_path)
        omni.usd.get_context().new_stage()
        for _ in range(240):
            simulation_app.update()
            if not is_stage_loading():
                break
        self.stage = omni.usd.get_context().get_stage()
        UsdGeom.SetStageMetersPerUnit(self.stage, 1.0)
        if not self.stage.GetPrimAtPath("/World").IsValid():
            UsdGeom.Xform.Define(self.stage, "/World")
        add_scaled_env_reference(self.stage, self.args.env_usd_path, env_meters_per_unit)
        for _ in range(20):
            simulation_app.update()
        add_physics_scene(self.stage)
        add_dome_light(self.stage)
        add_collision_to_prim(
            self.stage,
            env_referenced_path(self.args.floor_collision_prim_path),
            self.args.floor_collision_approximation,
        )
        print(
            f"[ISAACSIM] referenced USD: {self.args.env_usd_path} "
            f"under {ENV_ROOT_PRIM_PATH} scale_parent={ENV_SCALE_PRIM_PATH} "
            f"scale={env_meters_per_unit}"
        )

        robot_usd = get_assets_root_path() + ROBOT_REL_PATH
        add_reference_to_stage(usd_path=robot_usd, prim_path=ROBOT_ROOT_PRIM_PATH)
        for _ in range(2):
            simulation_app.update()
        self._deactivate_robot_camera_ros_graphs()
        self._set_robot_xform_pose(self.args.spawn_x, self.args.spawn_y, self.args.spawn_z, self.args.spawn_yaw)

        self.timeline = omni.timeline.get_timeline_interface()
        self.timeline.play()
        for _ in range(20):
            simulation_app.update()

        self.robot = Articulation(ROBOT_ROOT_PRIM_PATH)
        self.robot.initialize()
        self.reset_robot_pose(self.args.spawn_x, self.args.spawn_y, self.args.spawn_yaw, self.args.spawn_z)
        self._setup_camera()
        set_realtime_renderer()
        print("[ISAACSIM] ready")

    def _deactivate_robot_camera_ros_graphs(self):
        if not self.args.enable_ros2_bridge:
            return
        disabled = []
        for prim_path in CAMERA_ROS_GRAPH_PATHS:
            prim = self.stage.GetPrimAtPath(prim_path)
            if prim.IsValid() and prim.IsActive():
                prim.SetActive(False)
                disabled.append(prim_path)
        if disabled:
            print(f"[ISAACSIM] deactivated built-in robot camera ROS graphs: {len(disabled)}")

    def _set_robot_xform_pose(self, x: float, y: float, z: float, yaw: float):
        robot_prim = self.stage.GetPrimAtPath(ROBOT_ROOT_PRIM_PATH)
        if robot_prim.IsValid():
            set_xform_pose(robot_prim, [x, y, z], quat_xyzw_from_yaw(yaw))

    def _setup_camera(self):
        if not self.stage.GetPrimAtPath("/World/replay_camera").IsValid():
            UsdGeom.Xform.Define(self.stage, "/World/replay_camera")
        if not self.stage.GetPrimAtPath(CAMERA_PRIM_PATH).IsValid():
            UsdGeom.Camera.Define(self.stage, CAMERA_PRIM_PATH)

        self.camera = Camera(
            prim_path=CAMERA_PRIM_PATH,
            name="front_camera",
            frequency=30,
            resolution=self.camera_cfg["resolution"],
        )
        self.camera.initialize()
        cam_geom = UsdGeom.Camera(self.stage.GetPrimAtPath(CAMERA_PRIM_PATH))
        focal = fov_to_focal_length(
            self.camera_cfg["fov_deg"],
            self.camera_cfg["horizontal_aperture"],
        )
        cam_geom.GetHorizontalApertureAttr().Set(self.camera_cfg["horizontal_aperture"])
        cam_geom.GetVerticalApertureAttr().Set(
            vertical_aperture_from_fov(self.camera_cfg["vertical_fov_deg"], focal)
        )
        cam_geom.GetFocalLengthAttr().Set(focal)
        cam_geom.GetClippingRangeAttr().Set(Gf.Vec2f(*self.camera_cfg["clipping_range"]))
        for _ in range(5):
            self.sync_camera_pose()
            simulation_app.update()

    def reset_robot_pose(self, x: float, y: float, yaw: float, z: float | None = None):
        z = self.args.spawn_z if z is None else float(z)
        if self.robot is None:
            self._set_robot_xform_pose(x, y, z, yaw)
        else:
            self.robot.set_world_pose(
                position=np.array([x, y, z], dtype=np.float32),
                orientation=quat_wxyz_from_yaw(yaw),
            )
            self.robot.set_linear_velocity(np.zeros(3, dtype=np.float32))
            self.robot.set_angular_velocity(np.zeros(3, dtype=np.float32))
        for _ in range(20):
            simulation_app.update()
        pose = self.get_pose()
        print(
            f"[ISAACSIM] reset pose requested=({x:.3f},{y:.3f},{z:.3f},{yaw:.3f}) "
            f"actual=({pose['x']:.3f},{pose['y']:.3f},{pose['yaw']:.3f})"
        )
        return pose

    def get_pose(self):
        x, y, yaw = get_world_xy_yaw(self.stage, ROBOT_BODY_PRIM_PATH)
        return {"x": x, "y": y, "yaw": yaw}

    def sync_camera_pose(self):
        pose = self.get_pose()
        base_x, base_y, base_yaw = pose["x"], pose["y"], pose["yaw"]
        dx, dy, dz = self.camera_cfg["offset_xyz"]
        cam_x = base_x + math.cos(base_yaw) * dx - math.sin(base_yaw) * dy
        cam_y = base_y + math.sin(base_yaw) * dx + math.cos(base_yaw) * dy
        cam_z = dz
        q_base = quat_xyzw_from_yaw(base_yaw)
        q_cam = euler_xyz_deg_to_quat_xyzw(90.0, -90.0, 0.0)
        set_xform_pose(
            self.stage.GetPrimAtPath(CAMERA_PRIM_PATH),
            [cam_x, cam_y, cam_z],
            quat_multiply_xyzw(q_base, q_cam),
        )
        return pose

    def get_obs(self):
        obs_timestamp = time.time()
        pose = self.sync_camera_pose()
        for _ in range(max(0, int(self.args.camera_settle_frames))):
            simulation_app.update()
        rgba = self.camera.get_rgba()
        if rgba is None:
            raise RuntimeError("camera returned no image")
        rgb = np.asarray(rgba[:, :, :3], dtype=np.uint8)
        image = Image.fromarray(rgb)
        buf = io.BytesIO()
        if self.args.image_format == "jpeg":
            image.save(buf, format="JPEG", quality=self.args.jpeg_quality, optimize=False)
        else:
            image.save(buf, format="PNG")
        image_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        capture_timestamp = time.time()
        return {
            "image_b64": image_b64,
            "images_b64": {"ego_view": image_b64},
            "image_capture_timestamp": capture_timestamp,
            "image_capture_timestamps": {"ego_view": capture_timestamp},
            "camera_mode": "single",
            "camera_preset": self.args.camera_preset,
            "camera_layout": "default",
            "views": ["ego_view"],
            "pose": pose,
            "timestamp": obs_timestamp,
        }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-usd-path", required=True)
    parser.add_argument("--spawn-x", type=float, default=0.0)
    parser.add_argument("--spawn-y", type=float, default=0.0)
    parser.add_argument("--spawn-z", type=float, default=0.0)
    parser.add_argument("--spawn-yaw", type=float, default=0.0)
    parser.add_argument("--camera-preset", choices=sorted(CAMERA_PRESETS), default="gemini_336l")
    parser.add_argument("--camera-settle-frames", type=int, default=0)
    parser.add_argument("--image-format", choices=["jpeg", "png"], default="jpeg")
    parser.add_argument("--jpeg-quality", type=int, default=85)
    parser.add_argument("--sim-port", type=int, default=8765)
    parser.add_argument("--enable-ros2-bridge", action="store_true")
    parser.add_argument("--floor-collision-prim-path", default="")
    parser.add_argument(
        "--floor-collision-approximation",
        choices=["none", "convexHull", "convexDecomposition", "meshSimplification"],
        default="none",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    sim = IsaacSimServer(args)
    server = JsonSocketServer("0.0.0.0", args.sim_port)
    sim.setup()
    print(f"[ISAACSIM OBS SERVER] listening on 0.0.0.0:{args.sim_port}")

    try:
        while simulation_app.is_running():
            simulation_app.update()
            msg = server.recv_message()
            if msg is None:
                continue
            cmd = msg.get("cmd")
            try:
                if cmd == "ping":
                    server.send_message({"ok": True, "msg": "pong"})
                elif cmd == "reset":
                    pose = sim.reset_robot_pose(args.spawn_x, args.spawn_y, args.spawn_yaw, args.spawn_z)
                    server.send_message({"ok": True, "pose": pose})
                elif cmd == "reset_to_pose":
                    pose = sim.reset_robot_pose(
                        float(msg["x"]),
                        float(msg["y"]),
                        float(msg["yaw"]),
                        float(msg.get("z", args.spawn_z)),
                    )
                    server.send_message({"ok": True, "pose": pose})
                elif cmd == "get_obs":
                    server.send_message({"ok": True, **sim.get_obs()})
                else:
                    server.send_message({"ok": False, "error": f"unknown cmd: {cmd}"})
            except Exception as exc:
                print(f"[ISAACSIM] command failed: {exc}")
                server.send_message({"ok": False, "error": str(exc)})
    except KeyboardInterrupt:
        print("\n[ISAACSIM] interrupted")
    finally:
        server.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
