"""
Combined ROS2 Bridge (Python 3.12 + ROS2 Jazzy)

두 가지 기능을 하나의 프로세스에서 처리:
  1. AMCL 포즈 서버: TF map→base_link 읽어서 TCP 8767 제공
  2. cmd_vel 브리지: TCP 8766 수신 → /cmd_vel 퍼블리시

실행:
    /usr/bin/python3.12 examples/PointNav/inference/ros2_amcl_cmdvel_bridge.py
"""

import json
import math
import select
import socket
import time
from typing import Optional

import rclpy
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener
from geometry_msgs.msg import Twist, PoseWithCovarianceStamped

AMCL_POSE_PORT = 8767
CMD_VEL_PORT   = 8766


def _quat_xyzw_to_yaw(x, y, z, w):
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


# ── ROS2 node ─────────────────────────────────────────────────────────────────

class BridgeNode(Node):
    def __init__(self):
        super().__init__("gr00t_ros2_bridge")
        # self.declare_parameter("use_sim_time", True)

        self.tf_buffer   = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self._pose: Optional[dict] = None
        self._amcl_pose: Optional[dict] = None
        self.create_timer(0.05, self._poll_tf)  # 20 Hz

        self._cmd_pub  = self.create_publisher(Twist, "/cmd_vel", 10)
        self._init_pub = self.create_publisher(PoseWithCovarianceStamped, "/initialpose", 10)
        self._amcl_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            "/amcl_pose",
            self._on_amcl_pose,
            10,
        )

    def _on_amcl_pose(self, msg: PoseWithCovarianceStamped):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self._amcl_pose = {
            "x": p.x,
            "y": p.y,
            "yaw": _quat_xyzw_to_yaw(q.x, q.y, q.z, q.w),
            "source": "amcl_pose",
        }

    def _poll_tf(self):
        import rclpy.time as _rt
        try:
            tf = self.tf_buffer.lookup_transform("map", "base_link", _rt.Time())
            t, q = tf.transform.translation, tf.transform.rotation
            self._pose = {
                "x":   t.x,
                "y":   t.y,
                "yaw": _quat_xyzw_to_yaw(q.x, q.y, q.z, q.w),
                "source": "tf",
            }
        except Exception:
            pass

    def get_pose(self) -> Optional[dict]:
        if self._amcl_pose:
            return dict(self._amcl_pose)
        return dict(self._pose) if self._pose else None

    def publish_cmd_vel(self, linear: float, angular: float):
        msg = Twist()
        msg.linear.x = float(linear)
        msg.linear.y = 0.0
        msg.linear.z = 0.0
        msg.angular.x = 0.0
        msg.angular.y = 0.0
        msg.angular.z = float(angular)
        self._cmd_pub.publish(msg)

    def publish_initial_pose(self, x, y, yaw, repeat=3, gap_sec=0.8, settle_sec=5.0):
        # ROS2 spin은 별도 스레드에서 계속 돌고 있으므로 여기서 sleep 해도 안전
        half = yaw * 0.5
        for i in range(repeat):
            msg = PoseWithCovarianceStamped()
            msg.header.frame_id = "map"
            msg.header.stamp    = self.get_clock().now().to_msg()
            msg.pose.pose.position.x    = float(x)
            msg.pose.pose.position.y    = float(y)
            msg.pose.pose.orientation.z = math.sin(half)
            msg.pose.pose.orientation.w = math.cos(half)
            msg.pose.covariance[0]  = 0.25
            msg.pose.covariance[7]  = 0.25
            msg.pose.covariance[35] = 0.0685
            self._init_pub.publish(msg)
            print(f"[ROS2 BRIDGE] /initialpose ({i+1}/{repeat}) "
                  f"x={x:.2f} y={y:.2f} yaw={yaw:.2f}")
            if i < repeat - 1:
                time.sleep(gap_sec)
        print(f"[ROS2 BRIDGE] settling {settle_sec:.1f}s ...")
        time.sleep(settle_sec)


# ── Non-blocking TCP server ────────────────────────────────────────────────────

class NonBlockingServer:
    def __init__(self, host: str, port: int, label: str):
        self.label = label
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind((host, port))
        self.server.listen(1)
        self.server.setblocking(False)
        self.client: Optional[socket.socket] = None
        self.buffer = b""

    def _drop(self, reason=""):
        if self.client:
            try:
                self.client.close()
            except Exception:
                pass
            self.client = None
            self.buffer = b""
            if reason:
                print(f"[{self.label}] client dropped: {reason}")

    def poll_accept(self):
        if self.client:
            return
        r, _, _ = select.select([self.server], [], [], 0.0)
        if r:
            try:
                conn, addr = self.server.accept()
                conn.setblocking(False)
                self.client = conn
                self.buffer = b""
                print(f"[{self.label}] client connected from {addr}")
            except OSError:
                pass

    def recv(self) -> Optional[dict]:
        self.poll_accept()
        if not self.client:
            return None
        try:
            r, _, ex = select.select([self.client], [], [self.client], 0.0)
        except (OSError, ValueError) as e:
            self._drop(f"select: {e}")
            return None
        if ex:
            self._drop("exception")
            return None
        if not r:
            return None
        try:
            while True:
                chunk = self.client.recv(65536)
                if not chunk:
                    self._drop("closed")
                    return None
                self.buffer += chunk
                r2, _, _ = select.select([self.client], [], [], 0.0)
                if not r2:
                    break
        except BlockingIOError:
            pass
        except (ConnectionResetError, BrokenPipeError, OSError) as e:
            self._drop(f"recv: {e}")
            return None
        if b"\n" not in self.buffer:
            return None
        line, self.buffer = self.buffer.split(b"\n", 1)
        line = line.strip()
        if not line:
            return None
        try:
            return json.loads(line.decode("utf-8"))
        except json.JSONDecodeError:
            return None

    def send(self, payload: dict) -> bool:
        if not self.client:
            return False
        try:
            self.client.sendall((json.dumps(payload) + "\n").encode("utf-8"))
            return True
        except (ConnectionResetError, BrokenPipeError, OSError) as e:
            self._drop(f"send: {e}")
            return False

    def close(self):
        self._drop()
        try:
            self.server.close()
        except Exception:
            pass


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    rclpy.init()
    node = BridgeNode()

    pose_srv = NonBlockingServer("0.0.0.0", AMCL_POSE_PORT, "POSE SRV")
    cmd_srv  = NonBlockingServer("0.0.0.0", CMD_VEL_PORT,   "CMD  SRV")

    print(f"[ROS2 BRIDGE] AMCL pose server : 0.0.0.0:{AMCL_POSE_PORT}")
    print(f"[ROS2 BRIDGE] cmd_vel bridge   : 0.0.0.0:{CMD_VEL_PORT}")

    try:
        while rclpy.ok():
            # ── AMCL pose requests ──────────────────────────────────────────
            msg = pose_srv.recv()
            if msg is not None:
                cmd = msg.get("cmd")
                if cmd == "get_pose":
                    pose = node.get_pose()
                    if pose:
                        pose_srv.send({"ok": True, "pose": pose})
                    else:
                        pose_srv.send({"ok": False, "error": "no TF yet"})
                elif cmd == "publish_initial_pose":
                    node.publish_initial_pose(
                        x          = float(msg["x"]),
                        y          = float(msg["y"]),
                        yaw        = float(msg["yaw"]),
                        repeat     = int(msg.get("repeat",     3)),
                        gap_sec    = float(msg.get("gap_sec",  0.8)),
                        settle_sec = float(msg.get("settle_sec", 5.0)),
                    )
                    pose_srv.send({"ok": True})
                elif cmd == "ping":
                    pose_srv.send({"ok": True})

            # ── cmd_vel commands ────────────────────────────────────────────
            msg = cmd_srv.recv()
            if msg is not None:
                # Same cmd_vel protocol as AsyncVLA: {"linear": v, "angular": w}.
                linear = msg.get("linear", 0.0)
                angular = msg.get("angular", 0.0)
                node.publish_cmd_vel(linear, angular)
                print(f"[CMD BRIDGE] /cmd_vel: v={linear:.3f} w={angular:.3f}")

            rclpy.spin_once(node, timeout_sec=0.005)

    except KeyboardInterrupt:
        print("\n[ROS2 BRIDGE] shutting down")
    finally:
        node.publish_cmd_vel(0.0, 0.0)
        rclpy.spin_once(node, timeout_sec=0.0)
        pose_srv.close()
        cmd_srv.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
