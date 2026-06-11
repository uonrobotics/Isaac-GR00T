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
from nav_msgs.msg import Odometry

AMCL_POSE_PORT = 8767
CMD_VEL_PORT   = 8766
CMD_VEL_TIMEOUT_SEC = 0.5
CMD_VEL_REPEAT_PERIOD_SEC = 0.02
ODOM_TOPIC = "/chassis/odom"
MOTION_LINEAR_THRESHOLD = 0.02
MOTION_ANGULAR_THRESHOLD = 0.03
MOTION_DEBUG_CMD_THRESHOLD = 0.01


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
        self._last_cmd_time = 0.0
        self._last_cmd_publish_time = 0.0
        self._cmd_linear = 0.0
        self._cmd_angular = 0.0
        self._has_cmd = False
        self._cmd_seq = None
        self._cmd_client_send_time = None
        self._cmd_bridge_recv_time = None
        self._cmd_first_publish_time = None
        self._pending_motion_cmd = None
        self._init_pub = self.create_publisher(PoseWithCovarianceStamped, "/initialpose", 10)
        self._amcl_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            "/amcl_pose",
            self._on_amcl_pose,
            10,
        )
        self._odom_sub = self.create_subscription(
            Odometry,
            ODOM_TOPIC,
            self._on_odom,
            20,
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

    def _on_odom(self, msg: Odometry):
        lin = abs(float(msg.twist.twist.linear.x))
        ang = abs(float(msg.twist.twist.angular.z))
        if self._pending_motion_cmd is None:
            return
        if lin < MOTION_LINEAR_THRESHOLD and ang < MOTION_ANGULAR_THRESHOLD:
            return
        now = time.monotonic()
        cmd = self._pending_motion_cmd
        recv_to_pub_ms = (cmd["first_publish_time"] - cmd["bridge_recv_time"]) * 1000.0
        pub_to_odom_ms = (now - cmd["first_publish_time"]) * 1000.0
        recv_to_odom_ms = (now - cmd["bridge_recv_time"]) * 1000.0
        if cmd.get("client_send_time") is not None:
            client_to_recv_ms = (cmd["bridge_recv_wall_time"] - cmd["client_send_time"]) * 1000.0
            client_to_odom_ms = (time.time() - cmd["client_send_time"]) * 1000.0
            client_text = (
                f"client->bridge={client_to_recv_ms:.1f}ms "
                f"client->odom={client_to_odom_ms:.1f}ms "
            )
        else:
            client_text = ""
        print(
            f"[CMD LATENCY] seq={cmd.get('seq')} "
            f"{client_text}"
            f"bridge_recv->publish={recv_to_pub_ms:.1f}ms "
            f"publish->odom={pub_to_odom_ms:.1f}ms "
            f"bridge_recv->odom={recv_to_odom_ms:.1f}ms "
            f"odom_v={lin:.3f} odom_w={ang:.3f}"
        )
        self._pending_motion_cmd = None

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

    def _publish_cmd_vel_now(self, linear: float, angular: float):
        msg = Twist()
        msg.linear.x = float(linear)
        msg.linear.y = 0.0
        msg.linear.z = 0.0
        msg.angular.x = 0.0
        msg.angular.y = 0.0
        msg.angular.z = float(angular)
        self._cmd_pub.publish(msg)
        self._last_cmd_publish_time = time.monotonic()

    def publish_cmd_vel(self, linear: float, angular: float, seq=None, client_send_time=None, bridge_recv_wall_time=None):
        self._cmd_linear = float(linear)
        self._cmd_angular = float(angular)
        self._last_cmd_time = time.monotonic()
        self._has_cmd = True
        self._cmd_seq = seq
        self._cmd_client_send_time = client_send_time
        self._cmd_bridge_recv_time = self._last_cmd_time
        self._publish_cmd_vel_now(self._cmd_linear, self._cmd_angular)
        self._cmd_first_publish_time = self._last_cmd_publish_time
        if abs(self._cmd_linear) > MOTION_DEBUG_CMD_THRESHOLD or abs(self._cmd_angular) > MOTION_DEBUG_CMD_THRESHOLD:
            self._pending_motion_cmd = {
                "seq": seq,
                "client_send_time": client_send_time,
                "bridge_recv_wall_time": bridge_recv_wall_time,
                "bridge_recv_time": self._cmd_bridge_recv_time,
                "first_publish_time": self._cmd_first_publish_time,
                "linear": self._cmd_linear,
                "angular": self._cmd_angular,
            }
            print(
                f"[CMD LATENCY] seq={seq} cmd received "
                f"v={self._cmd_linear:.3f} w={self._cmd_angular:.3f}; waiting for {ODOM_TOPIC}"
            )

    def repeat_latest_cmd_vel(self):
        if not self._has_cmd:
            return
        now = time.monotonic()
        if now - self._last_cmd_time > CMD_VEL_TIMEOUT_SEC:
            self._cmd_linear = 0.0
            self._cmd_angular = 0.0
        if now - self._last_cmd_publish_time >= CMD_VEL_REPEAT_PERIOD_SEC:
            self._publish_cmd_vel_now(self._cmd_linear, self._cmd_angular)

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
                conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                self.client = conn
                self.buffer = b""
                print(f"[{self.label}] client connected from {addr}")
            except OSError:
                pass

    def recv(self) -> Optional[dict]:
        self.poll_accept()
        if not self.client:
            return None
        msg = self._pop_buffered_message()
        if msg is not None:
            return msg
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
        return self._pop_buffered_message()

    def _pop_buffered_message(self) -> Optional[dict]:
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
                dropped_stale = 0
                while True:
                    newer_msg = cmd_srv.recv()
                    if newer_msg is None:
                        break
                    msg = newer_msg
                    dropped_stale += 1
                bridge_recv_wall_time = time.time()
                # Same cmd_vel protocol as AsyncVLA: {"linear": v, "angular": w}.
                linear = msg.get("linear", 0.0)
                angular = msg.get("angular", 0.0)
                seq = msg.get("seq")
                client_send_time = msg.get("client_send_time")
                node.publish_cmd_vel(
                    linear,
                    angular,
                    seq=seq,
                    client_send_time=client_send_time,
                    bridge_recv_wall_time=bridge_recv_wall_time,
                )
                if isinstance(client_send_time, (int, float)):
                    client_to_bridge_ms = (bridge_recv_wall_time - float(client_send_time)) * 1000.0
                    timing_text = f" client->bridge={client_to_bridge_ms:.1f}ms"
                else:
                    timing_text = ""
                stale_text = f" dropped_stale={dropped_stale}" if dropped_stale else ""
                print(f"[CMD BRIDGE] /cmd_vel: seq={seq} v={linear:.3f} w={angular:.3f}{timing_text}{stale_text}")
            node.repeat_latest_cmd_vel()

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
