"""
ROS2 cmd_vel bridge for SignNav.

Receives newline-delimited JSON velocity commands on TCP 8766 and publishes
them to /cmd_vel. No localization, AMCL, TF, or initialpose handling is used.
"""

from __future__ import annotations

import json
import select
import socket
import time
from typing import Optional

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist


CMD_VEL_PORT = 8766
CMD_VEL_TIMEOUT_SEC = 0.5
CMD_VEL_REPEAT_PERIOD_SEC = 0.02


class CmdVelBridgeNode(Node):
    def __init__(self):
        super().__init__("gr00t_signnav_cmdvel_bridge")
        self._cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self._last_cmd_time = 0.0
        self._last_cmd_publish_time = 0.0
        self._cmd_linear = 0.0
        self._cmd_angular = 0.0
        self._has_cmd = False

    def _publish_cmd_vel_now(self, linear: float, angular: float):
        msg = Twist()
        msg.linear.x = float(linear)
        msg.angular.z = float(angular)
        self._cmd_pub.publish(msg)
        self._last_cmd_publish_time = time.monotonic()

    def publish_cmd_vel(self, linear: float, angular: float, seq=None, action_step=None):
        self._cmd_linear = float(linear)
        self._cmd_angular = float(angular)
        self._last_cmd_time = time.monotonic()
        self._has_cmd = True
        self._publish_cmd_vel_now(self._cmd_linear, self._cmd_angular)
        print(
            f"[CMD] seq={seq} action_step={action_step} "
            f"v={self._cmd_linear:+.3f} w={self._cmd_angular:+.3f}"
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

    def close(self):
        self._drop()
        try:
            self.server.close()
        except Exception:
            pass


def main():
    rclpy.init()
    node = CmdVelBridgeNode()
    cmd_srv = NonBlockingServer("0.0.0.0", CMD_VEL_PORT, "CMD SRV")
    print(f"[ROS2 BRIDGE] cmd_vel bridge: 0.0.0.0:{CMD_VEL_PORT}")

    try:
        while rclpy.ok():
            msg = cmd_srv.recv()
            if msg is not None:
                while True:
                    newer_msg = cmd_srv.recv()
                    if newer_msg is None:
                        break
                    msg = newer_msg
                node.publish_cmd_vel(
                    msg.get("linear", 0.0),
                    msg.get("angular", 0.0),
                    seq=msg.get("seq"),
                    action_step=msg.get("action_step"),
                )
            node.repeat_latest_cmd_vel()
            rclpy.spin_once(node, timeout_sec=0.005)
    except KeyboardInterrupt:
        print("\n[ROS2 BRIDGE] interrupted")
    finally:
        node.publish_cmd_vel(0.0, 0.0)
        cmd_srv.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
