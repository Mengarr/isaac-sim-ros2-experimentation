"""
pi0_inference_broker.py
------------------------
ROS2 node that runs locally alongside Isaac Sim and brokers between it and the
remote pi0_inference_server. Isaac Sim is unchanged — it keeps publishing camera
and joint state topics as before, and this node subscribes to them on the same
machine (no network cost). When the action queue runs dry, the broker JPEG-encodes
the latest frames, calls the remote /get_action_chunk service over the network
connection, and republishes the returned action chunk to /joint_command at the
local control rate.

Subscribes to (local, same machine as Isaac Sim):
  /wrist_camera/image_raw   (sensor_msgs/Image)
  /base_camera/image_raw    (sensor_msgs/Image)
  /joint_states              (sensor_msgs/JointState)

Calls (remote, over the network):
  /get_action_chunk          (pre_trained_vla_test_interfaces/srv/GetActionChunk)

Publishes to (local):
  /joint_command              (sensor_msgs/JointState)
"""

import collections
import sys
import threading
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image, JointState

from pre_trained_vla_test_interfaces.srv import GetActionChunk

_JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]
_NUM_JOINTS = len(_JOINT_NAMES)

_CONTROL_HZ = 30.0


def _ros_image_to_compressed(msg: Image, jpeg_quality: int) -> CompressedImage:
    """Re-encode a raw sensor_msgs/Image as a JPEG-compressed sensor_msgs/CompressedImage."""
    data = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
    bgr = data[:, :, :3]
    if msg.encoding.lower() != "bgr8":
        bgr = cv2.cvtColor(bgr, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    if not ok:
        raise RuntimeError("JPEG encoding failed")

    out = CompressedImage()
    out.header = msg.header
    out.format = "jpeg"
    out.data = encoded.tobytes()
    return out


class PI0InferenceBrokerNode(Node):
    def __init__(self):
        super().__init__("pi0_inference_broker")

        self.declare_parameter("jpeg_quality", 80)
        self._jpeg_quality = self.get_parameter("jpeg_quality").get_parameter_value().integer_value

        self._lock = threading.Lock()
        self._action_queue: collections.deque = collections.deque()
        self._chunk_done = threading.Event()
        self._chunk_done.set()  # Ready to request a chunk immediately on startup

        # Cached observations (None until first message received)
        self._obs_wrist: Image | None = None
        self._obs_base: Image | None = None
        self._joint_state: JointState | None = None

        # Subscribers (local — same machine as Isaac Sim)
        self.create_subscription(Image, "/wrist_camera/image_raw", self._cb_wrist, 1)
        self.create_subscription(Image, "/base_camera/image_raw", self._cb_base, 1)
        self.create_subscription(JointState, "/joint_states", self._cb_joints, 1)

        # Publisher (local)
        self._pub = self.create_publisher(JointState, "/joint_command", 10)

        # Service client (remote, over the network)
        self._client = self.create_client(GetActionChunk, "get_action_chunk")

        # Control timer runs on the executor thread (lightweight — just publishes)
        self.create_timer(1.0 / _CONTROL_HZ, self._control_step)

        self._paused = False

        self.get_logger().info("PI0InferenceBrokerNode ready.")
        self.get_logger().info("Commands: pause | resume | reset")

        # Service-request and stdin run on their own threads so they don't block the executor
        self._request_thread = threading.Thread(target=self._request_loop, daemon=True)
        self._request_thread.start()
        self._input_thread = threading.Thread(target=self._stdin_listener, daemon=True)
        self._input_thread.start()

    # ------------------------------------------------------------------
    # Subscriber callbacks — cache latest message
    # ------------------------------------------------------------------

    def _cb_wrist(self, msg: Image) -> None:
        with self._lock:
            self._obs_wrist = msg

    def _cb_base(self, msg: Image) -> None:
        with self._lock:
            self._obs_base = msg

    def _cb_joints(self, msg: JointState) -> None:
        with self._lock:
            self._joint_state = msg

    # ------------------------------------------------------------------
    # Terminal command listener
    # ------------------------------------------------------------------

    def _stdin_listener(self) -> None:
        for line in sys.stdin:
            cmd = line.strip().lower()
            if cmd in ("pause", "p"):
                self._paused = True
                self.get_logger().info("Paused.")
            elif cmd in ("resume", "r"):
                self._paused = False
                self.get_logger().info("Resumed.")
            elif cmd in ("reset", "rst"):
                with self._lock:
                    self._action_queue.clear()
                self._chunk_done.set()
                self._paused = True
                self.get_logger().info("Reset. Type 'resume' to start requesting chunks.")
            else:
                self.get_logger().warn(f"Unknown command: '{cmd}'. Use pause | resume | reset.")

    # ------------------------------------------------------------------
    # Request thread (~5 Hz, runs independently of the ROS executor)
    # ------------------------------------------------------------------

    def _request_loop(self) -> None:
        while rclpy.ok():
            self._chunk_done.wait()
            self._chunk_done.clear()
            if not self._paused:
                try:
                    self._request_step_impl()
                except Exception as e:
                    self.get_logger().error(
                        f"Request step failed: {type(e).__name__}: {e}", throttle_duration_sec=5.0
                    )
            # If the queue is still empty (missing obs, error, or paused), retry after a
            # short delay so the loop doesn't deadlock waiting for a control-step signal
            # that will never come.
            with self._lock:
                queue_empty = len(self._action_queue) == 0
            if queue_empty:
                time.sleep(0.2)
                self._chunk_done.set()

    def _request_step_impl(self) -> None:
        with self._lock:
            missing = [
                name for name, val in (
                    ("wrist_camera", self._obs_wrist),
                    ("base_camera", self._obs_base),
                    ("joint_states", self._joint_state),
                )
                if val is None
            ]
        if missing:
            self.get_logger().warn(
                f"Waiting for: {', '.join(missing)}", throttle_duration_sec=5.0
            )
            return

        if not self._client.wait_for_service(timeout_sec=5.0):
            self.get_logger().warn("get_action_chunk service unavailable.", throttle_duration_sec=5.0)
            return

        with self._lock:
            wrist_compressed = _ros_image_to_compressed(self._obs_wrist, self._jpeg_quality)
            base_compressed = _ros_image_to_compressed(self._obs_base, self._jpeg_quality)
            joint_state = self._joint_state

        request = GetActionChunk.Request()
        request.wrist_image = wrist_compressed
        request.base_image = base_compressed
        request.joint_state = joint_state

        self.get_logger().debug("Requesting action chunk ...")
        done = threading.Event()
        future = self._client.call_async(request)
        future.add_done_callback(lambda _f: done.set())
        if not done.wait(timeout=30.0):
            future.cancel()
            self.get_logger().warn("get_action_chunk request timed out.", throttle_duration_sec=5.0)
            return

        if future.exception() is not None:
            raise future.exception()
        response = future.result()
        if response is None or not response.action_chunk:
            self.get_logger().warn("Received empty action chunk.", throttle_duration_sec=5.0)
            return

        with self._lock:
            self._action_queue = collections.deque(response.action_chunk)

    # ------------------------------------------------------------------
    # Control timer (30 Hz)
    # ------------------------------------------------------------------

    def _control_step(self) -> None:
        if self._paused:
            return
        with self._lock:
            if not self._action_queue:
                return
            action = self._action_queue.popleft()  # sensor_msgs/JointState
            if not self._action_queue:
                self._chunk_done.set()  # Queue just emptied — trigger next request

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = _JOINT_NAMES
        msg.position = list(action.position)
        self._pub.publish(msg)


def main() -> None:
    rclpy.init()
    node = PI0InferenceBrokerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
