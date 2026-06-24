"""
pi0_inference_broker.py
------------------------
ROS2 node that runs locally alongside Isaac Sim and brokers between it and the
remote pi0_inference_server. Isaac Sim is unchanged — it keeps publishing camera
and joint state topics as before, and this node subscribes to them on the same
machine (no network cost). The broker owns the action queue and the control
clock: a control timer publishes one action per tick to /joint_command, while a
background request thread continuously calls the remote /get_action_chunk
service and merges each returned chunk into the queue.

Real-Time Chunking (RTC):
  This broker uses a *synchronous* RTC scheme. Each round it executes exactly
  `execution_horizon` actions out of the current chunk and then stops, holding the
  last commanded position while it requests the next chunk. Because the robot is
  stationary during inference, the server sees zero inference delay, and the
  request is built from the freshest possible camera/joint observations — both of
  which maximise model accuracy at the cost of continuous motion.

  The guidance math runs on the server (it needs the model + autograd), but the
  bookkeeping lives here because only the broker owns the control clock and queue:
    - execution_horizon: how many actions of each chunk we execute before stopping
      and re-planning (a ROS param). Distinct from the server's blend_horizon, which
      is the RTC guidance window applied to the new chunk.
    - prev_chunk_left_over: the fixed, unexecuted tail of the previous chunk
      (original[execution_horizon:], in raw/normalised action space), time-aligned
      with the new chunk and sent with each request to guide the seam.
    - inference_delay: always 0 — nothing is consumed during the round trip, so no
      actions are committed while inference runs.

  When RTC is disabled the broker falls back to the original drain-then-request
  behaviour (optionally waiting pre_request_delay_sec for fresh observations).

Subscribes to (local, same machine as Isaac Sim):
  /wrist_camera/image_raw   (sensor_msgs/Image)
  /base_camera/image_raw    (sensor_msgs/Image)
  /joint_states              (sensor_msgs/JointState)

Calls (remote, over the network):
  /get_action_chunk          (pre_trained_vla_test_interfaces/srv/GetActionChunk)

Publishes to (local):
  /joint_command              (sensor_msgs/JointState)
"""

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


def _ros_image_to_compressed(
    msg: Image, jpeg_quality: int, blackout: bool = False
) -> CompressedImage:
    """Re-encode a raw sensor_msgs/Image as a JPEG-compressed sensor_msgs/CompressedImage.

    If `blackout` is set, the image content is zeroed before encoding (used for
    model short-cutting ablations — the server still receives a frame of the
    expected shape, just with no information in it).
    """
    data = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
    bgr = data[:, :, :3]
    if msg.encoding.lower() != "bgr8":
        bgr = cv2.cvtColor(bgr, cv2.COLOR_RGB2BGR)
    if blackout:
        bgr = np.zeros_like(bgr)
    ok, encoded = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    if not ok:
        raise RuntimeError("JPEG encoding failed")

    out = CompressedImage()
    out.header = msg.header
    out.format = "jpeg"
    out.data = encoded.tobytes()
    return out


class ActionQueue:
    """Minimal thread-safe action queue mirroring lerobot's RTC ActionQueue.

    Kept dependency-free (numpy only) so the broker stays lightweight and never
    imports lerobot/torch. Holds two parallel arrays:
      - processed: denormalised actions for execution (T, n_joints)
      - original:  raw/normalised actions for RTC leftover (T, action_dim)
    `last_index` is the consumption cursor advanced by get(), reset to 0 on every
    merge.

    In RTC mode consumption is capped at `execution_horizon`: get() serves at most
    that many actions from each chunk and then starves, which signals the control
    loop to stop and request the next chunk. The leftover handed to that request is
    the fixed, unexecuted tail original[execution_horizon:].
    """

    def __init__(self, rtc_enabled: bool, execution_horizon: int):
        self._lock = threading.Lock()
        self._rtc_enabled = rtc_enabled
        self._execution_horizon = execution_horizon
        self.processed: np.ndarray | None = None
        self.original: np.ndarray | None = None
        self.last_index = 0

    def _limit_locked(self) -> int:
        """How many actions of the current chunk may be executed."""
        if self.processed is None:
            return 0
        if self._rtc_enabled:
            return min(self._execution_horizon, len(self.processed))
        return len(self.processed)

    def get(self) -> np.ndarray | None:
        with self._lock:
            if self.processed is None or self.last_index >= self._limit_locked():
                return None
            action = self.processed[self.last_index].copy()
            self.last_index += 1
            return action

    def get_left_over(self) -> np.ndarray | None:
        """Fixed unexecuted tail original[execution_horizon:], for prev_chunk_left_over."""
        with self._lock:
            if self.original is None:
                return None
            tail = self.original[self._execution_horizon :]
            return tail.copy() if len(tail) > 0 else None

    def clear(self) -> None:
        with self._lock:
            self.processed = None
            self.original = None
            self.last_index = 0

    def qsize(self) -> int:
        with self._lock:
            return max(0, self._limit_locked() - self.last_index)

    def merge(self, original: np.ndarray, processed: np.ndarray) -> None:
        """Insert a new chunk.

        RTC: replace the queue with the full chunk (inference_delay is always 0, so
        nothing is dropped); get() will cap execution at execution_horizon. Non-RTC:
        append, trimming already-consumed actions to maintain continuity.
        """
        with self._lock:
            if self._rtc_enabled:
                self.processed = processed.copy()
                self.original = original.copy()
                self.last_index = 0
                return

            if self.processed is None:
                self.processed = processed.copy()
                self.original = original.copy()
            else:
                self.processed = np.concatenate([self.processed[self.last_index :], processed])
                self.original = np.concatenate([self.original[self.last_index :], original])
            self.last_index = 0


class PI0InferenceBrokerNode(Node):
    def __init__(self):
        super().__init__("pi0_inference_broker")

        self.declare_parameter("jpeg_quality", 80)
        self._jpeg_quality = self.get_parameter("jpeg_quality").get_parameter_value().integer_value

        # Model short-cutting ablation: black out one or both camera streams
        # before they're sent to the server, to test whether the model is
        # actually using that view or just shortcutting on the other one.
        self.declare_parameter("blackout_wrist_camera", False)
        self._blackout_wrist_camera = (
            self.get_parameter("blackout_wrist_camera").get_parameter_value().bool_value
        )
        self.declare_parameter("blackout_base_camera", False)
        self._blackout_base_camera = (
            self.get_parameter("blackout_base_camera").get_parameter_value().bool_value
        )

        # RTC: synchronous re-planning. When enabled, the control loop executes
        # execution_horizon actions per chunk, then stops and requests the next one
        # (inference_delay is always 0). When disabled, it uses the legacy
        # drain-the-whole-chunk-then-request behaviour.
        self.declare_parameter("enable_rtc", False)
        self._rtc_enabled = self.get_parameter("enable_rtc").get_parameter_value().bool_value

        # How many actions of each chunk to execute before stopping and re-planning.
        # The leftover sent with each request is chunk_len - execution_horizon. This
        # is independent of the server's blend_horizon (the RTC guidance window).
        self.declare_parameter("execution_horizon", 10)
        self._execution_horizon = (
            self.get_parameter("execution_horizon").get_parameter_value().integer_value
        )

        # Optionally wait before sampling observations so the request is built from
        # the freshest frames/state.
        self.declare_parameter("pre_request_delay_sec", 0.0)
        self._pre_request_delay_sec = (
            self.get_parameter("pre_request_delay_sec").get_parameter_value().double_value
        )

        # Joint positions the robot is commanded to on `reset`, so it returns to a
        # known home pose instead of holding wherever the last chunk left it. The
        # sim's home pose is all zeros (see sim_launcher.py).
        self.declare_parameter("reset_pose", [0.0] * _NUM_JOINTS)
        reset_pose = list(self.get_parameter("reset_pose").get_parameter_value().double_array_value)
        if len(reset_pose) != _NUM_JOINTS:
            self.get_logger().warn(
                f"reset_pose has {len(reset_pose)} values, expected {_NUM_JOINTS}; "
                "falling back to all-zeros."
            )
            reset_pose = [0.0] * _NUM_JOINTS
        self._reset_pose = np.array(reset_pose, dtype=np.float32)

        self._lock = threading.Lock()
        self._action_queue = ActionQueue(self._rtc_enabled, self._execution_horizon)
        # Drain-then-request gate (used by both modes).
        self._chunk_done = threading.Event()
        self._chunk_done.set()

        # Cached observations (None until first message received)
        self._obs_wrist: Image | None = None
        self._obs_base: Image | None = None
        self._joint_state: JointState | None = None

        # Last published command, republished to hold position when the queue starves.
        self._last_command: np.ndarray | None = None

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

        rtc_status = (
            f"enabled (execution_horizon={self._execution_horizon})"
            if self._rtc_enabled
            else "disabled"
        )
        self.get_logger().info(f"PI0InferenceBrokerNode ready (RTC {rtc_status}).")
        if self._blackout_wrist_camera or self._blackout_base_camera:
            self.get_logger().warn(
                f"Camera blackout ablation active: wrist={self._blackout_wrist_camera}, "
                f"base={self._blackout_base_camera}."
            )
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
                self._action_queue.clear()
                self._chunk_done.set()
                self._paused = True
                # Drive the robot back to the home pose instead of holding the last
                # commanded position, and forget that position so a later resume
                # starts from home.
                self._last_command = self._reset_pose.copy()
                self._publish_command(self._reset_pose)
                self.get_logger().info("Reset to home pose. Type 'resume' to start requesting chunks.")
            else:
                self.get_logger().warn(f"Unknown command: '{cmd}'. Use pause | resume | reset.")

    # ------------------------------------------------------------------
    # Request thread — runs independently of the ROS executor
    # ------------------------------------------------------------------

    def _request_loop(self) -> None:
        while rclpy.ok():
            if self._paused:
                time.sleep(0.05)
                continue

            # Both modes wait until the executable portion of the current chunk has
            # drained before re-requesting: the whole chunk for non-RTC, the first
            # execution_horizon actions for RTC.
            self._chunk_done.wait()
            self._chunk_done.clear()
            if self._paused:
                continue
            if self._pre_request_delay_sec > 0.0:
                time.sleep(self._pre_request_delay_sec)

            try:
                ok = self._request_step_impl()
            except Exception as e:
                self.get_logger().error(
                    f"Request step failed: {type(e).__name__}: {e}", throttle_duration_sec=5.0
                )
                ok = False

            if not ok:
                # Missing obs, service down, paused server, or error — back off a little
                # so we don't busy-spin, then re-arm the gate to retry.
                time.sleep(0.05)
                self._chunk_done.set()

    def _request_step_impl(self) -> bool:
        """Build a request, call the server, merge the response. Returns True on success."""
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
            return False

        if not self._client.wait_for_service(timeout_sec=5.0):
            self.get_logger().warn("get_action_chunk service unavailable.", throttle_duration_sec=5.0)
            return False

        # The robot is stopped while we request, so the leftover is fixed and the
        # observations are the freshest available. leftover[i] is time-aligned with
        # new_chunk[i]; inference_delay is always 0 (nothing committed during the
        # round trip).
        with self._lock:
            wrist_compressed = _ros_image_to_compressed(
                self._obs_wrist, self._jpeg_quality, blackout=self._blackout_wrist_camera
            )
            base_compressed = _ros_image_to_compressed(
                self._obs_base, self._jpeg_quality, blackout=self._blackout_base_camera
            )
            joint_state = self._joint_state

        leftover = self._action_queue.get_left_over() if self._rtc_enabled else None

        request = GetActionChunk.Request()
        request.wrist_image = wrist_compressed
        request.base_image = base_compressed
        request.joint_state = joint_state
        request.inference_delay = 0
        if leftover is not None and len(leftover) > 0:
            request.prev_chunk_left_over = leftover.reshape(-1).astype(np.float32).tolist()
            request.prev_chunk_action_dim = int(leftover.shape[1])
        else:
            request.prev_chunk_action_dim = 0

        self.get_logger().debug("Requesting action chunk ...")
        done = threading.Event()
        future = self._client.call_async(request)
        future.add_done_callback(lambda _f: done.set())
        if not done.wait(timeout=30.0):
            future.cancel()
            self.get_logger().warn("get_action_chunk request timed out.", throttle_duration_sec=5.0)
            return False

        if future.exception() is not None:
            raise future.exception()
        response = future.result()
        if response is None:
            return False

        # Never merge an empty chunk — that would wipe the queue. Paused/resetting
        # server or any error just leaves the existing queue to keep draining.
        if response.status == GetActionChunk.Response.STATUS_PAUSED:
            self.get_logger().info("Server paused — holding queue.", throttle_duration_sec=5.0)
            return False
        if response.status != GetActionChunk.Response.STATUS_OK or not response.action_chunk:
            self.get_logger().warn(
                f"Non-OK / empty response (status={response.status}).", throttle_duration_sec=5.0
            )
            return False

        # Processed actions for execution (T, n_joints).
        processed = np.array(
            [js.position for js in response.action_chunk], dtype=np.float32
        )

        if self._rtc_enabled:
            if self._execution_horizon >= len(processed):
                self.get_logger().warn(
                    f"execution_horizon ({self._execution_horizon}) >= chunk length "
                    f"({len(processed)}) — no leftover to guide the seam; RTC has no "
                    "effect. Lower execution_horizon.",
                    throttle_duration_sec=5.0,
                )

            adim = response.raw_chunk_action_dim
            if adim > 0 and len(response.raw_chunk) > 0:
                original = np.asarray(response.raw_chunk, dtype=np.float32).reshape(-1, adim)
            else:
                # Server returned no raw chunk (shouldn't happen with RTC on) — fall
                # back to processed so leftover tracking still works.
                original = processed
            self._action_queue.merge(original, processed)
        else:
            self._action_queue.merge(processed, processed)

        return True

    # ------------------------------------------------------------------
    # Control timer (30 Hz)
    # ------------------------------------------------------------------

    def _control_step(self) -> None:
        if self._paused:
            return

        action = self._action_queue.get()
        if action is None:
            # Queue starved — hold the last commanded position rather than jumping.
            if self._last_command is None:
                return
            action = self._last_command
        else:
            self._last_command = action
            if self._action_queue.qsize() == 0:
                # Executable portion just emptied (full chunk for non-RTC, the first
                # execution_horizon actions for RTC) — trigger the next request.
                self._chunk_done.set()

        self._publish_command(action)

    def _publish_command(self, action: np.ndarray) -> None:
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = _JOINT_NAMES
        msg.position = action.tolist()
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
