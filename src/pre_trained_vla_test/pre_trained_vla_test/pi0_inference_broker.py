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
  RTC is an *asynchronous* scheme — the request thread re-plans continuously while
  the control timer keeps draining the queue, so chunks overlap in time. The
  guidance math runs on the server (it needs the model + autograd), but the
  bookkeeping lives here because only the broker owns the control clock and queue:
    - prev_chunk_left_over: the unexecuted, time-aligned tail of the previous
      chunk in raw/normalised action space, sent with each request.
    - inference_delay: how many actions the robot consumes during one round trip.
      We can't know the future, so we *predict* it from an EMA of past round
      trips (in action units), send the prediction, and measure the actual count
      afterwards to update the estimate.

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


class ActionQueue:
    """Minimal thread-safe action queue mirroring lerobot's RTC ActionQueue.

    Kept dependency-free (numpy only) so the broker stays lightweight and never
    imports lerobot/torch. Holds two parallel arrays:
      - processed: denormalised actions for execution (T, n_joints)
      - original:  raw/normalised actions for RTC leftover (T, action_dim)
    `last_index` is the consumption cursor advanced by get(); it doubles as the
    per-chunk counter used to measure how many actions were consumed during a
    round trip (it resets to 0 on every merge).
    """

    def __init__(self, rtc_enabled: bool):
        self._lock = threading.Lock()
        self._rtc_enabled = rtc_enabled
        self.processed: np.ndarray | None = None
        self.original: np.ndarray | None = None
        self.last_index = 0

    def get(self) -> np.ndarray | None:
        with self._lock:
            if self.processed is None or self.last_index >= len(self.processed):
                return None
            action = self.processed[self.last_index].copy()
            self.last_index += 1
            return action

    def get_action_index(self) -> int:
        with self._lock:
            return self.last_index

    def get_left_over(self) -> np.ndarray | None:
        """Unconsumed raw actions, for RTC prev_chunk_left_over."""
        with self._lock:
            return self._left_over_locked()

    def _left_over_locked(self) -> np.ndarray | None:
        if self.original is None or self.last_index >= len(self.original):
            return None
        return self.original[self.last_index :].copy()

    def snapshot(self) -> tuple[np.ndarray | None, int]:
        """Atomically capture (leftover, consumption index) for an RTC request."""
        with self._lock:
            return self._left_over_locked(), self.last_index

    def clear(self) -> None:
        with self._lock:
            self.processed = None
            self.original = None
            self.last_index = 0

    def qsize(self) -> int:
        with self._lock:
            if self.processed is None:
                return 0
            return max(0, len(self.processed) - self.last_index)

    def merge(self, original: np.ndarray, processed: np.ndarray, delay: int) -> None:
        """Insert a new chunk.

        RTC: replace the queue, dropping the first `delay` actions (the robot
        already committed to them during inference). Non-RTC: append, trimming
        already-consumed actions to maintain continuity.
        """
        with self._lock:
            if self._rtc_enabled:
                d = max(0, min(delay, len(processed)))
                self.processed = processed[d:].copy()
                self.original = original[d:].copy()
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

        # RTC: when enabled, the request thread runs continuously (never drains to
        # empty). When disabled, it uses the legacy drain-then-request behaviour.
        self.declare_parameter("enable_rtc", False)
        self._rtc_enabled = self.get_parameter("enable_rtc").get_parameter_value().bool_value

        # Initial guess for inference_delay (in action steps), used until the EMA
        # has seen real round trips. The EMA smoothing factor controls how quickly
        # the estimate adapts to measured latency.
        self.declare_parameter("initial_inference_delay", 0)
        self._delay_ema = float(
            self.get_parameter("initial_inference_delay").get_parameter_value().integer_value
        )
        self.declare_parameter("delay_ema_alpha", 0.3)
        self._delay_ema_alpha = (
            self.get_parameter("delay_ema_alpha").get_parameter_value().double_value
        )

        # Non-RTC only: optionally wait before sampling observations so the request
        # is built from the freshest frames/state.
        self.declare_parameter("pre_request_delay_sec", 0.0)
        self._pre_request_delay_sec = (
            self.get_parameter("pre_request_delay_sec").get_parameter_value().double_value
        )

        self._lock = threading.Lock()
        self._action_queue = ActionQueue(self._rtc_enabled)
        # Non-RTC drain-then-request gate.
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

        self.get_logger().info(
            f"PI0InferenceBrokerNode ready (RTC {'enabled' if self._rtc_enabled else 'disabled'})."
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
                self.get_logger().info("Reset. Type 'resume' to start requesting chunks.")
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

            if not self._rtc_enabled:
                # Legacy behaviour: wait until the queue empties before re-requesting.
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
                # so we don't busy-spin, then retry. (In RTC mode there is no queue-empty
                # signal to wait on, so this also serves as the retry cadence.)
                time.sleep(0.05)
                if not self._rtc_enabled:
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

        # Snapshot the leftover and the consumption cursor together: both are anchored
        # to this instant. leftover[i] is time-aligned with new_chunk[i]; idx0 lets us
        # measure how many actions are consumed during the round trip.
        with self._lock:
            wrist_compressed = _ros_image_to_compressed(self._obs_wrist, self._jpeg_quality)
            base_compressed = _ros_image_to_compressed(self._obs_base, self._jpeg_quality)
            joint_state = self._joint_state

        if self._rtc_enabled:
            leftover, idx0 = self._action_queue.snapshot()
        else:
            leftover, idx0 = None, 0

        # Predict inference_delay from the EMA. On a cold start (no leftover) nothing
        # has been executed yet, so force 0 — we must not drop any of the first chunk.
        if leftover is None or len(leftover) == 0:
            predicted_delay = 0
        else:
            predicted_delay = max(0, int(round(self._delay_ema)))

        request = GetActionChunk.Request()
        request.wrist_image = wrist_compressed
        request.base_image = base_compressed
        request.joint_state = joint_state
        request.inference_delay = predicted_delay
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
            # Measure how many actions were actually consumed during the round trip and
            # fold it into the EMA for the next prediction. The cursor advanced from
            # idx0 (and has not been reset since — this loop is the only merger).
            measured_delay = max(0, self._action_queue.get_action_index() - idx0)
            self._delay_ema = (
                self._delay_ema_alpha * measured_delay
                + (1.0 - self._delay_ema_alpha) * self._delay_ema
            )
            if abs(measured_delay - predicted_delay) > 1:
                self.get_logger().warn(
                    f"inference_delay mismatch: predicted={predicted_delay}, "
                    f"measured={measured_delay}, ema={self._delay_ema:.2f}.",
                    throttle_duration_sec=5.0,
                )
            if predicted_delay >= len(processed):
                self.get_logger().warn(
                    "Inference slower than a full chunk — the queue will starve. "
                    "Reduce chunk consumption rate or speed up the model.",
                    throttle_duration_sec=5.0,
                )

            adim = response.raw_chunk_action_dim
            if adim > 0 and len(response.raw_chunk) > 0:
                original = np.asarray(response.raw_chunk, dtype=np.float32).reshape(-1, adim)
            else:
                # Server returned no raw chunk (shouldn't happen with RTC on) — fall
                # back to processed so leftover tracking still works.
                original = processed
            self._action_queue.merge(original, processed, predicted_delay)
        else:
            self._action_queue.merge(processed, processed, 0)

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
            if not self._rtc_enabled and self._action_queue.qsize() == 0:
                self._chunk_done.set()  # Queue just emptied — trigger next request

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
