"""ROS2 node that records a LeRobotDataset from a SO-101 arm in Isaac Sim."""

import threading
import time
from enum import Enum, auto
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState

from lerobot.datasets.lerobot_dataset import LeRobotDataset


class State(Enum):
    WAITING_FOR_SENSORS = auto()
    IDLE = auto()
    RECORDING = auto()
    RESETTING = auto()
    DONE = auto()


class DatasetRecorderNode(Node):
    def __init__(self):
        super().__init__("dataset_recorder")

        self.declare_parameter("num_episodes", 10)
        self.declare_parameter("episode_duration", 60.0)
        self.declare_parameter("reset_duration", 5.0)
        self.declare_parameter("task_description", "pick and place")
        self.declare_parameter("fps", 30)
        self.declare_parameter("dataset_name", "so101_dataset")
        self.declare_parameter("output_dir", "")

        self._num_episodes = self.get_parameter("num_episodes").value
        self._episode_duration = self.get_parameter("episode_duration").value
        self._reset_duration = self.get_parameter("reset_duration").value
        self._task_description = self.get_parameter("task_description").value
        self._fps = self.get_parameter("fps").value
        self._dataset_name = self.get_parameter("dataset_name").value
        output_dir = self.get_parameter("output_dir").value

        self._root = Path(output_dir).expanduser() if output_dir else None

        self._lock = threading.Lock()
        self._img_wrist: np.ndarray | None = None
        self._img_base: np.ndarray | None = None
        self._joint_state: np.ndarray | None = None
        self._joint_command: np.ndarray | None = None
        self._joint_names: list[str] | None = None

        self._state = State.WAITING_FOR_SENSORS
        self._dataset: LeRobotDataset | None = None
        self._dataset_ready = False
        self._episode_idx = 0
        self._frame_count = 0
        self._episode_start_time: float | None = None

        self._start_event = threading.Event()
        self._stop_event = threading.Event()

        self.create_subscription(Image, "/wrist_camera/image_raw", self._cb_wrist, 1)
        self.create_subscription(Image, "/base_camera/image_raw", self._cb_base, 1)
        self.create_subscription(JointState, "/joint_states", self._cb_joints, 1)
        self.create_subscription(JointState, "/joint_command", self._cb_command, 1)

        self.create_timer(1.0 / self._fps, self._step)

        self._stdin_thread = threading.Thread(target=self._stdin_loop, daemon=True)
        self._stdin_thread.start()

        self.get_logger().info("Dataset recorder started — waiting for all sensor topics...")

    # ------------------------------------------------------------------
    # Subscription callbacks
    # ------------------------------------------------------------------

    def _ros_image_to_numpy(self, msg: Image) -> np.ndarray:
        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
        if msg.encoding.lower() == "bgr8":
            arr = arr[:, :, ::-1].copy()
        return arr[:, :, :3]

    def _cb_wrist(self, msg: Image) -> None:
        with self._lock:
            self._img_wrist = self._ros_image_to_numpy(msg)

    def _cb_base(self, msg: Image) -> None:
        with self._lock:
            self._img_base = self._ros_image_to_numpy(msg)

    def _reorder_joints(self, msg: JointState) -> np.ndarray:
        """Return positions in the canonical joint order established from /joint_states."""
        if self._joint_names is None:
            return np.array(msg.position, dtype=np.float32)
        idx = [list(msg.name).index(n) for n in self._joint_names if n in msg.name]
        return np.array([msg.position[i] for i in idx], dtype=np.float32)

    def _cb_joints(self, msg: JointState) -> None:
        with self._lock:
            if self._joint_names is None and msg.name:
                self._joint_names = list(msg.name)
            self._joint_state = self._reorder_joints(msg)

    def _cb_command(self, msg: JointState) -> None:
        with self._lock:
            self._joint_command = self._reorder_joints(msg)

    # ------------------------------------------------------------------
    # Stdin loop (daemon thread)
    # ------------------------------------------------------------------

    def _stdin_loop(self) -> None:
        while True:
            input()  # block until Enter
            if self._state == State.IDLE:
                self._start_event.set()
            elif self._state == State.RECORDING:
                self._stop_event.set()

    # ------------------------------------------------------------------
    # Main timer step
    # ------------------------------------------------------------------

    def _step(self) -> None:
        if self._state == State.WAITING_FOR_SENSORS:
            self._try_init_dataset()
        elif self._state == State.IDLE:
            self._check_start()
        elif self._state == State.RECORDING:
            self._record_frame()
        elif self._state == State.RESETTING:
            pass  # handled inside _finish_episode via timer/sleep in a thread

    def _try_init_dataset(self) -> None:
        with self._lock:
            ready = all(
                x is not None
                for x in [
                    self._img_wrist,
                    self._img_base,
                    self._joint_state,
                    self._joint_command,
                ]
            )
            if not ready:
                return
            wrist_shape = list(self._img_wrist.shape)
            base_shape = list(self._img_base.shape)
            n_joints = len(self._joint_state)

        features = {
            "observation.images.wrist": {"dtype": "video", "shape": wrist_shape},
            "observation.images.base": {"dtype": "video", "shape": base_shape},
            "observation.state": {"dtype": "float32", "shape": [n_joints]},
            "action": {"dtype": "float32", "shape": [n_joints]},
        }

        kwargs = dict(
            repo_id=self._dataset_name,
            fps=self._fps,
            features=features,
            robot_type="so101_follower",
            use_videos=True,
            image_writer_threads=4,
        )
        if self._root is not None:
            kwargs["root"] = self._root

        self._dataset = LeRobotDataset.create(**kwargs)
        self._dataset_ready = True
        self._state = State.IDLE

        self.get_logger().info(
            f"All sensors ready. Dataset '{self._dataset_name}' created. "
            f"Joints: {self._joint_names}. "
            f"Press Enter to start episode 1 of {self._num_episodes}."
        )

    def _check_start(self) -> None:
        if self._start_event.is_set():
            self._start_event.clear()
            self._stop_event.clear()
            self._frame_count = 0
            self._episode_start_time = self.get_clock().now().nanoseconds * 1e-9
            self._state = State.RECORDING
            self.get_logger().info(
                f"[Episode {self._episode_idx + 1}/{self._num_episodes}] Recording started. "
                f"Press Enter to stop (max {self._episode_duration}s)."
            )

    def _record_frame(self) -> None:
        elapsed = self.get_clock().now().nanoseconds * 1e-9 - self._episode_start_time
        timed_out = elapsed >= self._episode_duration
        stopped = self._stop_event.is_set()

        if stopped or timed_out:
            if timed_out:
                self.get_logger().info("Episode duration cap reached — stopping episode.")
            self._finish_episode()
            return

        with self._lock:
            if any(
                x is None
                for x in [
                    self._img_wrist,
                    self._img_base,
                    self._joint_state,
                    self._joint_command,
                ]
            ):
                return

            frame = {
                "observation.images.wrist": self._img_wrist.copy(),
                "observation.images.base": self._img_base.copy(),
                "observation.state": self._joint_state.copy(),
                "action": self._joint_command.copy(),
                "task": self._task_description,
            }

        self._dataset.add_frame(frame)
        self._frame_count += 1

    def _finish_episode(self) -> None:
        self._state = State.RESETTING
        self._stop_event.clear()

        self.get_logger().info(
            f"[Episode {self._episode_idx + 1}] Stopped — {self._frame_count} frames. "
            "Saving episode..."
        )
        self._dataset.save_episode()
        self._episode_idx += 1

        if self._episode_idx >= self._num_episodes:
            self._state = State.DONE
            self._dataset.finalize()
            self.get_logger().info(
                f"All {self._num_episodes} episodes recorded. "
                f"Dataset saved to: {self._dataset.root}"
            )
            rclpy.shutdown()
            return

        self.get_logger().info(
            f"Resetting for {self._reset_duration}s... "
            f"({self._num_episodes - self._episode_idx} episodes remaining)"
        )
        threading.Thread(target=self._reset_wait, daemon=True).start()

    def _reset_wait(self) -> None:
        time.sleep(self._reset_duration)
        self._state = State.IDLE
        self.get_logger().info(
            f"Ready. Press Enter to start episode {self._episode_idx + 1} of {self._num_episodes}."
        )


def main(args=None):
    rclpy.init(args=args)
    node = DatasetRecorderNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node._dataset is not None and node._state != State.DONE:
            node.get_logger().info("Interrupted — finalizing dataset...")
            node._dataset.finalize()
        node.destroy_node()
