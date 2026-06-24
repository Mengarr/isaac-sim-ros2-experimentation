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
    CONFIRMING = auto()
    DONE = auto()


class DatasetRecorderNode(Node):
    def __init__(self):
        super().__init__("dataset_recorder")

        self.declare_parameter("num_episodes", 10)
        self.declare_parameter("episode_duration", 60.0)
        self.declare_parameter("task_description", "pick and place")
        self.declare_parameter("fps", 30)
        self.declare_parameter("dataset_name", "so101_dataset")
        self.declare_parameter("output_dir", "")
        self.declare_parameter("streaming_encoding", False)
        self.declare_parameter("resume", False)

        self._num_episodes = self.get_parameter("num_episodes").value
        self._episode_duration = self.get_parameter("episode_duration").value
        self._task_description = self.get_parameter("task_description").value
        self._fps = self.get_parameter("fps").value
        self._dataset_name = self.get_parameter("dataset_name").value
        output_dir = self.get_parameter("output_dir").value
        self._streaming_encoding = self.get_parameter("streaming_encoding").value
        self._resume = self.get_parameter("resume").value

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
        self._confirm_event = threading.Event()
        self._restart_event = threading.Event()

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
            line = input().strip()
            if line == "q":
                self._quit()
                return
            if line == "r":
                if self._state in (State.RECORDING, State.CONFIRMING):
                    self._restart_event.set()
            elif self._state == State.IDLE:
                self._start_event.set()
            elif self._state == State.RECORDING:
                self._stop_event.set()
            elif self._state == State.CONFIRMING:
                self._confirm_event.set()

    def _quit(self) -> None:
        self.get_logger().info("Quit requested — finalizing dataset...")
        if self._state == State.RECORDING:
            self._dataset.save_episode()
        if self._dataset is not None:
            self._dataset.finalize()
            self.get_logger().info(f"Dataset saved to: {self._dataset.root}")
        rclpy.shutdown()

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
        elif self._state == State.CONFIRMING:
            self._check_confirm()

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

        joint_names = list(self._joint_names) if self._joint_names else [str(i) for i in range(n_joints)]

        if self._resume:
            if self._root is None:
                raise ValueError("resume=True requires output_dir to point at the existing dataset.")
            self._dataset = LeRobotDataset.resume(
                repo_id=self._dataset_name,
                root=self._root,
                image_writer_threads=0 if self._streaming_encoding else 4,
                streaming_encoding=self._streaming_encoding,
            )
            self._check_resume_compatibility(wrist_shape, base_shape, n_joints, joint_names)
            self._episode_idx = self._dataset.meta.total_episodes
            self._dataset_ready = True
            self._state = State.IDLE
            self.get_logger().info(
                f"All sensors ready. Resumed dataset '{self._dataset_name}' with "
                f"{self._episode_idx} existing episodes. Joints: {self._joint_names}. "
                f"Press Enter to start episode {self._episode_idx + 1} of {self._num_episodes}."
            )
            return

        features = {
            "observation.images.wrist": {"dtype": "video", "shape": wrist_shape, "names": ["height", "width", "channel"]},
            "observation.images.base": {"dtype": "video", "shape": base_shape, "names": ["height", "width", "channel"]},
            "observation.state": {"dtype": "float32", "shape": [n_joints], "names": joint_names},
            "action": {"dtype": "float32", "shape": [n_joints], "names": joint_names},
        }

        kwargs = dict(
            repo_id=self._dataset_name,
            fps=self._fps,
            features=features,
            robot_type="so101_follower",
            use_videos=True,
            image_writer_threads=0 if self._streaming_encoding else 4,
            streaming_encoding=self._streaming_encoding,
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

    def _check_resume_compatibility(self, wrist_shape, base_shape, n_joints, joint_names) -> None:
        """Fail fast if live sensor shapes don't match the resumed dataset's schema.

        Appending frames with a different shape/dtype than the existing metadata
        would silently corrupt the dataset, so we abort before recording.
        """
        meta_features = self._dataset.meta.features
        expected = {
            "observation.images.wrist": tuple(wrist_shape),
            "observation.images.base": tuple(base_shape),
            "observation.state": (n_joints,),
            "action": (n_joints,),
        }

        mismatches = []
        for key, live_shape in expected.items():
            if key not in meta_features:
                mismatches.append(f"{key}: missing from existing dataset")
                continue
            old_shape = tuple(meta_features[key]["shape"])
            if old_shape != live_shape:
                mismatches.append(f"{key}: existing {old_shape} != live {live_shape}")

        old_joint_names = meta_features.get("observation.state", {}).get("names")
        if old_joint_names is not None and list(old_joint_names) != list(joint_names):
            mismatches.append(
                f"joint order/names differ: existing {list(old_joint_names)} != live {list(joint_names)}"
            )

        if mismatches:
            raise ValueError(
                "Cannot resume — recorded data is incompatible with the existing dataset:\n  "
                + "\n  ".join(mismatches)
            )

        self.get_logger().info("Resume compatibility check passed — sensor shapes match existing dataset.")

    def _check_start(self) -> None:
        if self._start_event.is_set():
            self._start_event.clear()
            self._stop_event.clear()
            self._frame_count = 0
            self._episode_start_time = self.get_clock().now().nanoseconds * 1e-9
            self._state = State.RECORDING
            self.get_logger().info(
                f"[Episode {self._episode_idx + 1}/{self._num_episodes}] Recording started. "
                f"Press Enter to stop, 'r' to discard (max {self._episode_duration}s)."
            )

    def _record_frame(self) -> None:
        if self._restart_event.is_set():
            self._discard_episode()
            return

        elapsed = self.get_clock().now().nanoseconds * 1e-9 - self._episode_start_time
        timed_out = elapsed >= self._episode_duration
        stopped = self._stop_event.is_set()

        if stopped or timed_out:
            if timed_out:
                self.get_logger().info("Episode duration cap reached.")
            self._stop_recording()
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

    def _stop_recording(self) -> None:
        self._stop_event.clear()
        self._state = State.CONFIRMING
        self.get_logger().info(
            f"[Episode {self._episode_idx + 1}] Stopped — {self._frame_count} frames captured. "
            "Press Enter to save, or 'r' to discard and re-record."
        )

    def _check_confirm(self) -> None:
        if self._restart_event.is_set():
            self._discard_episode()
        elif self._confirm_event.is_set():
            self._save_and_advance()

    def _discard_episode(self) -> None:
        self._dataset.clear_episode_buffer()
        self._restart_event.clear()
        self._frame_count = 0
        self._state = State.IDLE
        self.get_logger().info(
            f"Episode discarded. Press Enter to re-record episode {self._episode_idx + 1} "
            f"of {self._num_episodes}."
        )

    def _save_and_advance(self) -> None:
        self._confirm_event.clear()
        self.get_logger().info(f"[Episode {self._episode_idx + 1}] Saving...")
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

        self._state = State.IDLE
        self.get_logger().info(
            f"Episode saved. Press Enter to start episode {self._episode_idx + 1} "
            f"of {self._num_episodes}."
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
