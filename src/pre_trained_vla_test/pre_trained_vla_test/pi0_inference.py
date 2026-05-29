"""
pi0_inference.py
----------------
ROS2 node that runs PI0Policy (lerobot/pi0_base) against an SO-101 arm in Isaac Sim.

Subscribes to:
  /wrist_camera/image_raw   (sensor_msgs/Image)
  /scene_camera/image_raw   (sensor_msgs/Image)
  /base_camera/image_raw    (sensor_msgs/Image)
  /joint_states             (sensor_msgs/JointState)

Publishes to:
  /joint_command            (sensor_msgs/JointState)

Requirements:
  - ROS2 sourced
  - lerobot venv sourced
  - CUDA GPU available
"""

import collections
import threading

import numpy as np
import rclpy
import torch
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState

from lerobot.policies import make_pre_post_processors
from lerobot.policies.pi0 import PI0Policy

_JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]
_NUM_JOINTS = len(_JOINT_NAMES)

_MODEL_ID = "lerobot/pi0_base"
_INFERENCE_HZ = 5.0
_CONTROL_HZ = 50.0


def _ros_image_to_tensor(msg: Image) -> torch.Tensor:
    """Convert sensor_msgs/Image → (1, H, W, 3) uint8 CPU tensor (HWC, RGB)."""
    data = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
    rgb = data[:, :, :3].copy()  # drop alpha if present
    return torch.from_numpy(rgb).unsqueeze(0)  # (1, H, W, 3)


class PI0InferenceNode(Node):
    def __init__(self):
        super().__init__("pi0_inference")

        self.declare_parameter("prompt", "pick up the object")

        self.get_logger().info(f"Loading {_MODEL_ID} ...")
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._policy = PI0Policy.from_pretrained(_MODEL_ID)
        self._policy.eval()
        self._policy.to(self._device)

        # Normalization stats are loaded from the pretrained checkpoint
        self._preprocessor, self._postprocessor = make_pre_post_processors(
            self._policy.config,
            pretrained_path=_MODEL_ID,
        )

        self._lock = threading.Lock()
        self._action_queue: collections.deque = collections.deque()

        # Cached observations (None until first message received)
        self._obs_wrist: Image | None = None
        self._obs_scene: Image | None = None
        self._obs_base: Image | None = None
        self._joint_positions: list[float] | None = None  # radians, len == _NUM_JOINTS

        # Subscribers
        self.create_subscription(Image, "/wrist_camera/image_raw", self._cb_wrist, 1)
        self.create_subscription(Image, "/scene_camera/image_raw", self._cb_scene, 1)
        self.create_subscription(Image, "/base_camera/image_raw", self._cb_base, 1)
        self.create_subscription(JointState, "/joint_states", self._cb_joints, 1)

        # Publisher
        self._pub = self.create_publisher(JointState, "/joint_command", 10)

        # Timers
        self.create_timer(1.0 / _INFERENCE_HZ, self._inference_step)
        self.create_timer(1.0 / _CONTROL_HZ, self._control_step)

        self._reset_episode()
        self.get_logger().info("PI0InferenceNode ready.")

    # ------------------------------------------------------------------
    # Subscriber callbacks — cache latest message
    # ------------------------------------------------------------------

    def _cb_wrist(self, msg: Image) -> None:
        with self._lock:
            self._obs_wrist = msg

    def _cb_scene(self, msg: Image) -> None:
        with self._lock:
            self._obs_scene = msg

    def _cb_base(self, msg: Image) -> None:
        with self._lock:
            self._obs_base = msg

    def _cb_joints(self, msg: JointState) -> None:
        name_to_pos = dict(zip(msg.name, msg.position))
        try:
            positions = [name_to_pos[n] for n in _JOINT_NAMES]
        except KeyError:
            return  # Not all joints present yet
        with self._lock:
            self._joint_positions = positions

    # ------------------------------------------------------------------
    # Episode management
    # ------------------------------------------------------------------

    def _reset_episode(self) -> None:
        self._policy.reset()
        with self._lock:
            self._action_queue.clear()

    # ------------------------------------------------------------------
    # Inference timer (~5 Hz)
    # ------------------------------------------------------------------

    def _inference_step(self) -> None:
        with self._lock:
            if any(
                v is None
                for v in (self._obs_wrist, self._obs_scene, self._obs_base, self._joint_positions)
            ):
                return  # Observations not ready yet

            wrist_t = _ros_image_to_tensor(self._obs_wrist)
            scene_t = _ros_image_to_tensor(self._obs_scene)
            base_t = _ros_image_to_tensor(self._obs_base)
            state_t = torch.tensor(self._joint_positions, dtype=torch.float32).unsqueeze(0)  # (1, 6)

        prompt = self.get_parameter("prompt").get_parameter_value().string_value

        # LeRobot-convention observation batch.
        # Keys use dot notation: observation.state and observation.images.<cam>.
        # Images are (1, H, W, 3) uint8; the preprocessor handles resizing and normalisation.
        # State is (1, 6) float32 in radians; the preprocessor handles normalisation.
        raw_obs = {
            "observation.state": state_t,
            "observation.images.wrist_camera": wrist_t,
            "observation.images.scene_camera": scene_t,
            "observation.images.base_camera": base_t,
            "prompt": prompt,
        }

        try:
            self.get_logger().debug("Running inference ...")
            batch = self._preprocessor(raw_obs)
            batch = {
                k: v.to(self._device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            with torch.no_grad():
                actions = self._policy.predict_action_chunk(batch)
                # actions: (1, chunk_size, action_dim) tensor

            # Postprocessor denormalises actions back to joint space
            actions = self._postprocessor(actions)

            if actions.dim() == 3:
                actions = actions.squeeze(0)  # → (chunk_size, action_dim)

            # Slice to the joints we control
            actions = actions[:, :_NUM_JOINTS]  # (chunk_size, 6)

        except Exception as e:
            self.get_logger().error(f"Inference failed: {e}")
            return

        with self._lock:
            self._action_queue = collections.deque(actions.cpu().unbind(dim=0))

    # ------------------------------------------------------------------
    # Control timer (50 Hz)
    # ------------------------------------------------------------------

    def _control_step(self) -> None:
        with self._lock:
            if not self._action_queue:
                return
            action = self._action_queue.popleft()  # (6,) tensor

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = _JOINT_NAMES
        msg.position = action.tolist()
        self._pub.publish(msg)


def main() -> None:
    rclpy.init()
    node = PI0InferenceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
