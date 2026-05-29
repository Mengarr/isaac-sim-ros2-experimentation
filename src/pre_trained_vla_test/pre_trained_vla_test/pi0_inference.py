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

Camera → openpi slot mapping:
  base_camera  → image.base_0_rgb       (exterior/base view)
  wrist_camera → image.left_wrist_0_rgb (wrist view)
  scene_camera → image.right_wrist_0_rgb (scene overview)
"""

import collections
import threading

import numpy as np
import rclpy
import torch
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState

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
_IMAGE_SIZE = 224   # pi0 pre-trained image resolution
_INFERENCE_HZ = 5.0
_CONTROL_HZ = 50.0


def _resize_with_pad(img: np.ndarray, size: int) -> np.ndarray:
    """Resize HWC uint8 image to (size, size) with letterbox padding, no distortion."""
    from PIL import Image as PILImage
    h, w = img.shape[:2]
    scale = size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    pil = PILImage.fromarray(img).resize((new_w, new_h), PILImage.BILINEAR)
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    pad_y = (size - new_h) // 2
    pad_x = (size - new_w) // 2
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = np.asarray(pil)
    return canvas


def _ros_image_to_numpy(msg: Image) -> np.ndarray:
    """Convert sensor_msgs/Image → HWC uint8 RGB numpy array."""
    data = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
    return data[:, :, :3].copy()  # drop alpha if present


class PI0InferenceNode(Node):
    def __init__(self):
        super().__init__("pi0_inference")

        self.declare_parameter("prompt", "pick up the object")

        self.get_logger().info(f"Loading {_MODEL_ID} ...")
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._policy = PI0Policy.from_pretrained(_MODEL_ID)
        self._policy.eval()
        self._policy.to(self._device)
        self._action_dim: int = self._policy.config.action_dim

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

            wrist_np = _ros_image_to_numpy(self._obs_wrist)
            scene_np = _ros_image_to_numpy(self._obs_scene)
            base_np = _ros_image_to_numpy(self._obs_base)
            joint_positions = list(self._joint_positions)

        prompt = self.get_parameter("prompt").get_parameter_value().string_value

        # Resize to 224×224 with letterbox padding
        wrist_img = _resize_with_pad(wrist_np, _IMAGE_SIZE)
        scene_img = _resize_with_pad(scene_np, _IMAGE_SIZE)
        base_img = _resize_with_pad(base_np, _IMAGE_SIZE)

        # Pad state from _NUM_JOINTS to model's action_dim
        state = np.array(joint_positions, dtype=np.float32)
        state = np.pad(state, (0, max(0, self._action_dim - _NUM_JOINTS)))

        # Build openpi-native observation batch
        # Images: (H, W, 3) uint8 numpy — model handles normalisation internally
        # State: unnormalized, padded to action_dim — model handles normalisation internally
        batch = {
            "state": state,
            "image": {
                "base_0_rgb": base_img,
                "left_wrist_0_rgb": wrist_img,
                "right_wrist_0_rgb": scene_img,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
            "prompt": prompt,
        }

        try:
            self.get_logger().debug("Running inference ...")
            with torch.no_grad():
                actions = self._policy.predict_action_chunk(batch)
                # actions: (1, chunk_size, action_dim) or (chunk_size, action_dim)

            if isinstance(actions, torch.Tensor):
                actions = actions.cpu().numpy()
            if actions.ndim == 3:
                actions = actions.squeeze(0)  # → (chunk_size, action_dim)

            # Keep only the first _NUM_JOINTS dims (SO-101 has 6 joints)
            actions = actions[:, :_NUM_JOINTS]  # (chunk_size, 6)

        except Exception as e:
            self.get_logger().error(f"Inference failed: {e}")
            return

        with self._lock:
            self._action_queue = collections.deque(actions)  # each element: (6,) array

    # ------------------------------------------------------------------
    # Control timer (50 Hz)
    # ------------------------------------------------------------------

    def _control_step(self) -> None:
        with self._lock:
            if not self._action_queue:
                return
            action = self._action_queue.popleft()  # (6,) numpy array

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
