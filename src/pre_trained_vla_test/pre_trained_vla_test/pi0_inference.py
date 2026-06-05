"""
pi0_inference.py
----------------
ROS2 node that runs PI0Policy against an SO-101 arm in Isaac Sim.

Subscribes to:
  /wrist_camera/image_raw   (sensor_msgs/Image)
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
import sys
import threading
import time

import numpy as np
import rclpy
import torch
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState

from lerobot.policies import make_pre_post_processors
from lerobot.policies.pi0 import PI0Policy
from lerobot.policies.pi05 import PI05Policy

_POLICY_CLASSES = {
    "pi0": PI0Policy,
    "pi05": PI05Policy,
}

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


def _ros_image_to_tensor(msg: Image) -> torch.Tensor:
    """Convert sensor_msgs/Image → (1, 3, H, W) uint8 CPU tensor (CHW, RGB)."""
    data = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
    rgb = data[:, :, :3].copy()
    if msg.encoding.lower() == "bgr8":
        rgb = rgb[:, :, ::-1].copy()
    return torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)


class PI0InferenceNode(Node):
    def __init__(self):
        super().__init__("pi0_inference")

        self.declare_parameter("model_type", "pi05")  # "pi0" or "pi05"
        self.declare_parameter("model_path", "")
        self.declare_parameter("prompt", "pick up the object")
        self.declare_parameter("lora_adapter_path", "")

        model_type = self.get_parameter("model_type").get_parameter_value().string_value
        if model_type not in _POLICY_CLASSES:
            raise ValueError(f"model_type must be one of {list(_POLICY_CLASSES)}, got '{model_type}'")
        PolicyClass = _POLICY_CLASSES[model_type]

        _DEFAULT_MODEL_PATHS = {"pi0": "lerobot/pi0_base", "pi05": "lerobot/pi05_libero"}
        model_path = self.get_parameter("model_path").get_parameter_value().string_value
        if not model_path:
            model_path = _DEFAULT_MODEL_PATHS[model_type]

        lora_adapter_path = self.get_parameter("lora_adapter_path").get_parameter_value().string_value

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if lora_adapter_path:
            from peft import PeftConfig, PeftModel
            peft_config = PeftConfig.from_pretrained(lora_adapter_path)
            base_path = peft_config.base_model_name_or_path
            self.get_logger().info(f"Loading base model from {base_path} ...")
            self._policy = PolicyClass.from_pretrained(base_path)
            self.get_logger().info(f"Applying LoRA adapter from {lora_adapter_path} ...")
            self._policy = PeftModel.from_pretrained(
                self._policy, lora_adapter_path, config=peft_config, is_trainable=False
            )
        else:
            self.get_logger().info(f"Loading {model_type} from {model_path} ...")
            self._policy = PolicyClass.from_pretrained(model_path)
        self._policy.eval()
        self._policy.to(self._device)

        # Load normalization stats from the fine-tuned checkpoint if provided,
        # otherwise fall back to the base model.
        stats_path = lora_adapter_path if lora_adapter_path else model_path
        self._preprocessor, self._postprocessor = make_pre_post_processors(
            self._policy.config,
            pretrained_path=stats_path,
        )

        self._lock = threading.Lock()
        self._action_queue: collections.deque = collections.deque()
        self._chunk_done = threading.Event()
        self._chunk_done.set()  # Ready to infer immediately on startup

        # Cached observations (None until first message received)
        self._obs_wrist: Image | None = None
        self._obs_base: Image | None = None
        self._joint_positions: list[float] | None = None  # radians, len == _NUM_JOINTS

        # Subscribers
        self.create_subscription(Image, "/wrist_camera/image_raw", self._cb_wrist, 1)
        self.create_subscription(Image, "/base_camera/image_raw", self._cb_base, 1)
        self.create_subscription(JointState, "/joint_states", self._cb_joints, 1)

        # Publisher
        self._pub = self.create_publisher(JointState, "/joint_command", 10)

        # Control timer runs on the executor thread (lightweight — just publishes)
        self.create_timer(1.0 / _CONTROL_HZ, self._control_step)

        self._paused = False

        self._reset_episode()
        self.get_logger().info("PI0InferenceNode ready.")
        self.get_logger().info("Commands: pause | resume | reset")

        # Inference and stdin run on their own threads so they don't block the executor
        self._inference_thread = threading.Thread(target=self._inference_loop, daemon=True)
        self._inference_thread.start()
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
        name_to_pos = dict(zip(msg.name, msg.position))
        try:
            positions = [name_to_pos[n] for n in _JOINT_NAMES]
        except KeyError:
            return  # Not all joints present yet
        with self._lock:
            self._joint_positions = positions

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
                self._reset_episode()
                self._paused = True
                self.get_logger().info("Reset. Type 'resume' to start inferring.")
            else:
                self.get_logger().warn(f"Unknown command: '{cmd}'. Use pause | resume | reset.")

    # ------------------------------------------------------------------
    # Episode management
    # ------------------------------------------------------------------

    def _reset_episode(self) -> None:
        self._policy.reset()
        with self._lock:
            self._action_queue.clear()
            self._obs_wrist = None
            self._obs_base = None
            self._joint_positions = None
        self._chunk_done.set()  # Unblock inference thread if it was waiting

    # ------------------------------------------------------------------
    # Inference thread (~5 Hz, runs independently of the ROS executor)
    # ------------------------------------------------------------------

    def _inference_loop(self) -> None:
        while rclpy.ok():
            self._chunk_done.wait()
            self._chunk_done.clear()
            if not self._paused:
                try:
                    self._inference_step_impl()
                except Exception as e:
                    self.get_logger().error(
                        f"Inference step failed: {type(e).__name__}: {e}", throttle_duration_sec=5.0
                    )
            # If the queue is still empty (missing obs, error, or paused), retry after a
            # short delay so the loop doesn't deadlock waiting for a control-step signal
            # that will never come.
            with self._lock:
                queue_empty = len(self._action_queue) == 0
            if queue_empty:
                time.sleep(0.2)
                self._chunk_done.set()

    def _inference_step_impl(self) -> None:
        with self._lock:
            missing = [
                name for name, val in (
                    ("wrist_camera", self._obs_wrist),
                    ("base_camera", self._obs_base),
                    ("joint_states", self._joint_positions),
                )
                if val is None
            ]
        if missing:
            self.get_logger().warn(
                f"Waiting for: {', '.join(missing)}", throttle_duration_sec=5.0
            )
            return

        with self._lock:
            wrist_t = _ros_image_to_tensor(self._obs_wrist)
            base_t = _ros_image_to_tensor(self._obs_base)
            state_t = torch.tensor(self._joint_positions, dtype=torch.float32).unsqueeze(0)  # (1, 6)

        prompt = self.get_parameter("prompt").get_parameter_value().string_value

        # LeRobot-convention observation batch.
        # Keys use dot notation: observation.state and observation.images.<cam>.
        # Images are (1, 3, H, W) uint8 CHW; the preprocessor handles resizing and normalisation.
        # State is (1, 6) float32 in radians; the preprocessor handles normalisation.
        raw_obs = {
            "observation.state": state_t,
            "observation.images.wrist": wrist_t,
            "observation.images.base": base_t,
            "task": prompt,
        }

        self.get_logger().debug("Running inference ...")
        batch = self._preprocessor(raw_obs)
        batch = {
            k: v.to(self._device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        with torch.no_grad():
            actions_raw = self._policy.predict_action_chunk(batch)
            # actions_raw: (1, chunk_size, action_dim) tensor

        self.get_logger().info(
            f"Raw actions[0]: {actions_raw[0, 0, :_NUM_JOINTS].cpu().tolist()}",
            throttle_duration_sec=2.0,
        )

        # Postprocessor denormalises actions back to joint space
        actions = self._postprocessor(actions_raw)

        if actions.dim() == 3:
            actions = actions.squeeze(0)  # → (chunk_size, action_dim)

        # Slice to the joints we control
        actions = actions[:, :_NUM_JOINTS]  # (chunk_size, 6)

        self.get_logger().info(
            f"Post actions[0]: {actions[0].cpu().tolist()}",
            throttle_duration_sec=2.0,
        )

        with self._lock:
            self._action_queue = collections.deque(actions.cpu().unbind(dim=0))

    # ------------------------------------------------------------------
    # Control timer (30 Hz)
    # ------------------------------------------------------------------

    def _control_step(self) -> None:
        if self._paused:
            return
        with self._lock:
            if not self._action_queue:
                return
            action = self._action_queue.popleft()  # (6,) tensor
            if not self._action_queue:
                self._chunk_done.set()  # Queue just emptied — trigger next inference

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
