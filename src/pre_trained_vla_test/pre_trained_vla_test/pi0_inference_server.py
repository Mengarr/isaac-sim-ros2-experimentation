"""
pi0_inference_server.py
------------------------
ROS2 node that runs PI0Policy as a service server for an SO-101 arm in Isaac Sim.

A local broker node (pi0_inference_broker.py) acts as the client: it calls the
/get_action_chunk service, sending the latest wrist/base images (JPEG-compressed)
and joint state in the request, and receives a chunk of actions in the response.
No topics are subscribed to or published on a timer here — images only cross the
network when the broker actually requests an inference step.

Provides:
  /get_action_chunk   (pre_trained_vla_test_interfaces/srv/GetActionChunk)

Requirements:
  - ROS2 sourced
  - lerobot venv sourced
  - CUDA GPU available
"""

import sys
import threading

import cv2
import numpy as np
import rclpy
import torch
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, JointState

from pre_trained_vla_test_interfaces.srv import GetActionChunk

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


def _compressed_image_to_tensor(msg: CompressedImage) -> torch.Tensor:
    """Decode sensor_msgs/CompressedImage (JPEG) → (1, 3, H, W) uint8 CPU tensor (CHW, RGB)."""
    data = np.frombuffer(msg.data, dtype=np.uint8)
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)


class PI0InferenceServerNode(Node):
    def __init__(self):
        super().__init__("pi0_inference_server")

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

        # Service calls and stdin commands can race against each other.
        self._lock = threading.Lock()
        self._paused = False

        self._srv = self.create_service(
            GetActionChunk, "get_action_chunk", self._handle_get_action_chunk
        )

        self._policy.reset()
        self.get_logger().info("PI0InferenceServerNode ready, serving /get_action_chunk.")
        self.get_logger().info("Commands: pause | resume | reset")

        self._input_thread = threading.Thread(target=self._stdin_listener, daemon=True)
        self._input_thread.start()

    # ------------------------------------------------------------------
    # Terminal command listener
    # ------------------------------------------------------------------

    def _stdin_listener(self) -> None:
        for line in sys.stdin:
            cmd = line.strip().lower()
            if cmd in ("pause", "p"):
                with self._lock:
                    self._paused = True
                self.get_logger().info("Paused.")
            elif cmd in ("resume", "r"):
                with self._lock:
                    self._paused = False
                self.get_logger().info("Resumed.")
            elif cmd in ("reset", "rst"):
                with self._lock:
                    self._policy.reset()
                    self._paused = True
                self.get_logger().info("Reset. Type 'resume' to start serving inference.")
            else:
                self.get_logger().warn(f"Unknown command: '{cmd}'. Use pause | resume | reset.")

    # ------------------------------------------------------------------
    # Service callback — runs synchronously on the executor thread
    # ------------------------------------------------------------------

    def _handle_get_action_chunk(
        self, request: GetActionChunk.Request, response: GetActionChunk.Response
    ) -> GetActionChunk.Response:
        with self._lock:
            paused = self._paused
        if paused:
            self.get_logger().warn("Inference paused — returning empty action chunk.", throttle_duration_sec=5.0)
            return response

        name_to_pos = dict(zip(request.joint_state.name, request.joint_state.position))
        try:
            joint_positions = [name_to_pos[n] for n in _JOINT_NAMES]
        except KeyError:
            self.get_logger().error("Request joint_state is missing required joints.")
            return response

        wrist_t = _compressed_image_to_tensor(request.wrist_image)
        base_t = _compressed_image_to_tensor(request.base_image)
        state_t = torch.tensor(joint_positions, dtype=torch.float32).unsqueeze(0)  # (1, 6)

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
        with self._lock:
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

        stamp = self.get_clock().now().to_msg()
        for action in actions.cpu().unbind(dim=0):
            joint_state = JointState()
            joint_state.header.stamp = stamp
            joint_state.name = _JOINT_NAMES
            joint_state.position = action.tolist()
            response.action_chunk.append(joint_state)

        return response


def main() -> None:
    rclpy.init()
    node = PI0InferenceServerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
