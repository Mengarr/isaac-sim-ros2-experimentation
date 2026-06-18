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
from lerobot.configs.types import RTCAttentionSchedule
from lerobot.policies.rtc.configuration_rtc import RTCConfig

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

        # Real-Time Chunking (RTC) parameters
        self.declare_parameter("enable_rtc", False)
        self.declare_parameter("execution_horizon", 10)
        self.declare_parameter("max_guidance_weight", 10.0)

        model_type = self.get_parameter("model_type").get_parameter_value().string_value
        if model_type not in _POLICY_CLASSES:
            raise ValueError(f"model_type must be one of {list(_POLICY_CLASSES)}, got '{model_type}'")
        PolicyClass = _POLICY_CLASSES[model_type]

        _DEFAULT_MODEL_PATHS = {"pi0": "lerobot/pi0_base", "pi05": "lerobot/pi05_libero"}
        model_path = self.get_parameter("model_path").get_parameter_value().string_value
        if not model_path:
            model_path = _DEFAULT_MODEL_PATHS[model_type]

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.get_logger().info(f"Loading {model_type} from {model_path} ...")
        self._policy = PolicyClass.from_pretrained(model_path)

        # Configure RTC. Inference delay is 0 for this server (the broker requests a
        # new chunk only once the previous one has drained).
        self._rtc_enabled = self.get_parameter("enable_rtc").get_parameter_value().bool_value
        if self._rtc_enabled:
            execution_horizon = (
                self.get_parameter("execution_horizon").get_parameter_value().integer_value
            )
            max_guidance_weight = (
                self.get_parameter("max_guidance_weight").get_parameter_value().double_value
            )
            self.get_logger().info(
                f"RTC enabled (execution_horizon={execution_horizon}, "
                f"max_guidance_weight={max_guidance_weight})."
            )
            self._policy.config.rtc_config = RTCConfig(
                enabled=True,
                execution_horizon=execution_horizon,
                max_guidance_weight=max_guidance_weight,
                prefix_attention_schedule=RTCAttentionSchedule.EXP,
            )
            self._policy.init_rtc_processor()
        else:
            self.get_logger().info("RTC disabled.")

        self._policy.eval()
        self._policy.to(self._device)

        self._preprocessor, self._postprocessor = make_pre_post_processors(
            self._policy.config,
            pretrained_path=model_path,
        )

        # Service calls and stdin commands can race against each other.
        self._lock = threading.Lock()
        self._paused = False

        self._policy.reset()

        # Warm the policy up before advertising so the first real request reflects
        # steady-state latency (the first inference pays CUDA kernel compilation,
        # cuDNN autotune and allocator warmup — typically several times slower).
        self._warmup()

        self._srv = self.create_service(
            GetActionChunk, "get_action_chunk", self._handle_get_action_chunk
        )

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
    # Warmup
    # ------------------------------------------------------------------

    def _make_dummy_batch(self) -> dict:
        """Zeroed observation batch matching the real request shapes (for warmup).

        Image size is arbitrary — the preprocessor resizes to the policy's expected
        resolution — so any reasonable HxW works.
        """
        dummy_img = torch.zeros((1, 3, 224, 224), dtype=torch.uint8)
        raw_obs = {
            "observation.state": torch.zeros((1, _NUM_JOINTS), dtype=torch.float32),
            "observation.images.wrist": dummy_img,
            "observation.images.base": dummy_img.clone(),
            "task": "warmup",
        }
        batch = self._preprocessor(raw_obs)
        return {
            k: v.to(self._device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

    def _warmup(self, iterations: int = 3) -> None:
        self.get_logger().info(f"Warming up policy ({iterations} iterations) ...")
        batch = self._make_dummy_batch()
        with torch.no_grad():
            for _ in range(iterations):
                # No RTC kwargs → prev_chunk_left_over is None → no guidance applied,
                # but the model's denoising/forward kernels still compile and warm.
                self._policy.predict_action_chunk(batch)
        if self._device.type == "cuda":
            torch.cuda.synchronize()
        self._policy.reset()  # discard any state the warmup left behind
        self.get_logger().info("Warmup complete.")

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
            response.status = GetActionChunk.Response.STATUS_PAUSED
            return response

        name_to_pos = dict(zip(request.joint_state.name, request.joint_state.position))
        try:
            joint_positions = [name_to_pos[n] for n in _JOINT_NAMES]
        except KeyError:
            self.get_logger().error("Request joint_state is missing required joints.")
            response.status = GetActionChunk.Response.STATUS_ERROR
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

        # Unpack the RTC leftover sent by the broker. It is the unexecuted tail of the
        # previous chunk in raw/normalised action space, time-aligned with the new chunk.
        prev_chunk_left_over = None
        if (
            self._rtc_enabled
            and len(request.prev_chunk_left_over) > 0
            and request.prev_chunk_action_dim > 0
        ):
            adim = int(request.prev_chunk_action_dim)
            leftover_np = np.asarray(request.prev_chunk_left_over, dtype=np.float32).reshape(-1, adim)
            prev_chunk_left_over = torch.from_numpy(leftover_np).to(self._device)

        self.get_logger().debug("Running inference ...")
        with self._lock:
            batch = self._preprocessor(raw_obs)
            batch = {
                k: v.to(self._device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            with torch.no_grad():
                if self._rtc_enabled:
                    # Blend the new chunk with the unexecuted tail of the previous chunk.
                    # inference_delay (predicted by the broker) positions the prefix
                    # weights: the first inference_delay steps are frozen to what the
                    # robot already committed to during the round trip.
                    actions_raw = self._policy.predict_action_chunk(
                        batch,
                        inference_delay=int(request.inference_delay),
                        prev_chunk_left_over=prev_chunk_left_over,
                    )
                else:
                    actions_raw = self._policy.predict_action_chunk(batch)
                # actions_raw: (1, chunk_size, action_dim) tensor

            self.get_logger().info(
                f"Raw actions[0]: {actions_raw[0, 0, :_NUM_JOINTS].cpu().tolist()}",
                throttle_duration_sec=2.0,
            )

            # Postprocessor denormalises actions back to joint space
            actions = self._postprocessor(actions_raw)

        # Raw chunk (full action_dim, normalised) → returned so the broker can compute
        # the leftover for the next RTC request.
        if self._rtc_enabled:
            raw_np = actions_raw.squeeze(0).detach().cpu().numpy().astype(np.float32)  # (chunk, action_dim)
            response.raw_chunk = raw_np.reshape(-1).tolist()
            response.raw_chunk_action_dim = int(raw_np.shape[1])

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

        response.status = GetActionChunk.Response.STATUS_OK
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
