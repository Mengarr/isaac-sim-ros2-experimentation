"""
probe_marginal.py
-----------------
ROS2 node that probes the *marginal action distribution* of PI0/PI05 at a single,
frozen observation — the "sample the marginal" experiment.

It mirrors the service-server structure of pi0_inference_server.py: it advertises
/get_action_chunk and consumes the wrist/base images + joint state the broker
sends. Unlike the real server it does NOT drive the robot — it replies with
STATUS_PAUSED so the broker holds position, which keeps the scene (and therefore
the observation) frozen while we probe.

What it does on each probe:
  1. Takes the broker's latest frame as ONE pre-grasp observation.
  2. Tiles that single observation across the batch dimension to N rows and runs
     a single predict_action_chunk — because PI0 samples i.i.d. flow-matching
     noise per batch row, those N rows are N independent draws from
     p(action_chunk | observation). (Same trick as N sequential seeded calls,
     but one forward pass.)
  3. Plots, with matplotlib:
       - a histogram of the gripper command at the FIRST timestep of the chunk,
       - a spaghetti/fan plot of the gripper command over the whole chunk
         (gripper command on y, timestep-within-chunk on x), one line per sample.
  4. Shows the base + wrist camera views with cv2.

If the gripper marginal is bimodal (some samples close, some wait/open), the
model represents the grasp as multimodal at that observation.

Usage (N is the number of samples / batch size):
  ros2 run pre_trained_vla_test probe_marginal --ros-args -p num_samples:=50

Requirements:
  - ROS2 sourced
  - lerobot venv sourced
  - CUDA GPU available (N-way batch is heavier than a single inference)
"""

import sys
import threading
import time

import cv2
import matplotlib

matplotlib.use("TkAgg")  # interactive backend; falls back below if unavailable
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import rclpy  # noqa: E402
import torch  # noqa: E402
from rclpy.node import Node  # noqa: E402

from pre_trained_vla_test_interfaces.srv import GetActionChunk  # noqa: E402

from lerobot.configs.policies import PreTrainedConfig  # noqa: E402
from lerobot.policies import make_pre_post_processors  # noqa: E402
from lerobot.policies.pi0 import PI0Policy  # noqa: E402
from lerobot.policies.pi05 import PI05Policy  # noqa: E402

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
_GRIPPER_INDEX = _JOINT_NAMES.index("gripper")  # last joint we control


def _decode_compressed(msg) -> np.ndarray:
    """sensor_msgs/CompressedImage (JPEG) -> BGR uint8 HxWx3 numpy array."""
    data = np.frombuffer(msg.data, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)  # BGR


def _bgr_to_chw_tensor(bgr: np.ndarray) -> torch.Tensor:
    """BGR HxWx3 uint8 -> (1, 3, H, W) uint8 CPU tensor in RGB (model convention)."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)


class ProbeMarginalNode(Node):
    def __init__(self):
        super().__init__("probe_marginal")

        self.declare_parameter("model_type", "pi05")  # "pi0" or "pi05"
        self.declare_parameter("model_path", "")
        self.declare_parameter("prompt", "pick up the object")

        # Number of marginal samples == batch size of the single forward pass.
        self.declare_parameter("num_samples", 50)

        # Don't re-probe on every broker request (they arrive as fast as we reply).
        # Re-run the N-sample sweep at most once per this many seconds.
        self.declare_parameter("probe_interval_sec", 2.0)

        # torch.compile (max-autotune) is baked into some checkpoint configs; it
        # fights the variable batch size we use here, so disable it by default.
        self.declare_parameter("disable_compile", True)

        model_type = self.get_parameter("model_type").get_parameter_value().string_value
        if model_type not in _POLICY_CLASSES:
            raise ValueError(f"model_type must be one of {list(_POLICY_CLASSES)}, got '{model_type}'")
        PolicyClass = _POLICY_CLASSES[model_type]

        _DEFAULT_MODEL_PATHS = {"pi0": "lerobot/pi0_base", "pi05": "lerobot/pi05_libero"}
        model_path = self.get_parameter("model_path").get_parameter_value().string_value
        if not model_path:
            model_path = _DEFAULT_MODEL_PATHS[model_type]

        self._n = int(self.get_parameter("num_samples").get_parameter_value().integer_value)
        self._probe_interval = (
            self.get_parameter("probe_interval_sec").get_parameter_value().double_value
        )

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.get_logger().info(
            f"Loading {model_type} from {model_path} (N={self._n} marginal samples) ..."
        )

        config = PreTrainedConfig.from_pretrained(model_path)
        disable_compile = self.get_parameter("disable_compile").get_parameter_value().bool_value
        if disable_compile and getattr(config, "compile_model", False):
            self.get_logger().warn("Disabling torch.compile (incompatible with N-way batching).")
            config.compile_model = False
        # The marginal must be the model's raw conditional — RTC would couple the
        # chunk to a previous one. Ensure it is off.
        if getattr(config, "rtc_config", None) is not None:
            config.rtc_config = None

        self._policy = PolicyClass.from_pretrained(model_path, config=config)
        self._policy.eval()
        self._policy.to(self._device)

        self._preprocessor, self._postprocessor = make_pre_post_processors(
            self._policy.config,
            pretrained_path=model_path,
        )

        self._lock = threading.Lock()
        self._last_probe_t = 0.0

        # Matplotlib figure (created once, updated in place). cv2 windows are created
        # lazily on first frame. All GUI work happens on the executor/main thread
        # because rclpy.spin runs the service callback there.
        plt.ion()
        self._fig, (self._ax_hist, self._ax_fan) = plt.subplots(1, 2, figsize=(12, 5))
        self._fig.canvas.manager.set_window_title("PI0 gripper marginal probe")

        self._srv = self.create_service(
            GetActionChunk, "get_action_chunk", self._handle_get_action_chunk
        )
        self.get_logger().info(
            "ProbeMarginalNode ready, serving /get_action_chunk. "
            "Replies STATUS_PAUSED so the broker holds the arm still."
        )

    # ------------------------------------------------------------------
    # Service callback — runs on the executor (main) thread
    # ------------------------------------------------------------------

    def _handle_get_action_chunk(
        self, request: GetActionChunk.Request, response: GetActionChunk.Response
    ) -> GetActionChunk.Response:
        # Always reply PAUSED: we never command the robot, we just borrow its frames.
        response.status = GetActionChunk.Response.STATUS_PAUSED

        now = time.perf_counter()
        if now - self._last_probe_t < self._probe_interval:
            return response
        self._last_probe_t = now

        try:
            self._probe(request)
        except Exception as e:  # keep serving even if one probe blows up
            self.get_logger().error(f"Probe failed: {type(e).__name__}: {e}")

        return response

    # ------------------------------------------------------------------
    # The probe
    # ------------------------------------------------------------------

    def _probe(self, request: GetActionChunk.Request) -> None:
        # --- decode the one frozen observation ---
        name_to_pos = dict(zip(request.joint_state.name, request.joint_state.position))
        try:
            joint_positions = [name_to_pos[n] for n in _JOINT_NAMES]
        except KeyError:
            self.get_logger().error("Request joint_state is missing required joints.")
            return

        wrist_bgr = _decode_compressed(request.wrist_image)
        base_bgr = _decode_compressed(request.base_image)
        self._show_cameras(base_bgr, wrist_bgr)

        prompt = self.get_parameter("prompt").get_parameter_value().string_value
        raw_obs = {
            "observation.state": torch.tensor(joint_positions, dtype=torch.float32).unsqueeze(0),
            "observation.images.wrist": _bgr_to_chw_tensor(wrist_bgr),
            "observation.images.base": _bgr_to_chw_tensor(base_bgr),
            "task": prompt,
        }

        # --- preprocess (B=1), then tile to N rows ---
        batch = self._preprocessor(raw_obs)
        batch = {
            k: (v.to(self._device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        batch = self._tile_batch(batch, self._n)

        # --- one forward pass = N i.i.d. draws of the action chunk ---
        t0 = time.perf_counter()
        with torch.no_grad():
            actions_raw = self._policy.predict_action_chunk(batch)  # (N, chunk, action_dim)
            actions = self._postprocessor(actions_raw)  # denormalised to joint space
        if self._device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0

        actions = actions[:, :, :_NUM_JOINTS]  # (N, chunk, 6)
        grip = actions[..., _GRIPPER_INDEX].float().cpu().numpy()  # (N, chunk)
        self.get_logger().info(
            f"Probed N={grip.shape[0]} samples, chunk={grip.shape[1]} in {dt:.2f}s "
            f"(gripper step0: mean={grip[:, 0].mean():.4f} std={grip[:, 0].std():.4f})"
        )

        self._plot(grip)

    def _tile_batch(self, batch: dict, n: int) -> dict:
        """Repeat every (1, ...) tensor in the batch to (n, ...). Non-tensors pass through."""
        out = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor) and v.shape[0] == 1:
                out[k] = v.expand(n, *v.shape[1:]).contiguous()
            else:
                out[k] = v
        return out

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def _plot(self, grip: np.ndarray) -> None:
        n, chunk = grip.shape
        ts = np.arange(chunk)

        # Histogram of the gripper command at the first timestep.
        self._ax_hist.clear()
        self._ax_hist.hist(grip[:, 0], bins=min(40, max(10, n // 2)), color="#4c72b0",
                           edgecolor="white")
        self._ax_hist.set_title(f"Gripper command @ step 0  (N={n})")
        self._ax_hist.set_xlabel("gripper command")
        self._ax_hist.set_ylabel("count")

        # Spaghetti / fan plot over the whole chunk.
        self._ax_fan.clear()
        for i in range(n):
            self._ax_fan.plot(ts, grip[i], color="#4c72b0", alpha=min(0.3, 5.0 / n), lw=0.8)
        self._ax_fan.plot(ts, grip.mean(axis=0), color="#c44e52", lw=2.0, label="mean")
        self._ax_fan.set_title("Gripper command across the chunk")
        self._ax_fan.set_xlabel("timestep within chunk")
        self._ax_fan.set_ylabel("gripper command")
        self._ax_fan.legend(loc="best")

        self._fig.tight_layout()
        self._fig.canvas.draw_idle()
        plt.pause(0.001)  # let the GUI event loop run

    def _show_cameras(self, base_bgr: np.ndarray, wrist_bgr: np.ndarray) -> None:
        if base_bgr is not None:
            cv2.imshow("base camera", base_bgr)
        if wrist_bgr is not None:
            cv2.imshow("wrist camera", wrist_bgr)
        cv2.waitKey(1)


def main() -> None:
    rclpy.init()
    node = ProbeMarginalNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        plt.close("all")
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
