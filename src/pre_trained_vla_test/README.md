# pre_trained_vla_test

Runs `lerobot/pi05_libero` (PI0Policy) against an SO-101 arm simulated in Isaac Sim. Camera frames and joint states are consumed over ROS2; joint commands are published back to the sim.

Two ways to run inference:

- **Single-node (`pi0_inference`)** — everything (ROS2 + policy + GPU) runs on the same machine as Isaac Sim.
- **Broker/server (`pi0_inference_broker` + `pi0_inference_server`)** — splits inference onto a remote GPU machine while Isaac Sim and ROS2 stay local. See [Broker/server (remote GPU) setup](#brokerserver-remote-gpu-setup) below.

## Prerequisites

- Isaac Sim running with the SO-101 camera action graph publishing:
  - `/wrist_camera/image_raw`
  - `/base_camera/image_raw`
  - `/joint_states`
- ROS2 Jazzy installed
- LeRobot venv (includes torch) installed
- A GPU available for inference (locally for the single-node setup, or on a remote machine for the broker/server setup)

## HuggingFace Access

PI0Policy uses PaliGemma as its vision backbone, which is a **gated model** requiring license acceptance.

1. Accept the license at [huggingface.co/google/paligemma-3b-pt-224](https://huggingface.co/google/paligemma-3b-pt-224)
2. Generate an access token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
3. Export it in your shell (add to `~/.bashrc` to make permanent):

```bash
export HF_TOKEN=hf_...
```

Without this, the node will crash with a `403 Forbidden` error when loading the model.

## Quantising pi0_base to FP16

`lerobot/pi0_base` ships as FP32 (~13 GiB), which exceeds the VRAM of inference-class GPUs like the Tesla T4. Run the quantisation script once on any machine with >14 GiB RAM (no GPU required) to produce a FP16 checkpoint (~6.5 GiB):

```bash
python src/pre_trained_vla_test/pre_trained_vla_test/quantize_pi0.py
```

This saves a complete checkpoint to `~/checkpoints/pi0_base_fp16/`. The inference node is already configured to load from this path. To use a different output location:

```bash
python src/pre_trained_vla_test/pre_trained_vla_test/quantize_pi0.py --output /path/to/pi0_base_fp16
```

**You must update the model quantization** to `bfloat16` in the `/path/to/pi0_base_fp16/config.json` from `float32`, `float16` is **not** a supported option here.  

Then update `_MODEL_ID` in `pi0_inference.py` to match.

## Usage (single-node)

### 1. Source ROS2

```bash
source /opt/ros/jazzy/setup.bash
```

### 2. Activate the LeRobot venv

```bash
source ~/lerobot/.venv/bin/activate   # adjust path to your venv
```

### 3. Build and source the package

```bash
cd ~/repos/isaac-sim-ros2-experimentation
colcon build --packages-select pre_trained_vla_test_interfaces
colcon build --packages-select pre_trained_vla_test
source install/setup.bash
```

### 4. Run the inference node

Run directly with the venv's Python rather than `ros2 run` — `ros2 run` uses the system interpreter which doesn't have the lerobot dependencies:

```bash
python install/pre_trained_vla_test/lib/pre_trained_vla_test/pi0_inference
```

The node will download `lerobot/pi0_base` from the Hub on first run (requires internet access), then begin publishing to `/joint_command` once all camera and joint state topics are live.

### Overriding the task prompt

```bash
python install/pre_trained_vla_test/lib/pre_trained_vla_test/pi0_inference --ros-args -p prompt:="place the cup on the plate"
```

### Using delta actions

If your VLA outputs per-step joint deltas instead of absolute positions, pass `delta_actions:=true`. Each chunk is integrated from the joint state captured at inference time, so the whole chunk is self-consistent regardless of feedback latency.

```bash
python install/pre_trained_vla_test/lib/pre_trained_vla_test/pi0_inference \
  --ros-args -p delta_actions:=true
```

Default is `false` (absolute joint positions).

### Using a LoRA fine-tuned model

Two workflows are supported depending on how your adapter was saved:

**LeRobot training checkpoint** (`lerobot-train --peft.method_type=LORA`): the checkpoint directory already contains the merged weights. Just point `_MODEL_ID` at the checkpoint and omit `lora_adapter_path`.

**Raw PEFT adapter directory** (contains `adapter_config.json` + `adapter_model.safetensors`): keep `_MODEL_ID` pointing at the base model and pass the adapter directory as a ROS param. The adapter is merged into the base weights at startup so there is no inference overhead.

```bash
python install/pre_trained_vla_test/lib/pre_trained_vla_test/pi0_inference \
  --ros-args \
  -p prompt:="place the cup on the plate" \
  -p lora_adapter_path:="/home/ubuntu/checkpoints/my_lora_adapter"
```

## Topics (single-node)

| Direction | Topic | Type |
|-----------|-------|------|
| Subscribe | `/wrist_camera/image_raw` | `sensor_msgs/Image` |
| Subscribe | `/base_camera/image_raw` | `sensor_msgs/Image` |
| Subscribe | `/joint_states` | `sensor_msgs/JointState` |
| Publish | `/joint_command` | `sensor_msgs/JointState` |

## Broker/server (remote GPU) setup

If the machine running Isaac Sim doesn't have a GPU (or you want to keep the sim machine free of the lerobot/torch stack), split inference across two nodes connected over the network:

- **`pi0_inference_server`** — runs on the remote GPU machine. Loads the policy once and serves the `pre_trained_vla_test_interfaces/srv/GetActionChunk` service. Stateless with respect to ROS topics — it never subscribes or publishes anything itself.
- **`pi0_inference_broker`** — runs locally alongside Isaac Sim. Subscribes to the camera/joint-state topics exactly like the single-node setup, but instead of running inference itself, JPEG-encodes the latest frames and calls `/get_action_chunk` on the remote server whenever its local action queue runs dry (or, with RTC, after executing `execution_horizon` actions). Republishes the returned chunk to `/joint_command` at `30 Hz`, same as the single-node node.

Both nodes need ROS2 with the same `ROS_DOMAIN_ID` (or otherwise routable DDS discovery) so the broker can reach the server's service over the network. Images are JPEG-compressed before crossing the network to keep bandwidth low.

### 1. On the remote GPU machine — start the server

```bash
source /opt/ros/jazzy/setup.bash
source ~/lerobot/.venv/bin/activate
cd ~/repos/isaac-sim-ros2-experimentation
colcon build --packages-select pre_trained_vla_test_interfaces pre_trained_vla_test
source install/setup.bash

python install/pre_trained_vla_test/lib/pre_trained_vla_test/pi0_inference_server \
  --ros-args \
  -p model_type:=pi05 \
  -p prompt:="place the cup on the plate"
```

`model_type` is `"pi0"` or `"pi05"` (default `"pi05"`). Pass `model_path` to override the default checkpoint.

On startup the server warms the policy up with a few throwaway inferences before advertising the service, so the first real request reflects steady-state latency rather than the cold-start spike (CUDA kernel compilation, cuDNN autotune, allocator warmup).

While running, the server's terminal accepts the same `pause` / `resume` / `reset` commands as the single-node node. While paused or resetting it returns empty chunks (`status=PAUSED`); the broker holds its queue and resumes cleanly.

#### Real-Time Chunking (RTC)

RTC smooths the seam between consecutive action chunks by treating chunk generation as an inpainting problem: the new chunk is blended with the unexecuted tail of the previous one during the flow-matching denoising loop.

This setup drives RTC **synchronously**. The broker executes a fixed number of actions (`execution_horizon`) out of each chunk, then stops — holding the last commanded position — and requests the next chunk, passing the unexecuted tail as the leftover. Because the robot is stationary while inference runs, the server sees zero inference delay and the request is built from the freshest possible observations. Motion is no longer continuous, but accuracy is maximised (latest frame + perfect RTC continuity at the seam).

Enable it on the server and tune the two knobs as ROS params:

```bash
python install/pre_trained_vla_test/lib/pre_trained_vla_test/pi0_inference_server \
  --ros-args \
  -p model_type:=pi05 \
  -p prompt:="place the cup on the plate" \
  -p enable_rtc:=true \
  -p blend_horizon:=10 \
  -p max_guidance_weight:=10.0
```

| Param | Default | Meaning |
|-------|---------|---------|
| `enable_rtc` | `false` | Toggle RTC guidance on the server. |
| `blend_horizon` | `10` | How many steps of the new chunk are blended toward the previous chunk's leftover (the RTC guidance window). Independent of the broker's `execution_horizon`. |
| `max_guidance_weight` | `10.0` | Upper bound on how strongly consistency with the previous chunk is enforced. |

`inference_delay` is always `0` in this synchronous scheme (nothing is consumed during the round trip), so it is not a tunable param. `enable_rtc` must be set on **both** the server (for guidance) and the broker (for the queue/leftover bookkeeping).

### 2. On the machine running Isaac Sim — start the broker

```bash
source /opt/ros/jazzy/setup.bash
cd ~/repos/isaac-sim-ros2-experimentation
colcon build --packages-select pre_trained_vla_test_interfaces pre_trained_vla_test
source install/setup.bash

python install/pre_trained_vla_test/lib/pre_trained_vla_test/pi0_inference_broker \
  --ros-args \
  -p jpeg_quality:=80 \
  -p pre_request_delay_sec:=2.0
```

The broker doesn't need the lerobot venv — it has no torch/policy dependency (the RTC queue is a small numpy implementation), so `ros2 run` works fine here (unlike the inference nodes). It accepts the same `pause` / `resume` / `reset` commands in its own terminal.

To run with RTC, enable it on the broker too:

```bash
python install/pre_trained_vla_test/lib/pre_trained_vla_test/pi0_inference_broker \
  --ros-args \
  -p jpeg_quality:=80 \
  -p enable_rtc:=true \
  -p execution_horizon:=10
```

| Param | Default | Meaning |
|-------|---------|---------|
| `enable_rtc` | `false` | When true, the broker executes `execution_horizon` actions per chunk, then stops and re-plans with the unexecuted tail as the RTC leftover. When false, uses the legacy drain-the-whole-chunk-then-request behaviour. |
| `execution_horizon` | `10` | How many actions of each chunk to execute before stopping and requesting the next one. The leftover sent with each request is `chunk_len - execution_horizon`. Independent of the server's `blend_horizon`. |

With RTC enabled the broker executes exactly `execution_horizon` actions, holds the last commanded position, and requests the next chunk with `inference_delay=0` and the fixed unexecuted tail (`original[execution_horizon:]`) as `prev_chunk_left_over`. If a chunk comes back empty (server paused/resetting) the broker never merges it and holds position until chunks resume.

### Topics & service (broker/server)

| Node | Direction | Name | Type |
|------|-----------|------|------|
| Broker | Subscribe (local) | `/wrist_camera/image_raw` | `sensor_msgs/Image` |
| Broker | Subscribe (local) | `/base_camera/image_raw` | `sensor_msgs/Image` |
| Broker | Subscribe (local) | `/joint_states` | `sensor_msgs/JointState` |
| Broker | Publish (local) | `/joint_command` | `sensor_msgs/JointState` |
| Broker → Server | Call (remote) | `/get_action_chunk` | `pre_trained_vla_test_interfaces/srv/GetActionChunk` |

## Notes

- Inference runs at ~5 Hz; joint commands are published at 30 Hz from the action chunk (matching the dataset recording fps)
- The node waits silently until all four subscribed topics have published at least one message before running inference
- Call `policy.reset()` between task episodes — currently this happens automatically on node startup; add a ROS2 service call here if you need mid-session resets
- Camera key names (`wrist_camera`, `base_camera`) may need to be aligned with the keys `lerobot/pi05_base` was trained on — check the model card if inference errors on unrecognised observation keys


scp -r -i ~/.ssh/aws_ec2_ap.pem  ubuntu@43.220.1.89:/home/ubuntu/data/datasets/so101_cube_task_v2/ ~/data/datasets/