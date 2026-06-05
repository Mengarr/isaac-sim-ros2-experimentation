# pre_trained_vla_test

Runs `lerobot/pi05_libero` (PI0Policy) against an SO-101 arm simulated in Isaac Sim. Camera frames and joint states are consumed over ROS2; joint commands are published back to the sim.

## Prerequisites

- Isaac Sim running with the SO-101 camera action graph publishing:
  - `/wrist_camera/image_raw`
  - `/base_camera/image_raw`
  - `/joint_states`
- ROS2 Jazzy installed
- LeRobot venv (includes torch) installed
- A GPU available for inference

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

## Usage

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

## Topics

| Direction | Topic | Type |
|-----------|-------|------|
| Subscribe | `/wrist_camera/image_raw` | `sensor_msgs/Image` |
| Subscribe | `/base_camera/image_raw` | `sensor_msgs/Image` |
| Subscribe | `/joint_states` | `sensor_msgs/JointState` |
| Publish | `/joint_command` | `sensor_msgs/JointState` |

## Notes

- Inference runs at ~5 Hz; joint commands are published at 30 Hz from the action chunk (matching the dataset recording fps)
- The node waits silently until all four subscribed topics have published at least one message before running inference
- Call `policy.reset()` between task episodes — currently this happens automatically on node startup; add a ROS2 service call here if you need mid-session resets
- Camera key names (`wrist_camera`, `base_camera`) may need to be aligned with the keys `lerobot/pi05_base` was trained on — check the model card if inference errors on unrecognised observation keys


scp -r -i ~/.ssh/aws_ec2_ap.pem  ubuntu@43.220.1.89:/home/ubuntu/data/datasets/so101_cube_task_v2/ ~/data/datasets/