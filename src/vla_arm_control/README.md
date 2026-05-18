# vla_arm_control

ROS2 Jazzy C++ package implementing a VLA-to-robot-arm control pipeline for Isaac Sim.

Two Vision-Language-Action model backends (PI-0.5, OpenVLA-OFT) publish action chunks that are consumed by a temporal-ensemble smoother, which publishes joint commands at 50 Hz. Model inference calls are stubs — the pipeline is functional end-to-end without a real model.

## Architecture

```
/observations/rgb_base        ┐
/observations/rgb_wrist       ├─► Pi0Node / OpenVlaOftNode ──► /action_chunk ──► SmootherNode ──► /joint_commands
/observations/joint_states    ┘         (5 Hz, inference thread)                  (50 Hz)
/observations/language_instruction ┘
```

### Nodes

| Node | Executable | Purpose |
|---|---|---|
| `SmootherNode` | `smoother_node` | Buffers action chunks, interpolates at wall-clock `now`, applies exponential temporal ensemble weighting, publishes a single-point command at 50 Hz |
| `Pi0Node` | `pi0_node` | PI-0.5 backend — absolute joint angle output, 50-step / 1 s horizon, uses base + wrist cameras |
| `OpenVlaOftNode` | `openvla_oft_node` | OpenVLA-OFT backend — delta action output accumulated to absolute positions, 10-step / 0.2 s horizon, base camera only |

`VlaBaseNode` (`include/vla_arm_control/vla_base_node.hpp`) is a header-only abstract base that manages observation subscriptions, the inference thread, and chunk publication. It is not built as a standalone executable.

### Timestamp convention

`/action_chunk` carries `header.stamp` equal to the **observation image timestamp**, not the publish time. `time_from_start` on each trajectory point is relative to that stamp, so the smoother can compute the absolute wall-clock moment of each planned action:

```
abs_time[i] = chunk.header.stamp + points[i].time_from_start
```

This matters because VLA inference latency (100–200 ms at 5 Hz) means chunks arrive after the observations they were computed from. Using observation time lets the smoother correctly interpolate even under variable latency.

## Build

```bash
cd /path/to/ros2_ws
colcon build --packages-select vla_arm_control
source install/setup.bash
```

Dependencies: `rclcpp`, `trajectory_msgs`, `sensor_msgs`, `std_msgs` (all standard ROS2).

## Run

**Smoke test (stubs only):**
```bash
# Terminal 1
ros2 run vla_arm_control smoother_node

# Terminal 2
ros2 run vla_arm_control pi0_node

# Terminal 3 — should see 50 Hz single-point JointTrajectory
ros2 topic echo /joint_commands
```

**Launch with named joints:**
```bash
ros2 launch vla_arm_control pi0.launch.py \
  joint_names:=joint_1,joint_2,joint_3,joint_4,joint_5,joint_6,gripper

ros2 launch vla_arm_control openvla_oft.launch.py \
  joint_names:=joint_1,joint_2,joint_3,joint_4,joint_5,joint_6,gripper
```

`joint_names` can also be set directly in the yaml configs under `config/`.

## Topics

| Topic | Type | Direction |
|---|---|---|
| `/observations/rgb_base` | `sensor_msgs/Image` | VLA node subscribes |
| `/observations/rgb_wrist` | `sensor_msgs/Image` | VLA node subscribes (PI-0.5 only) |
| `/observations/joint_states` | `sensor_msgs/JointState` | VLA node subscribes |
| `/observations/language_instruction` | `std_msgs/String` | VLA node subscribes |
| `/action_chunk` | `trajectory_msgs/JointTrajectory` | VLA node publishes → smoother subscribes |
| `/joint_commands` | `trajectory_msgs/JointTrajectory` | Smoother publishes (single point, 50 Hz) |

## Key Parameters

**smoother_node** (`config/smoother.yaml`):

| Parameter | Default | Description |
|---|---|---|
| `execution_frequency_hz` | `50.0` | Command publish rate |
| `max_chunk_buffer_size` | `3` | Max buffered action chunks |
| `ensemble_lambda` | `2.0` | Exponential decay rate for chunk weighting |
| `joint_names` | `[]` | Must match Isaac Sim joint names |

**pi0_node** (`config/pi0.yaml`) / **openvla_oft_node** (`config/openvla_oft.yaml`):

| Parameter | PI-0.5 default | OFT default | Description |
|---|---|---|---|
| `inference_frequency_hz` | `5.0` | `5.0` | Inference calls per second |
| `use_wrist_camera` | `true` | `false` | Subscribe to `/observations/rgb_wrist` |
| `chunk_length` | `50` | `10` | Trajectory points per chunk |
| `action_chunk_dt` | `0.02` | `0.02` | Seconds between points |
| `action_scale` | — | `1.0` | Delta multiplier (OFT only) |

## What Needs to Be Done

### 1. Implement model inference (per backend)

In each VLA node, replace the stub body in `runInference()` with the actual model call:

- **`src/pi0_node.cpp`** — Load the PI-0.5 checkpoint and call the flow-matching forward pass. The function receives raw `sensor_msgs::Image` pointers; resize/convert to `(image_width_, image_height_)` RGB before passing to the model. Return a `JointTrajectory` with `chunk_length_` absolute joint-angle points. Do **not** set `header.stamp` — the base class sets it.

- **`src/openvla_oft_node.cpp`** — Load the OpenVLA-OFT checkpoint and run the parallel-decoding forward pass. Store the raw delta output in trajectory point positions — `adaptOutput()` will accumulate them. Populate `joint_state_snapshot_` before the model call (already done in the stub) so `adaptOutput()` has the correct base positions.

### 2. Image preprocessing

`sensor_msgs::Image` messages are stored as-is (no cv_bridge dependency). Add image conversion inside `runInference()` before the model call: decode the encoding in `msg->encoding`, resize to model input dimensions, and normalise to the range the model expects.

### 3. Set joint names

Populate `joint_names` in `config/smoother.yaml`, `config/pi0.yaml`, and `config/openvla_oft.yaml` to match the joint names reported by Isaac Sim. All three must be identical.

### 4. Wire up Isaac Sim topics

Confirm that Isaac Sim publishes on the expected topic names and message types:
- `/observations/rgb_base` and `/observations/rgb_wrist` as `sensor_msgs/Image`
- `/observations/joint_states` as `sensor_msgs/JointState`
- `/joint_commands` consumed by the Isaac Sim joint controller as `trajectory_msgs/JointTrajectory`

Use remapping in the launch files if the Isaac Sim topic names differ.

### 5. Tune smoother parameters

Once inference is running, tune `ensemble_lambda` and `max_chunk_buffer_size` to balance latency and smoothness. Higher `ensemble_lambda` makes the smoother track the newest chunk more aggressively; lower values blend chunks more gradually.
