# pre_trained_vla_test

Runs `lerobot/pi0_base` (PI0Policy) against an SO-101 arm simulated in Isaac Sim. Camera frames and joint states are consumed over ROS2; joint commands are published back to the sim.

## Prerequisites

- Isaac Sim running with the SO-101 scene and camera action graph publishing:
  - `/wrist_camera/image_raw`
  - `/scene_camera/image_raw`
  - `/base_camera/image_raw`
  - `/joint_states`
- ROS2 Jazzy installed
- LeRobot venv (includes torch) installed
- A GPU available for inference

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

## Topics

| Direction | Topic | Type |
|-----------|-------|------|
| Subscribe | `/wrist_camera/image_raw` | `sensor_msgs/Image` |
| Subscribe | `/scene_camera/image_raw` | `sensor_msgs/Image` |
| Subscribe | `/base_camera/image_raw` | `sensor_msgs/Image` |
| Subscribe | `/joint_states` | `sensor_msgs/JointState` |
| Publish | `/joint_command` | `sensor_msgs/JointState` |

## Notes

- Inference runs at ~5 Hz; joint commands are published at 50 Hz from the action chunk
- The node waits silently until all four subscribed topics have published at least one message before running inference
- Call `policy.reset()` between task episodes — currently this happens automatically on node startup; add a ROS2 service call here if you need mid-session resets
- Camera key names (`wrist_camera`, `scene_camera`, `base_camera`) may need to be aligned with the keys `lerobot/pi0_base` was trained on — check the model card if inference errors on unrecognised observation keys

