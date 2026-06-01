# record_dataset_so_101

ROS2 node that records a [LeRobotDataset](https://github.com/huggingface/lerobot) from a SO-101 arm running in Isaac Sim. Teleoperation is handled by a separate node — this node only observes and records.

## Design

### State machine

```
WAITING_FOR_SENSORS → IDLE → RECORDING → RESETTING → IDLE → ... → DONE
```

| State | Description |
|---|---|
| `WAITING_FOR_SENSORS` | Blocks until the first message arrives on all 5 topics. Once ready, lazily creates the `LeRobotDataset` using real image shapes and joint count from the messages. |
| `IDLE` | Prints a prompt. Waits for the user to press **Enter** to begin the next episode. |
| `RECORDING` | Timer fires at `fps` Hz. Each tick snapshots the latest sensor data and calls `dataset.add_frame()`. Stops on **Enter** or when `episode_duration` seconds elapse (safety cap). |
| `RESETTING` | Calls `dataset.save_episode()`, waits `reset_duration` seconds, then returns to `IDLE`. |
| `DONE` | Calls `dataset.finalize()` and shuts down. |

### Subscriptions

| Topic | Type | Role |
|---|---|---|
| `/wrist_camera/image_raw` | `sensor_msgs/Image` | `observation.images.wrist` |
| `/scene_camera/image_raw` | `sensor_msgs/Image` | `observation.images.scene` |
| `/base_camera/image_raw` | `sensor_msgs/Image` | `observation.images.base` |
| `/joint_states` | `sensor_msgs/JointState` | `observation.state` |
| `/joint_command` | `sensor_msgs/JointState` | `action` (teleop commands) |

### Dataset storage

Saved locally to `~/.cache/huggingface/lerobot/<dataset_name>/` by default. Images are encoded to MP4 in real time (`use_videos=True`, `streaming_encoding=True`). No data is uploaded to the Hugging Face Hub.

### Joint ordering

Canonical joint order is established from the first `/joint_states` message. All subsequent `/joint_command` positions are remapped to this order, so out-of-order fields are handled safely.

### Image encoding

Both `rgb8` and `bgr8` encodings are handled — BGR images are converted to RGB automatically before being written to the dataset.

### Graceful shutdown

Ctrl+C triggers `dataset.finalize()` before exit so no data is lost mid-session.

## Parameters

| Parameter | Default | Description |
|---|---|---|
| `num_episodes` | `10` | Total number of episodes to record |
| `episode_duration` | `60.0` | Max episode length in seconds (safety cap) |
| `reset_duration` | `5.0` | Cooldown seconds between episodes |
| `task_description` | `"pick and place"` | Task label stored with every frame |
| `fps` | `30` | Recording frame rate |
| `dataset_name` | `"so101_dataset"` | Dataset name / local folder name |
| `output_dir` | `""` | Override storage path (empty = use default cache) |

## Execution

### Build

```bash
cd ~/repos/isaac-sim-ros2-experimentation
colcon build --packages-select record_dataset_so_101
source install/setup.bash
```

### Run

Run directly with the venv's Python rather than `ros2 run` — `ros2 run` uses the system interpreter which doesn't have the lerobot dependencies:

```bash
python install/record_dataset_so_101/lib/record_dataset_so_101/dataset_recorder
```

With custom parameters:

```bash
python install/record_dataset_so_101/lib/record_dataset_so_101/dataset_recorder --ros-args \
  -p num_episodes:=10 \
  -p episode_duration:=60.0 \
  -p reset_duration:=5.0 \
  -p task_description:="pick and place cube" \
  -p fps:=30 \
  -p dataset_name:="so101_cube_task" \
  -p output_dir:="~/data/datasets" \
  -p streaming_encoding:=true
```

### Episode workflow

1. Start Isaac Sim with the SO-101 arm and camera bridges active
2. Start the teleoperation node (publishes `/joint_command`)
3. Run the recorder — it will print **"Waiting for all sensor topics..."**
4. Once all topics are live it prints **"All sensors ready. Dataset created."**
5. Press **Enter** to begin an episode
6. Teleop the arm through the task
7. Press **Enter** to stop and save the episode
8. Wait for the reset countdown, then repeat from step 5
9. After the last episode the node finalizes and exits automatically

### Inspecting the dataset

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

ds = LeRobotDataset("so101_dataset")
print(ds)
print(ds[0])  # first frame
```
