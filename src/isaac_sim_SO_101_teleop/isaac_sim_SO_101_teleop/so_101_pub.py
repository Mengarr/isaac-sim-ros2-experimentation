"""
so_101_pub.py
-------------
Standalone script for publishing SO-101 leader arm joint positions

Requirements:
    - ROS2 sourced
    - lerobot venv sourced

"""

import math

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig

# sts3215 resolution - 1; used in the DEGREES normalization formula
_FEETECH_MAX_RES = 4095

# Target output limits (degrees for body joints, raw 0-100 for gripper)
_OUTPUT_LIMITS = {
    "shoulder_pan":  (-110.0,    110.0),
    "shoulder_lift": (-100.0,    100.0),
    "elbow_flex":    (-96.83,     96.83),
    "wrist_flex":    (-95.0,      95.0),
    "wrist_roll":    (-157.211,  162.789),
    "gripper":       (-10.0,     100.0),
}


def _input_range(calib, motor: str) -> tuple[float, float]:
    """Return the (min, max) that get_action() can produce for a motor.

    Body joints use DEGREES norm: (raw - mid) * 360 / max_res.
    Gripper uses RANGE_0_100: always 0–100.
    """
    if motor == "gripper":
        return 0.0, 100.0
    min_ = calib[motor].range_min
    max_ = calib[motor].range_max
    mid = (min_ + max_) / 2
    half = (max_ - min_) / 2 * 360 / _FEETECH_MAX_RES
    return -half, half


def _remap(value: float, in_min: float, in_max: float, out_min: float, out_max: float, clamp: bool = False) -> float:
    if clamp:
        value = max(in_min, min(in_max, value))
    return out_min + (value - in_min) / (in_max - in_min) * (out_max - out_min)


class SO101Publisher(Node):
    def __init__(self):
        super().__init__("so101_publisher")

        self.declare_parameter("map_joints", True)
        self.declare_parameter("clamp_joints", True)
        self.declare_parameter("publish_radians", True)
        # -1 = disabled; 0–100 = % of gripper range from closed that triggers fully-open snap
        self.declare_parameter("gripper_open_threshold", -1)

        config = SO101LeaderConfig(port="/dev/ttyACM0", id="leader")
        self.robot = SO101Leader(config)
        self.robot.connect()

        # Pre-compute input ranges from calibration loaded by the robot
        self._input_ranges = {
            motor: _input_range(self.robot.calibration, motor)
            for motor in _OUTPUT_LIMITS
        }

        self.pub = self.create_publisher(JointState, "joint_command", 10)
        self.create_timer(1 / 60, self.publish)  # 60 Hz to match lerobot default

    def publish(self):
        action = self.robot.get_action()
        names = [k.removesuffix(".pos") for k in action]
        positions = list(action.values())

        if self.get_parameter("map_joints").value:
            clamp = self.get_parameter("clamp_joints").value
            mapped = []
            for name, val in zip(names, positions):
                in_min, in_max = self._input_ranges[name]
                out_min, out_max = _OUTPUT_LIMITS[name]
                mapped.append(_remap(val, in_min, in_max, out_min, out_max, clamp=clamp))
            positions = mapped

        # 90° offset makes the leader arm's neutral wrist pose feel natural to the operator
        wrist_offset_deg = 90.0
        positions = [
            p + wrist_offset_deg if name == "wrist_roll" else p
            for name, p in zip(names, positions)
        ]

        gripper_threshold = self.get_parameter("gripper_open_threshold").value
        if 0 <= gripper_threshold <= 100:
            g_min, g_max = _OUTPUT_LIMITS["gripper"]
            cutoff = g_min + (gripper_threshold / 100.0) * (g_max - g_min)
            positions = [
                (g_max if p >= cutoff else g_min) if name == "gripper" else p
                for name, p in zip(names, positions)
            ]

        if self.get_parameter("publish_radians").value:
            positions = [math.radians(p) for p in positions]

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = names
        msg.position = positions
        self.pub.publish(msg)

    def destroy_node(self):
        self.robot.disconnect()
        super().destroy_node()

rclpy.init()
node = SO101Publisher()
rclpy.spin(node)
