"""
launch/joy.launch.py
--------------------
Launches the ROS2 joy driver on the controller machine.
This is separate from the Isaac Sim script which runs on the sim machine.

Usage:
    ros2 launch isaac_sim_bridge joy.launch.py

Requirements:
    sudo apt install ros-<distro>-joy
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            "joy_dev",
            default_value="0",
            description="Joystick device ID (integer index, e.g. 0 for /dev/input/js0)",
        ),
        DeclareLaunchArgument(
            "deadzone",
            default_value="0.05",
            description="Joystick axis deadzone",
        ),

        Node(
            package="joy",
            executable="joy_node",
            name="joy_node",
            parameters=[{
                "device_id": LaunchConfiguration("joy_dev"),
                "deadzone": LaunchConfiguration("deadzone"),
                "autorepeat_rate": 20.0,   # Hz — publishes even with no input
                "coalesce_interval_ms": 1,
            }],
            output="screen",
        ),
    ])