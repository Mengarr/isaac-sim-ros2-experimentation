import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('vla_arm_control')
    pi0_config     = os.path.join(pkg_share, 'config', 'pi0.yaml')
    smoother_config = os.path.join(pkg_share, 'config', 'smoother.yaml')

    joint_names_arg = DeclareLaunchArgument(
        'joint_names',
        default_value='',
        description=(
            'Comma-separated joint names matching Isaac Sim joint names, e.g. '
            'joint_1,joint_2,joint_3,joint_4,joint_5,joint_6,gripper. '
            'If empty, joint_names from the yaml config is used.'
        ),
    )

    def create_nodes(context):
        jn_str = LaunchConfiguration('joint_names').perform(context)
        joint_names = [j.strip() for j in jn_str.split(',') if j.strip()]
        joint_names_param = {'joint_names': joint_names} if joint_names else {}

        pi0_node = Node(
            package='vla_arm_control',
            executable='pi0_node',
            name='pi0_node',
            output='screen',
            parameters=[pi0_config, joint_names_param],
        )

        smoother_node = Node(
            package='vla_arm_control',
            executable='smoother_node',
            name='smoother_node',
            output='screen',
            parameters=[smoother_config, joint_names_param],
        )

        return [pi0_node, smoother_node]

    return LaunchDescription([
        joint_names_arg,
        OpaqueFunction(function=create_nodes),
    ])
