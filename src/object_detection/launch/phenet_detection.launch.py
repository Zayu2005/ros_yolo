import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import Shutdown
from launch_ros.actions import Node


def generate_launch_description():
    package_share = get_package_share_directory("object_detection")
    params_file = os.path.join(
        package_share,
        "config",
        "phenet_params.yaml",
    )

    return LaunchDescription(
        [
            Node(
                package="object_detection",
                executable="phenet_subscriber",
                name="phenet_subscriber",
                output="screen",
                parameters=[params_file],
                on_exit=Shutdown(
                    reason="PHENet detection node exited",
                ),
            ),
        ]
    )
