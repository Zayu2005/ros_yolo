import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    package_share = get_package_share_directory("object_detection")
    params_file = os.path.join(
        package_share,
        "config",
        "drone_publisher.yaml",
    )

    return LaunchDescription(
        [
            Node(
                package="object_detection",
                executable="drone_data_publisher",
                name="drone_data_publisher",
                output="screen",
                parameters=[params_file],
            ),
        ]
    )
