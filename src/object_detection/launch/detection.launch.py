import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    package_share = get_package_share_directory("object_detection")
    params_file = os.path.join(package_share, "config", "params.yaml")

    return LaunchDescription(
        [
            Node(
                package="usb_cam",
                executable="usb_cam_node_exe",
                name="usb_cam",
                output="screen",
                parameters=[params_file],
                remappings=[
                    ("image_raw", "/usb_cam/image_raw"),
                ],
            ),
            Node(
                package="object_detection",
                executable="camera_publisher",
                name="camera_publisher",
                output="screen",
                parameters=[params_file],
            ),
            Node(
                package="object_detection",
                executable="yolo_subscriber",
                name="yolo_subscriber",
                output="screen",
                parameters=[params_file],
            ),
        ]
    )
