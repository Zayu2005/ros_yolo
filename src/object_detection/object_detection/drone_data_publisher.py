#!/usr/bin/env python3

from pathlib import Path

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image


class DroneDataPublisher(Node):
    """Replay paired RGB and infrared images as ROS2 image topics."""

    def __init__(self):
        super().__init__("drone_data_publisher")

        self.declare_parameter("rgb_directory", "")
        self.declare_parameter("infrared_directory", "")
        self.declare_parameter("rgb_topic", "/drone/rgb/image_raw")
        self.declare_parameter(
            "infrared_topic",
            "/drone/infrared/image_raw",
        )
        self.declare_parameter("publish_rate", 10.0)
        self.declare_parameter("loop_playback", True)
        self.declare_parameter("rgb_frame_id", "rgb_camera_frame")
        self.declare_parameter(
            "infrared_frame_id",
            "infrared_camera_frame",
        )
        self.declare_parameter("log_every_n_frames", 50)
        self.declare_parameter(
            "image_extensions",
            [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"],
        )

        self.rgb_directory = self._read_directory("rgb_directory")
        self.infrared_directory = self._read_directory(
            "infrared_directory"
        )
        self.rgb_topic = self.get_parameter("rgb_topic").value
        self.infrared_topic = self.get_parameter(
            "infrared_topic"
        ).value
        self.publish_rate = float(
            self.get_parameter("publish_rate").value
        )
        self.loop_playback = bool(
            self.get_parameter("loop_playback").value
        )
        self.rgb_frame_id = self.get_parameter("rgb_frame_id").value
        self.infrared_frame_id = self.get_parameter(
            "infrared_frame_id"
        ).value
        self.log_every_n_frames = int(
            self.get_parameter("log_every_n_frames").value
        )
        self.image_extensions = {
            extension.lower()
            if extension.startswith(".")
            else f".{extension.lower()}"
            for extension in self.get_parameter(
                "image_extensions"
            ).value
        }

        self._validate_parameters()
        self.image_pairs = self._find_image_pairs()
        self.bridge = CvBridge()
        self.current_index = 0
        self.published_count = 0

        self.rgb_publisher = self.create_publisher(
            Image,
            self.rgb_topic,
            qos_profile_sensor_data,
        )
        self.infrared_publisher = self.create_publisher(
            Image,
            self.infrared_topic,
            qos_profile_sensor_data,
        )

        self.timer = self.create_timer(
            1.0 / self.publish_rate,
            self.publish_next_pair,
        )

        self.get_logger().info(
            f"找到 {len(self.image_pairs)} 对 RGB/红外图像"
        )
        self.get_logger().info(f"RGB话题：{self.rgb_topic}")
        self.get_logger().info(
            f"红外话题：{self.infrared_topic}"
        )
        self.get_logger().info(
            f"数据回放频率：{self.publish_rate:.2f} Hz"
        )

    def _read_directory(self, parameter_name):
        value = str(self.get_parameter(parameter_name).value).strip()
        if not value:
            raise ValueError(f"参数 {parameter_name} 不能为空")
        return Path(value).expanduser().resolve()

    def _validate_parameters(self):
        if not self.rgb_directory.is_dir():
            raise FileNotFoundError(
                f"RGB图像目录不存在：{self.rgb_directory}"
            )
        if not self.infrared_directory.is_dir():
            raise FileNotFoundError(
                f"红外图像目录不存在：{self.infrared_directory}"
            )
        if self.publish_rate <= 0.0:
            raise ValueError("publish_rate必须大于0")
        if self.log_every_n_frames <= 0:
            raise ValueError("log_every_n_frames必须大于0")
        if not self.image_extensions:
            raise ValueError("image_extensions不能为空")
        if self.rgb_topic == self.infrared_topic:
            raise ValueError("RGB话题和红外话题不能相同")

    def _collect_images(self, directory):
        images = {}
        for path in directory.iterdir():
            if not path.is_file():
                continue
            if path.suffix.lower() not in self.image_extensions:
                continue

            sample_id = path.stem
            if sample_id in images:
                raise ValueError(
                    f"目录中存在重复样本编号 {sample_id}：{directory}"
                )
            images[sample_id] = path
        return images

    def _find_image_pairs(self):
        rgb_images = self._collect_images(self.rgb_directory)
        infrared_images = self._collect_images(
            self.infrared_directory
        )
        common_ids = sorted(
            rgb_images.keys() & infrared_images.keys()
        )

        rgb_only_count = len(rgb_images.keys() - infrared_images.keys())
        infrared_only_count = len(
            infrared_images.keys() - rgb_images.keys()
        )
        if rgb_only_count or infrared_only_count:
            self.get_logger().warning(
                "发现未配对图像："
                f"仅RGB={rgb_only_count}，仅红外={infrared_only_count}"
            )

        if not common_ids:
            raise RuntimeError(
                "未找到同名的RGB和红外图像，请检查数据集目录"
            )

        return [
            (sample_id, rgb_images[sample_id], infrared_images[sample_id])
            for sample_id in common_ids
        ]

    def publish_next_pair(self):
        """Read and publish the next synchronized image pair."""
        if self.current_index >= len(self.image_pairs):
            if self.loop_playback:
                self.current_index = 0
                self.get_logger().info("数据集回放已重新开始")
            else:
                self.timer.cancel()
                self.get_logger().info("数据集回放完成，停止发布")
                return

        sample_id, rgb_path, infrared_path = self.image_pairs[
            self.current_index
        ]
        rgb_image = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        infrared_image = cv2.imread(
            str(infrared_path),
            cv2.IMREAD_COLOR,
        )

        if rgb_image is None or infrared_image is None:
            self.get_logger().error(
                f"样本 {sample_id} 读取失败，已跳过"
            )
            self.current_index += 1
            return

        # 两个消息共享同一个采集时刻，便于接收端进行双模态同步。
        stamp = self.get_clock().now().to_msg()
        rgb_message = self.bridge.cv2_to_imgmsg(
            rgb_image,
            encoding="bgr8",
        )
        infrared_message = self.bridge.cv2_to_imgmsg(
            infrared_image,
            encoding="bgr8",
        )
        rgb_message.header.stamp = stamp
        infrared_message.header.stamp = stamp
        rgb_message.header.frame_id = self.rgb_frame_id
        infrared_message.header.frame_id = self.infrared_frame_id

        self.rgb_publisher.publish(rgb_message)
        self.infrared_publisher.publish(infrared_message)

        self.current_index += 1
        self.published_count += 1
        if self.published_count == 1:
            self.get_logger().info(
                f"已发布第一对图像，样本编号：{sample_id}"
            )
        elif self.published_count % self.log_every_n_frames == 0:
            self.get_logger().info(
                f"已发布 {self.published_count} 对图像，"
                f"当前样本：{sample_id}"
            )


def main(args=None):
    rclpy.init(args=args)
    node = None

    try:
        node = DroneDataPublisher()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as error:
        print(f"开发板数据回放节点运行失败：{error}")
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
