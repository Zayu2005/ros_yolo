#!/usr/bin/env python3

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image


class ImageSubscriber(Node):
    """Subscribe to ROS2 images and display them with OpenCV."""

    def __init__(self):
        super().__init__("image_subscriber")

        self.declare_parameter("image_topic", "/camera/image_raw")
        self.declare_parameter("window_name", "ROS2 Camera")

        self.image_topic = self.get_parameter("image_topic").value
        self.window_name = self.get_parameter("window_name").value

        self.bridge = CvBridge()
        self.first_frame_received = False

        self.subscription = self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
            qos_profile_sensor_data,
        )

        self.get_logger().info(
            f"正在订阅图像话题：{self.image_topic}"
        )

    def image_callback(self, message):
        """Convert one ROS2 image message and display it."""
        try:
            frame = self.bridge.imgmsg_to_cv2(
                message,
                desired_encoding="bgr8",
            )
        except Exception as error:
            self.get_logger().error(
                f"ROS2图像转换失败：{error}"
            )
            return

        if not self.first_frame_received:
            self.first_frame_received = True
            self.get_logger().info(
                "已收到第一帧图像："
                f"{message.width}x{message.height}，"
                f"编码={message.encoding}"
            )

        try:
            cv2.imshow(self.window_name, frame)
            key = cv2.waitKey(1) & 0xFF
        except Exception as error:
            self.get_logger().error(
                f"OpenCV窗口显示失败：{error}"
            )
            return

        if key in (ord("q"), 27):
            self.get_logger().info("收到退出指令，正在关闭节点")
            rclpy.shutdown()

    def destroy_node(self):
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None

    try:
        node = ImageSubscriber()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
