#!/usr/bin/env python3

import time

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from sensor_msgs.msg import Image
from ultralytics import YOLO


class YoloSubscriber(Node):
    """Subscribe to camera images, run YOLOv8, and display detections."""

    def __init__(self):
        super().__init__("yolo_subscriber")

        self.declare_parameter("image_topic", "/camera/image_raw")
        self.declare_parameter("model_path", "yolov8n.pt")
        self.declare_parameter("confidence", 0.5)
        self.declare_parameter("device", "auto")
        self.declare_parameter("window_name", "YOLOv8 Detection")
        self.declare_parameter("show_fps", True)

        self.image_topic = self.get_parameter("image_topic").value
        self.model_path = self.get_parameter("model_path").value
        self.confidence = self.get_parameter("confidence").value
        self.device = self.get_parameter("device").value
        self.window_name = self.get_parameter("window_name").value
        self.show_fps = self.get_parameter("show_fps").value

        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0")

        self.bridge = CvBridge()
        self.smoothed_fps = None
        self.first_frame_received = False
        self.display_error_reported = False
        self.shutdown_requested = False

        self.get_logger().info(
            f"正在加载YOLOv8模型：{self.model_path}"
        )

        try:
            self.model = YOLO(self.model_path)
        except Exception as error:
            raise RuntimeError(
                f"YOLOv8模型加载失败：{error}"
            ) from error

        # A one-frame best-effort queue favors current camera data over stale frames.
        image_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.subscription = self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
            image_qos,
        )

        self.get_logger().info(
            f"模型加载完成，正在订阅：{self.image_topic}"
        )

    def image_callback(self, message):
        """Convert one image, run inference, and display the result."""
        if self.shutdown_requested:
            return

        started_at = time.perf_counter()

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

        predict_options = {
            "source": frame,
            "conf": self.confidence,
            "verbose": False,
        }

        if self.device != "auto":
            predict_options["device"] = self.device

        try:
            results = self.model.predict(**predict_options)
            annotated_frame = results[0].plot()
        except Exception as error:
            self.get_logger().error(
                f"YOLOv8推理失败：{error}"
            )
            return

        elapsed = time.perf_counter() - started_at
        current_fps = 1.0 / elapsed if elapsed > 0 else 0.0
        self.smoothed_fps = (
            current_fps
            if self.smoothed_fps is None
            else 0.9 * self.smoothed_fps + 0.1 * current_fps
        )

        if self.show_fps:
            cv2.putText(
                annotated_frame,
                f"FPS: {self.smoothed_fps:.1f}",
                (16, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        try:
            cv2.imshow(self.window_name, annotated_frame)
            key = cv2.waitKey(1) & 0xFF
        except Exception as error:
            if not self.display_error_reported:
                self.display_error_reported = True
                self.get_logger().error(
                    f"OpenCV窗口显示失败：{error}"
                )
            return

        if key in (ord("q"), 27):
            self.shutdown_requested = True
            self.get_logger().info("收到退出指令，正在关闭节点")
            self.close_display()
            rclpy.shutdown()

    def close_display(self):
        """Close OpenCV windows and flush pending GUI events."""
        try:
            cv2.destroyAllWindows()

            # HighGUI processes window destruction through its event loop.
            for _ in range(5):
                cv2.waitKey(1)
        except cv2.error:
            pass

    def destroy_node(self):
        self.close_display()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None

    try:
        node = YoloSubscriber()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except Exception as error:
        print(f"YOLOv8接收节点运行失败：{error}")
    finally:
        if node is not None:
            node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
