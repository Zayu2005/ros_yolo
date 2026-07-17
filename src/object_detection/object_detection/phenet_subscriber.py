#!/usr/bin/env python3

import importlib
import sys
import time
from pathlib import Path

import cv2
import message_filters
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image


class PhenetSubscriber(Node):
    """Synchronize RGB/IR images and run PHENet RGB-T OBB inference."""

    def __init__(self):
        super().__init__("phenet_subscriber")

        self.declare_parameter("rgb_topic", "/drone/rgb/image_raw")
        self.declare_parameter(
            "infrared_topic",
            "/drone/infrared/image_raw",
        )
        self.declare_parameter("model_path", "")
        self.declare_parameter("backend_path", "")
        self.declare_parameter("imgsz", 640)
        self.declare_parameter("confidence", 0.25)
        self.declare_parameter("iou", 0.7)
        self.declare_parameter("device", "auto")
        self.declare_parameter("half", False)
        self.declare_parameter("max_det", 300)
        self.declare_parameter("sync_queue_size", 10)
        self.declare_parameter("sync_slop", 0.05)
        self.declare_parameter("resize_infrared", True)
        self.declare_parameter("window_name", "PHENet RGB-T OBB Detection")
        self.declare_parameter("show_fps", True)

        self.rgb_topic = str(self.get_parameter("rgb_topic").value)
        self.infrared_topic = str(
            self.get_parameter("infrared_topic").value
        )
        self.model_path = Path(
            str(self.get_parameter("model_path").value)
        ).expanduser()
        self.backend_path = Path(
            str(self.get_parameter("backend_path").value)
        ).expanduser()
        self.imgsz = int(self.get_parameter("imgsz").value)
        self.confidence = float(
            self.get_parameter("confidence").value
        )
        self.iou = float(self.get_parameter("iou").value)
        self.device = str(self.get_parameter("device").value)
        self.half = bool(self.get_parameter("half").value)
        self.max_det = int(self.get_parameter("max_det").value)
        self.sync_queue_size = int(
            self.get_parameter("sync_queue_size").value
        )
        self.sync_slop = float(self.get_parameter("sync_slop").value)
        self.resize_infrared = bool(
            self.get_parameter("resize_infrared").value
        )
        self.window_name = str(
            self.get_parameter("window_name").value
        )
        self.show_fps = bool(self.get_parameter("show_fps").value)

        self._validate_parameters()
        yolo_class = self._load_yolo_class()
        self.bridge = CvBridge()
        self.first_frame_received = False
        self.display_error_reported = False
        self.shutdown_requested = False
        self.smoothed_fps = None

        self.get_logger().info(
            f"正在加载PHENet模型：{self.model_path}"
        )
        try:
            self.model = yolo_class(str(self.model_path))
        except Exception as error:
            raise RuntimeError(
                f"PHENet模型加载失败：{error}"
            ) from error

        self.rgb_subscriber = message_filters.Subscriber(
            self,
            Image,
            self.rgb_topic,
            qos_profile=qos_profile_sensor_data,
        )
        self.infrared_subscriber = message_filters.Subscriber(
            self,
            Image,
            self.infrared_topic,
            qos_profile=qos_profile_sensor_data,
        )
        self.synchronizer = message_filters.ApproximateTimeSynchronizer(
            [self.rgb_subscriber, self.infrared_subscriber],
            queue_size=self.sync_queue_size,
            slop=self.sync_slop,
        )
        self.synchronizer.registerCallback(self.image_callback)

        self.get_logger().info(
            f"RGB话题：{self.rgb_topic}，红外话题：{self.infrared_topic}"
        )
        self.get_logger().info(
            "PHENet输入模式：RGBRGB6C，输入通道数：6，任务类型：OBB"
        )

    def _validate_parameters(self):
        backend_package = self.backend_path / "ultralytics"
        if not backend_package.is_dir():
            raise FileNotFoundError(
                f"PHENet后端目录不存在：{backend_package}"
            )
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"模型文件不存在：{self.model_path}"
            )
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence必须位于0.0到1.0之间")
        if not 0.0 <= self.iou <= 1.0:
            raise ValueError("iou必须位于0.0到1.0之间")
        if self.imgsz <= 0:
            raise ValueError("imgsz必须大于0")
        if self.max_det <= 0:
            raise ValueError("max_det必须大于0")
        if self.sync_queue_size <= 0:
            raise ValueError("sync_queue_size必须大于0")
        if self.sync_slop < 0.0:
            raise ValueError("sync_slop不能小于0")
        if self.rgb_topic == self.infrared_topic:
            raise ValueError("RGB话题和红外话题不能相同")

    def _load_yolo_class(self):
        """Import the bundled RGB-T backend before any other ultralytics copy."""
        backend_path = str(self.backend_path.resolve())
        if backend_path in sys.path:
            sys.path.remove(backend_path)
        sys.path.insert(0, backend_path)

        loaded_modules = [
            name
            for name in sys.modules
            if name == "ultralytics" or name.startswith("ultralytics.")
        ]
        for module_name in loaded_modules:
            del sys.modules[module_name]

        try:
            ultralytics_module = importlib.import_module("ultralytics")
        except Exception as error:
            raise RuntimeError(
                f"PHENet自定义后端导入失败：{error}"
            ) from error

        module_file = Path(ultralytics_module.__file__).resolve()
        expected_package = (self.backend_path / "ultralytics").resolve()
        if not module_file.is_relative_to(expected_package):
            raise RuntimeError(
                "加载了错误的Ultralytics后端："
                f"{module_file}，期望目录为{expected_package}"
            )

        self.get_logger().info(
            f"使用PHENet自定义后端：{module_file}"
        )
        return ultralytics_module.YOLO

    def image_callback(self, rgb_message, infrared_message):
        """Convert one synchronized pair and run PHENet inference."""
        if self.shutdown_requested:
            return

        started_at = time.perf_counter()
        try:
            rgb_image = self.bridge.imgmsg_to_cv2(
                rgb_message,
                desired_encoding="bgr8",
            )
            infrared_image = self.bridge.imgmsg_to_cv2(
                infrared_message,
                desired_encoding="bgr8",
            )
        except Exception as error:
            self.get_logger().error(
                f"双模态图像转换失败：{error}"
            )
            return

        if not self.first_frame_received:
            self.first_frame_received = True
            self.get_logger().info(
                "已收到第一对同步图像："
                f"RGB={rgb_image.shape}，IR={infrared_image.shape}"
            )

        if rgb_image.shape[:2] != infrared_image.shape[:2]:
            if not self.resize_infrared:
                self.get_logger().error(
                    "RGB和红外图像尺寸不同，且resize_infrared为false"
                )
                return
            infrared_image = cv2.resize(
                infrared_image,
                (rgb_image.shape[1], rgb_image.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )

        image_6c = np.ascontiguousarray(
            np.concatenate((rgb_image, infrared_image), axis=2)
        )
        predict_options = {
            "source": image_6c,
            "imgsz": self.imgsz,
            "conf": self.confidence,
            "iou": self.iou,
            "half": self.half,
            "max_det": self.max_det,
            "use_simotm": "RGBRGB6C",
            "channels": 6,
            "verbose": False,
        }
        if self.device and self.device.lower() != "auto":
            predict_options["device"] = self.device

        try:
            results = self.model.predict(**predict_options)
            # Use RGB as the display canvas; the model still receives all 6 channels.
            annotated_image = results[0].plot(img=rgb_image.copy())
        except Exception as error:
            self.get_logger().error(
                f"PHENet推理失败：{error}"
            )
            return

        elapsed = time.perf_counter() - started_at
        current_fps = 1.0 / elapsed if elapsed > 0.0 else 0.0
        self.smoothed_fps = (
            current_fps
            if self.smoothed_fps is None
            else 0.9 * self.smoothed_fps + 0.1 * current_fps
        )
        if self.show_fps:
            cv2.putText(
                annotated_image,
                f"FPS: {self.smoothed_fps:.1f}",
                (16, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        try:
            cv2.imshow(self.window_name, annotated_image)
            key = cv2.waitKey(1) & 0xFF
        except Exception as error:
            if not self.display_error_reported:
                self.display_error_reported = True
                self.get_logger().error(
                    f"OpenCV窗口显示失败：{error}"
                )
            return

        if key in (ord("q"), ord("Q"), 27):
            self.shutdown_requested = True
            self.get_logger().info(
                "收到退出指令，正在关闭PHENet检测节点"
            )
            self.close_display()
            rclpy.shutdown()

    def close_display(self):
        """Close OpenCV windows and flush pending GUI events."""
        try:
            cv2.destroyAllWindows()
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
        node = PhenetSubscriber()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except Exception as error:
        print(f"PHENet双模态检测节点运行失败：{error}")
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
