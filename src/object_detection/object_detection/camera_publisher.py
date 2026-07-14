#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image


class CameraPublisher(Node):
    """将usb_cam发布的图像转发到项目统一的摄像头话题。"""

    def __init__(self):
        # 创建名为camera_publisher的ROS2节点
        super().__init__("camera_publisher")

        # 输入话题：由usb_cam_node_exe发布
        self.declare_parameter(
            "input_topic",
            "/usb_cam/image_raw",
        )

        # 输出话题：提供给YOLO接收节点
        self.declare_parameter(
            "output_topic",
            "/camera/image_raw",
        )

        # 可选的坐标系名称。
        # 设置为空字符串时保留usb_cam消息原有的frame_id。
        self.declare_parameter(
            "override_frame_id",
            "",
        )

        # 每转发多少帧输出一次统计日志
        self.declare_parameter(
            "log_every_n_frames",
            100,
        )

        # 获取参数的实际值
        self.input_topic = self.get_parameter(
            "input_topic"
        ).value

        self.output_topic = self.get_parameter(
            "output_topic"
        ).value

        self.override_frame_id = self.get_parameter(
            "override_frame_id"
        ).value

        self.log_every_n_frames = self.get_parameter(
            "log_every_n_frames"
        ).value

        # 防止输入和输出使用相同话题。
        # 如果两个话题相同，节点会不断收到自己发布的消息，
        # 从而形成无限循环。
        if self.input_topic == self.output_topic:
            raise ValueError(
                "input_topic和output_topic不能相同"
            )

        if self.log_every_n_frames <= 0:
            raise ValueError(
                "log_every_n_frames必须大于0"
            )

        # 创建ROS2图像发布者。
        # 发布的消息类型是sensor_msgs/msg/Image。
        self.publisher = self.create_publisher(
            Image,
            self.output_topic,
            qos_profile_sensor_data,
        )

        # 创建ROS2图像订阅者。
        # 每收到一帧usb_cam图像，就调用image_callback。
        self.subscription = self.create_subscription(
            Image,
            self.input_topic,
            self.image_callback,
            qos_profile_sensor_data,
        )

        # 记录已经转发的图像数量
        self.frame_count = 0

        # 用于判断是否收到第一帧图像
        self.first_frame_received = False

        self.get_logger().info(
            f"输入图像话题：{self.input_topic}"
        )
        self.get_logger().info(
            f"输出图像话题：{self.output_topic}"
        )
        self.get_logger().info(
            "正在等待usb_cam发布图像"
        )

    def image_callback(self, message):
        """收到usb_cam图像后，将其发布到项目输出话题。"""

        # message本身已经是sensor_msgs/msg/Image类型，
        # 因此不需要使用cv_bridge或OpenCV进行转换。

        # 如果配置了新的frame_id，则覆盖驱动提供的值。
        # 默认保留usb_cam原始frame_id和采集时间戳。
        if self.override_frame_id:
            message.header.frame_id = (
                self.override_frame_id
            )

        # 将收到的图像直接发布到输出话题
        self.publisher.publish(message)

        # 更新帧计数
        self.frame_count += 1

        # 第一帧到达时，输出图像基本信息，
        # 用于确认摄像头数据已经成功进入本节点。
        if not self.first_frame_received:
            self.first_frame_received = True

            self.get_logger().info(
                "已收到第一帧图像："
                f"{message.width}x{message.height}，"
                f"编码={message.encoding}，"
                f"frame_id={message.header.frame_id}"
            )

        # 每转发指定数量的帧输出一次日志，
        # 避免每帧输出日志影响程序性能。
        if self.frame_count % self.log_every_n_frames == 0:
            self.get_logger().info(
                f"已转发{self.frame_count}帧图像"
            )


def main(args=None):
    # 初始化ROS2 Python通信环境
    rclpy.init(args=args)

    node = None

    try:
        # 创建图像中转发布节点
        node = CameraPublisher()

        # 持续处理订阅回调，直到用户按下Ctrl+C
        rclpy.spin(node)

    except KeyboardInterrupt:
        # 用户按下Ctrl+C时正常退出
        pass

    except Exception as error:
        # 输出参数错误、节点初始化错误等信息
        print(f"图像发布节点运行失败：{error}")

    finally:
        # 释放节点占用的发布者和订阅者资源
        if node is not None:
            node.destroy_node()

        # 关闭ROS2通信环境
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()