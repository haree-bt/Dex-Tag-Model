#!/usr/bin/env python3
"""
End-to-End ROS 2 Loop Verification for Dex-TAG
Publishes test RGB-D frames, listens for target grasp pose and contact mode.
"""

import sys
import time
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String, Header
from cv_bridge import CvBridge


class DexPipelineTester(Node):
    def __init__(self):
        super().__init__('dex_pipeline_tester')
        self.bridge = CvBridge()

        self.rgb_pub = self.create_publisher(Image, '/camera/image', 10)
        self.depth_pub = self.create_publisher(Image, '/camera/depth_image', 10)
        self.info_pub = self.create_publisher(CameraInfo, '/camera/camera_info', 10)
        self.task_pub = self.create_publisher(String, '/task_command', 10)

        self.pose_received = False
        self.mode_received = None

        self.pose_sub = self.create_subscription(PoseStamped, '/target_grasp_pose', self.pose_cb, 10)
        self.mode_sub = self.create_subscription(String, '/dex_tag/contact_mode', self.mode_cb, 10)
        self.telemetry_sub = self.create_subscription(String, '/dex_tag/telemetry', self.telemetry_cb, 10)

        self.timer = self.create_timer(0.4, self.publish_frame)
        self.get_logger().info('Dex-TAG Pipeline Tester Online. Streaming frames...')

    def pose_cb(self, msg: PoseStamped):
        p = msg.pose.position
        q = msg.pose.orientation
        self.get_logger().info(f'[PERCEPTION] Target Pose: X={p.x:.3f}, Y={p.y:.3f}, Z={p.z:.3f} | Qz={q.z:.3f}, Qw={q.w:.3f}')
        self.pose_received = True

    def mode_cb(self, msg: String):
        self.mode_received = msg.data
        self.get_logger().info(f'[TACTILE MODE] Contact Mode: {self.mode_received}')

    def telemetry_cb(self, msg: String):
        self.get_logger().info(f'[TELEMETRY] {msg.data}')

    def publish_frame(self):
        img = np.full((480, 640, 3), 35, dtype=np.uint8)
        # Blue cylindrical handle (Safe Grip Zone)
        import cv2
        cv2.rectangle(img, (180, 220), (320, 260), (220, 110, 30), -1)
        # Orange tool blade (Functional End)
        cv2.rectangle(img, (320, 210), (460, 270), (30, 110, 230), -1)

        depth = np.full((480, 640), 0.70, dtype=np.float32)
        depth[210:270, 180:460] = 0.65

        hdr = Header(stamp=self.get_clock().now().to_msg(), frame_id='camera_link')

        rgb_msg = self.bridge.cv2_to_imgmsg(img, encoding='bgr8')
        rgb_msg.header = hdr

        depth_msg = self.bridge.cv2_to_imgmsg(depth, encoding='32FC1')
        depth_msg.header = hdr

        info_msg = CameraInfo()
        info_msg.header = hdr
        info_msg.k = [554.25, 0.0, 320.0, 0.0, 554.25, 240.0, 0.0, 0.0, 1.0]

        task_msg = String()
        task_msg.data = 'handover'

        self.rgb_pub.publish(rgb_msg)
        self.depth_pub.publish(depth_msg)
        self.info_pub.publish(info_msg)
        self.task_pub.publish(task_msg)


def main():
    rclpy.init()
    tester = DexPipelineTester()
    start_t = time.time()
    try:
        while rclpy.ok() and (time.time() - start_t < 6.0):
            rclpy.spin_once(tester, timeout_sec=0.1)
            if tester.pose_received and tester.mode_received:
                tester.get_logger().info(f'>>> SUCCESS: Dex-TAG Verified! Mode: {tester.mode_received}')
                break
    finally:
        tester.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
