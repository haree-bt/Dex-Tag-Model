#!/usr/bin/env python3
"""
Dex-TAG ROS 2 Perception & Dexterous Grasp Synthesis Node
Subscribes to RGB-D camera sensor streams from Gazebo Harmonic,
executes high-speed inference on RTX 5070 Ti,
and publishes 6-DoF task-aware grasp poses with multi-finger contact modes.
"""

import os
import sys
import time
import math
import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped, Point, Quaternion
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import String, Header
from cv_bridge import CvBridge

# Discover Dex-TAG Project Root
candidate_roots = [
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')),
    '/mnt/c/Users/hareh/.gemini/antigravity/scratch/dex_tag_net',
    'C:/Users/hareh/.gemini/antigravity/scratch/dex_tag_net'
]
PROJECT_ROOT = candidate_roots[0]
for p in candidate_roots:
    if os.path.exists(os.path.join(p, 'models')):
        PROJECT_ROOT = p
        break

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    import torch
    torch.set_num_threads(2)
    from models.dex_tag import DexTAG
    TORCH_AVAILABLE = True
except Exception as e:
    print(f"Warning: Failed to import torch/DexTAG: {e}")
    TORCH_AVAILABLE = False


class DexTAGNode(Node):
    def __init__(self):
        super().__init__('dex_tag_node')
        self.get_logger().info('Initializing Dex-TAG ROS 2 Node...')

        # Parameters
        self.declare_parameter('checkpoint_path', os.path.join(PROJECT_ROOT, 'checkpoints', 'dex_tag_best.pth'))
        self.declare_parameter('camera_frame', 'camera_link')
        self.declare_parameter('default_task', 'handover')

        self.checkpoint_path = self.get_parameter('checkpoint_path').value
        self.camera_frame = self.get_parameter('camera_frame').value
        self.current_task = self.get_parameter('default_task').value

        # Sensor State
        self.bridge = CvBridge()
        self.latest_rgb = None
        self.latest_depth = None
        self.camera_intrinsics = None

        # Hardware Backend Selection (RTX 5070 Ti CUDA with fallback)
        self.device = torch.device('cpu')
        if TORCH_AVAILABLE and torch.cuda.is_available():
            try:
                _ = torch.zeros(1, device='cuda') + 1
                self.device = torch.device('cuda')
                self.get_logger().info(f'Perception Hardware Backend: CUDA ({torch.cuda.get_device_name(0)})')
            except Exception as e:
                self.get_logger().warn(f'CUDA available but device execution fallback to CPU: {e}')
                self.device = torch.device('cpu')
        else:
            self.get_logger().info('Perception Hardware Backend: CPU')

        # Load DexTAG Model
        self.model = None
        self.load_model()

        # Sensor Subscribers (Best Effort QoS for real-time cameras)
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        self.rgb_sub = self.create_subscription(Image, '/camera/image', self.rgb_callback, sensor_qos)
        self.depth_sub = self.create_subscription(Image, '/camera/depth_image', self.depth_callback, sensor_qos)
        self.info_sub = self.create_subscription(CameraInfo, '/camera/camera_info', self.info_callback, sensor_qos)
        self.task_sub = self.create_subscription(String, '/task_command', self.task_callback, 10)

        # Publishers
        self.grasp_pub = self.create_publisher(PoseStamped, '/target_grasp_pose', 10)
        self.mode_pub = self.create_publisher(String, '/dex_tag/contact_mode', 10)
        self.vis_pub = self.create_publisher(Image, '/dex_tag/visualization', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/dex_tag/markers', 10)
        self.telemetry_pub = self.create_publisher(String, '/dex_tag/telemetry', 10)

        # Perception Cycle (10 Hz)
        self.timer = self.create_timer(0.1, self.inference_cycle)
        self.get_logger().info('Dex-TAG Node online and listening for camera feeds!')

    def load_model(self):
        if not TORCH_AVAILABLE:
            self.get_logger().warn('PyTorch unavailable. Running in mock perception mode.')
            return

        try:
            self.model = DexTAG(in_channels=4, base_channels=32, num_modes=3).to(self.device)
            if os.path.exists(self.checkpoint_path):
                ckpt = torch.load(self.checkpoint_path, map_location=self.device)
                state_dict = ckpt['model_state_dict'] if (isinstance(ckpt, dict) and 'model_state_dict' in ckpt) else ckpt
                self.model.load_state_dict(state_dict)
                self.model.eval()
                self.get_logger().info(f'Loaded Dex-TAG weights from {self.checkpoint_path}')
            else:
                self.get_logger().warn(f'No checkpoint at {self.checkpoint_path}. Using initial weights.')
        except Exception as e:
            self.get_logger().error(f'Failed to load Dex-TAG model: {e}')
            self.model = None

    def rgb_callback(self, msg: Image):
        try:
            self.latest_rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'RGB decode error: {e}')

    def depth_callback(self, msg: Image):
        try:
            depth_raw = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
            self.latest_depth = np.nan_to_num(depth_raw, nan=1.0, posinf=2.0, neginf=0.0)
        except Exception as e:
            self.get_logger().error(f'Depth decode error: {e}')

    def info_callback(self, msg: CameraInfo):
        if self.camera_intrinsics is None:
            self.camera_intrinsics = {
                'fx': msg.k[0],
                'fy': msg.k[4],
                'cx': msg.k[2],
                'cy': msg.k[5]
            }
            self.get_logger().info(f'Camera Intrinsics: fx={msg.k[0]:.1f}, fy={msg.k[4]:.1f}, cx={msg.k[2]:.1f}, cy={msg.k[5]:.1f}')

    def task_callback(self, msg: String):
        self.current_task = msg.data.lower()
        self.get_logger().info(f'Task Context Updated: {self.current_task.upper()}')

    def inference_cycle(self):
        if self.latest_rgb is None or self.latest_depth is None:
            return

        t0 = time.time()
        H_orig, W_orig = self.latest_rgb.shape[:2]

        rgb_224 = cv2.resize(self.latest_rgb, (224, 224))
        depth_224 = cv2.resize(self.latest_depth, (224, 224))

        rgb_norm = (rgb_224.astype(np.float32) / 255.0 - 0.5) / 0.5
        depth_norm = np.clip(depth_224.astype(np.float32), 0.0, 2.0) / 2.0

        rgbd = np.concatenate([rgb_norm, np.expand_dims(depth_norm, -1)], axis=-1)
        rgbd_tensor = torch.from_numpy(rgbd.transpose(2, 0, 1)).unsqueeze(0).to(self.device)

        if self.model is not None:
            with torch.no_grad():
                out = self.model(rgbd_tensor)
                q_dex = out['quality_dex'].squeeze().cpu().numpy()
                sin_2th = out['sin_2theta'].squeeze().cpu().numpy()
                cos_2th = out['cos_2theta'].squeeze().cpu().numpy()
                aperture = out['aperture'].squeeze().cpu().numpy()
                contact_modes = out['contact_modes'].squeeze().cpu().numpy()
                mode_classes = np.argmax(contact_modes, axis=0)
        else:
            q_dex = np.ones((224, 224), dtype=np.float32) * 0.8
            sin_2th = np.zeros((224, 224), dtype=np.float32)
            cos_2th = np.ones((224, 224), dtype=np.float32)
            aperture = np.ones((224, 224), dtype=np.float32) * 0.05
            mode_classes = np.ones((224, 224), dtype=np.int64)

        latency_ms = (time.time() - t0) * 1000.0

        # Optimal Grasp Selection
        max_idx = np.unravel_index(np.argmax(q_dex), q_dex.shape)
        v_net, u_net = max_idx
        best_score = float(q_dex[v_net, u_net])

        u_orig = int(u_net * W_orig / 224)
        v_orig = int(v_net * H_orig / 224)

        theta = 0.5 * math.atan2(float(sin_2th[v_net, u_net]), float(cos_2th[v_net, u_net]))
        grasp_aperture = float(aperture[v_net, u_net])
        selected_mode_idx = int(mode_classes[v_net, u_net])
        mode_names = DexTAG.CONTACT_MODES if hasattr(DexTAG, 'CONTACT_MODES') else ['PINCH', 'TRIPOD', 'POWER']
        selected_mode = mode_names[selected_mode_idx]

        # 3D Metric Coordinate Unprojection
        Z = float(self.latest_depth[v_orig, u_orig])
        if Z <= 0.05 or Z > 3.0:
            Z = 0.65

        fx = self.camera_intrinsics['fx'] if self.camera_intrinsics else 554.25
        fy = self.camera_intrinsics['fy'] if self.camera_intrinsics else 554.25
        cx = self.camera_intrinsics['cx'] if self.camera_intrinsics else W_orig / 2.0
        cy = self.camera_intrinsics['cy'] if self.camera_intrinsics else H_orig / 2.0

        X = (u_orig - cx) * Z / fx
        Y = (v_orig - cy) * Z / fy

        qz = math.sin(theta / 2.0)
        qw = math.cos(theta / 2.0)

        hdr = Header(stamp=self.get_clock().now().to_msg(), frame_id=self.camera_frame)

        # 1. Publish 6-DoF Pose
        pose_msg = PoseStamped()
        pose_msg.header = hdr
        pose_msg.pose.position = Point(x=float(X), y=float(Y), z=float(Z))
        pose_msg.pose.orientation = Quaternion(x=0.0, y=0.0, z=float(qz), w=float(qw))
        self.grasp_pub.publish(pose_msg)

        # 2. Publish Contact Mode
        mode_msg = String()
        mode_msg.data = selected_mode
        self.mode_pub.publish(mode_msg)

        # 3. Publish Visualization Image
        vis_img = self.latest_rgb.copy()
        cv2.circle(vis_img, (u_orig, v_orig), 7, (0, 0, 255), -1)

        length = 40
        w_px = max(15, int(grasp_aperture * 250))
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)

        p1 = (int(u_orig - length*sin_t - w_px*cos_t), int(v_orig + length*cos_t - w_px*sin_t))
        p2 = (int(u_orig + length*sin_t - w_px*cos_t), int(v_orig - length*cos_t - w_px*sin_t))
        p3 = (int(u_orig + length*sin_t + w_px*cos_t), int(v_orig - length*cos_t + w_px*sin_t))
        p4 = (int(u_orig - length*sin_t + w_px*cos_t), int(v_orig + length*cos_t + w_px*sin_t))

        box_color = (0, 255, 0) if selected_mode == 'TRIPOD_DEXTEROUS' else (255, 120, 0)
        cv2.polylines(vis_img, [np.array([p1, p2, p3, p4], dtype=np.int32)], isClosed=True, color=box_color, thickness=2)
        cv2.putText(vis_img, f'Mode: {selected_mode} | TSR: {best_score:.2f} | {latency_ms:.1f}ms',
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

        try:
            vis_msg = self.bridge.cv2_to_imgmsg(vis_img, encoding='bgr8')
            vis_msg.header = hdr
            self.vis_pub.publish(vis_msg)
        except Exception:
            pass

        # 4. Publish 3D Markers for RViz
        self.publish_markers(X, Y, Z, theta, hdr, selected_mode)

        # 5. Publish Telemetry
        telemetry_msg = String()
        telemetry_msg.data = f'{{"task": "{self.current_task}", "mode": "{selected_mode}", "score": {best_score:.3f}, "latency_ms": {latency_ms:.2f}, "x": {X:.3f}, "y": {Y:.3f}, "z": {Z:.3f}, "theta_deg": {math.degrees(theta):.1f}}}'
        self.telemetry_pub.publish(telemetry_msg)

    def publish_markers(self, x, y, z, theta, header, mode):
        m_arr = MarkerArray()

        arrow = Marker()
        arrow.header = header
        arrow.ns = 'dex_approach'
        arrow.id = 0
        arrow.type = Marker.ARROW
        arrow.scale.x = 0.015
        arrow.scale.y = 0.03
        arrow.scale.z = 0.05
        arrow.color.r = 0.2
        arrow.color.g = 0.9
        arrow.color.b = 0.3
        arrow.color.a = 0.95
        arrow.points.append(Point(x=x, y=y, z=z - 0.15))
        arrow.points.append(Point(x=x, y=y, z=z))
        m_arr.markers.append(arrow)

        sphere = Marker()
        sphere.header = header
        sphere.ns = 'dex_contact'
        sphere.id = 1
        sphere.type = Marker.SPHERE
        sphere.pose.position = Point(x=x, y=y, z=z)
        sphere.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        sphere.scale.x = 0.03
        sphere.scale.y = 0.03
        sphere.scale.z = 0.03
        sphere.color.r = 1.0
        sphere.color.g = 0.5
        sphere.color.b = 0.0
        sphere.color.a = 0.95
        m_arr.markers.append(sphere)

        self.marker_pub.publish(m_arr)


def main(args=None):
    rclpy.init(args=args)
    node = DexTAGNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
