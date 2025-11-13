import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, QoSDurabilityPolicy
from rclpy.time import Time
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
import tf2_ros

from builtin_interfaces.msg import Time as TimeMsg
from geometry_msgs.msg import (
    PoseWithCovarianceStamped,
    Transform,
    TransformStamped
)
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid
from nav_msgs.srv import GetMap

import numpy as np
from threading import Lock

from lidar_localization import ICP2D
import transform2d_utils as tf2d
from occupancy_grid import OccupancyGridMap

# Indexing values
X = 0
Y = 1
THETA = 2


class ICPLocalizationNode(Node):
    def __init__(self) -> None:
        super().__init__('icp_localization_node')

        # Declare parameters
        self.declare_parameter(
            'initial_pose_x',
            0.0,
            descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_DOUBLE))
        self.declare_parameter(
            'initial_pose_y',
            0.0,
            descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_DOUBLE))
        self.declare_parameter(
            'initial_pose_a',
            0.0,
            descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_DOUBLE))
        self.declare_parameter(
            'odom_frame',
            'odom',
            descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING))
        self.declare_parameter(
            'time_travel',
            0.0,
            descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_DOUBLE))
        self.declare_parameter(
            'use_map_topic',
            False,
            descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_BOOL))

        # Set up required data
        self.have_map = False
        self.have_transform = False

        # Get initial pose (of odom with respect to map frame)
        initial_pose_x = self.get_parameter('initial_pose_x').get_parameter_value().double_value
        initial_pose_y = self.get_parameter('initial_pose_y').get_parameter_value().double_value
        initial_pose_a = self.get_parameter('initial_pose_a').get_parameter_value().double_value

        # Store current pose (of odom with respect to map) in (x, y, theta) format
        self.pose = [initial_pose_x, initial_pose_y, initial_pose_a]

        # Initialize transformation (of odom with respect to map)
        self.tf_map_odom = TransformStamped()
        self.tf_map_odom.child_frame_id = \
            self.get_parameter('odom_frame').get_parameter_value().string_value

        # Initialize ICP object
        self.icp = ICP2D()
        self.tolerance = None  # tolerance for fitting
        self.lock = Lock()  # to ensure ICP object can be locked

        # Publishers
        self.pose_pub = self.create_publisher(PoseWithCovarianceStamped, 'icp_pose', 10)

        # Subscribers
        self.sensor_frame_id = None  # frame ID of the lidar
        self.scan_sub = self.create_subscription(LaserScan, 'scan', self.scan_callback, 10)
        self.pose_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            'initialpose',
            self.initialpose_callback,
            1)

        # Get map
        if self.get_parameter('use_map_topic').get_parameter_value().bool_value:
            latching_qos = QoSProfile(
                depth=1,
                durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
            self.map_sub = self.create_subscription(
                OccupancyGrid,
                'map',
                self.map_callback,
                qos_profile=latching_qos)
        else:
            # Get map from map server
            map_client = self.create_client(GetMap, 'map_server/map')
            map_client.wait_for_service()
            self.get_clock().sleep_for(Duration(seconds=1.0))
            future = map_client.call_async(GetMap.Request())
            rclpy.spin_until_future_complete(self, future)
            if future.result() is None:
                self.get_logger().error('Map service call failed')
                return
            with self.lock:
                # Initialize the ICP map points
                self.ogm = OccupancyGridMap.from_msg(future.result().map)
                self.have_map = True
                self.initialize_icp(future.result().map)

        # TF information
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # Time delay to account for delay in transforms
        self.tf_time_travel = \
            Duration(seconds=self.get_parameter('time_travel').get_parameter_value().double_value)

        # Make sure the transformation between odom and robot exists
        ready = False
        while not ready:
            self.get_clock().sleep_for(Duration(seconds=0.1))
            with self.lock:
                ready = self.sensor_frame_id is not None
        future = self.tf_listener.buffer.wait_for_transform_async(
            self.sensor_frame_id,
            self.tf_map_odom.child_frame_id,
            Time())
        rclpy.spin_until_future_complete(self, future)
        self.have_transform = True
        self.get_logger().info(
            f'Got transform from {self.tf_map_odom.child_frame_id} to {self.sensor_frame_id}')

        # Publish the initial transform
        self.publish_map_odom_tf(self.get_clock().now().to_msg())

    def initialize_icp(self, map: OccupancyGrid) -> None:
        """
        Initialize the ICP algorithm with the given map.

        Inputs:
            map: the occupancy grid map (nav_msgs/OccupancyGrid)
        Outputs:
            None

        This function should:
            1. Set the map tolerance to be the map cell size
            2. Set the transformation frame to be the map frame
            3. Extract the locations of all objects in the map
            4. Use those points to initialize the ICP using the set_map_points method
        """
        # Set map tolerance to be the map cell size
        self.tolerance = map.info.resolution

        # Set transformation frame to be the map frame
        self.tf_map_odom.header.frame_id = map.header.frame_id

        ##### YOUR CODE STARTS HERE ##### # noqa: E266
        # TODO Extract the locations of all objects in the map
        self.get_logger().info(
            f"[ICP INIT] ogm boundary={self.ogm.boundary}, shape={self.ogm.data.shape}, "
            f"frame={self.ogm.frame_id}"
        )
        pts_xy = self.ogm.where_occupied(format='xy', threshold=50)
        # self.get_logger().info(f"[ICP INIT] points: {pts_xy.shape[0]} and {pts_xy.shape[1]}")
        # origin_tf = Transform()
        # origin_tf.translation.x = map.info.origin.position.x
        # origin_tf.translation.y = map.info.origin.position.y
        # origin_tf.translation.z = map.info.origin.position.z
        # origin_tf.rotation = map.info.origin.orientation
        # origin_T = tf2d.transform2homogeneous(origin_tf)
        # pts_h = np.vstack((pts_xy.T, np.ones((1, pts_xy.shape[0]))))  # 3×K
        # self.get_logger().info(f"[ICP INIT] pts_h: {pts_h.shape[0]} and {pts_h.shape[1]}")
        # pts_map = (origin_T @ pts_h)[:2, :].astype(np.float32)        # 2×K
        # self.get_logger().info(f"[ICP INIT] pts_map: {pts_map.shape[0]} and {pts_map.shape[1]}")
        pts_map = pts_xy.T.astype(np.float32)
        # TODO use those points to inialize the ICP using the set_map_points method
        self.icp.set_map_points(pts_map)
        ##### YOUR CODE ENDS HERE   ##### # noqa: E266

    def map_callback(self, msg: OccupancyGrid) -> None:
        """
        Save the map and use it to initialize the ICP map points.

        Inputs:
            msg: the occupancy grid map (nav_msgs/OccupancyGrid)
        Outputs:
            None
        """
        with self.lock:
            # self.ogm = OccupancyGridMap(msg)
            self.ogm = OccupancyGridMap.from_msg(msg)
            self.have_map = True
            self.initialize_icp(msg)

    def publish_map_odom_tf(self, time: TimeMsg) -> None:
        """
        Update the current transform from the map to odom frames stored in self.tf_map_odom.

        Inputs:
            time: the time to stamp the transform with (builtin_interfaces.msg.Time)
        Outputs:
            None

        This function should:
            1. Fill in the header stamp with the given time
            2. Fill in the translation and rotation based on the current pose
               (stored in self.pose)
            3. Publish the transform using self.tf_broadcaster
        """
        ##### YOUR CODE STARTS HERE ##### # noqa: E266
        # TODO Fill in the current time
        self.tf_map_odom.header.stamp = time.to_msg()

        # TODO Fill in the current transform (using the x, y, theta in self.pose)
        x, y, theta = self.pose
        self.tf_map_odom.transform = tf2d.xyt2transform(x, y, theta)
        ##### YOUR CODE ENDS HERE   ##### # noqa: E266

        # Publish the transform
        self.tf_broadcaster.sendTransform(self.tf_map_odom)

    def publish_pose(self, stamp: TimeMsg) -> None:
        """
        Publish the current pose as a PoseWithCovarianceStamped message.

        Inputs:
            stamp: the time to stamp the pose message with (builtin_interfaces.msg.Time)
        Outputs:
            None

        This function should:
            1. Fill in the pose message
            2. Publish the message using self.pose_pub
        """
        # Initialize pose message
        pose_msg = PoseWithCovarianceStamped()
        # Initialize covariance to reasonable values
        pose_msg.pose.covariance = np.diag((0.1, 0.1, 0.0, 0.0, 0.0, 0.2)).flatten().tolist()
        ##### YOUR CODE STARTS HERE ##### # noqa: E266
        # TODO Fill in the pose message
        pass

        # TODO Publish the message using self.pose_pub
        pass
        ##### YOUR CODE ENDS HERE   ##### # noqa: E266

    def initialpose_callback(self, msg: PoseWithCovarianceStamped) -> None:
        """Process an incoming initial pose message."""
        # Convert from a Pose to a Transform
        T = Transform(translation=msg.pose.pose.position,
                      rotation=msg.pose.pose.orientation)
        # Update the pose
        self.pose = tf2d.transform2xyt(T)
        # Publish the pose
        self.publish_map_odom_tf(msg.header.stamp)
        self.publish_pose(msg.header.stamp)

    def scan_callback(self, msg: LaserScan) -> None:
        """
        Process the lidar scan and use ICP to perform localization.

        Inputs:
            msg: the lidar scan (sensor_msgs/LaserScan)
        Outputs:
            None

        This function should:
            1. Convert the lidar points to (x, y) coordinates in the *odom* frame
            2. Use ICP to find the pose of the odom frame with respect to the map frame
            3. Update self.pose with the new pose
        """
        # Make sure the node is ready
        with self.lock:
            if self.sensor_frame_id is None:
                self.sensor_frame_id = msg.header.frame_id
        if not self.have_map:
            self.get_logger().warning('No map received yet', once=True)
            return
        if not self.have_transform:
            odom_frame = self.tf_map_odom.child_frame_id
            self.get_logger().warning(
                f'Waiting for transform from {odom_frame} to {self.sensor_frame_id}',
                once=True)
            return
        ##### YOUR CODE STARTS HERE ##### # noqa: E266
        # TODO Convert lidar all points to (x,y)
        # NOTE You should only keep points that are between the minimum and maximum range
        ranges = np.asarray(msg.ranges, dtype=np.float32)
        n = ranges.shape[0]
        angles = msg.angle_min + np.arange(n, dtype=np.float32) * msg.angle_increment
        valid = np.isfinite(ranges) & (ranges >= msg.range_min) & (ranges <= msg.range_max)
        r = ranges[valid]
        a = angles[valid]
        x_l = r * np.cos(a)
        y_l = r * np.sin(a)
        pts_lidar = np.vstack((x_l, y_l))
        if pts_lidar.size == 0:
            return

        # TODO Lookup transformation from odom to lidar coordiante frame
        odom_frame = self.tf_map_odom.child_frame_id
        stamp = Time.from_msg(msg.header.stamp) - self.tf_time_travel

        try:
            if not self.tf_buffer.can_transform(odom_frame, self.sensor_frame_id, stamp):
                # self.get_logger().warn("TF not ready, using latest")
                tf_odom_lidar = self.tf_buffer.lookup_transform(
                    odom_frame, self.sensor_frame_id, rclpy.time.Time())
            else:
                tf_odom_lidar = self.tf_buffer.lookup_transform(
                    odom_frame, self.sensor_frame_id, stamp)
        except Exception as e:
            self.get_logger().warn(f"TF lookup failed ({type(e).__name__}): {e}")
            tf_odom_lidar = self.tf_buffer.lookup_transform(
                odom_frame, self.sensor_frame_id, rclpy.time.Time())

        # tf_odom_lidar = self.tf_buffer.lookup_transform(
        #     odom_frame,               # target frame
        #     self.sensor_frame_id,     # source frame
        #     stamp)

        # TODO Use the transformation to transform the lidar points into the odom coordinate frame
        T_odom_lidar = tf2d.transform2homogeneous(tf_odom_lidar.transform)
        pts_h = np.vstack((pts_lidar, np.ones((1, pts_lidar.shape[1]))))
        pts_odom = (T_odom_lidar @ pts_h)[:2, :]

        # TODO Use ICP to find pose of odom with respect to the map
        init_T = tf2d.xyt2homogeneous(*self.pose)
        with self.lock:
            # NOTE need to do this inside of the lock to ensure correct operation
            T_map_odom, mean_error, iters = self.icp.icp(
                pts_odom.astype(np.float32),
                init_pose=init_T.astype(np.float32),
                max_iterations=50,
                tolerance=self.tolerance)

        # TODO Update self.pose using the transformation found by ICP
        self.pose = tf2d.homogeneous2xyt(T_map_odom)
        ##### YOUR CODE ENDS HERE   ##### # noqa: E266

        # Publish transform
        self.publish_map_odom_tf(msg.header.stamp)
        self.publish_pose(msg.header.stamp)


def main(args=None):
    # Start up ROS2
    rclpy.init(args=args)

    # Create the node
    icp_node = ICPLocalizationNode()

    # Let the node run until it is killed
    rclpy.spin(icp_node)

    # Clean up the node and stop ROS2
    icp_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
