#!/usr/bin/env python3
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from rclpy.qos import QoSProfile, QoSDurabilityPolicy
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
import tf2_ros

from nav_msgs.msg import OccupancyGrid
from nav_msgs.srv import GetMap

from occupancy_grid import MapConversions
from tb3_utils import TB3Params
from transform2d_utils import lookup_transform

from collision import Poly, Vector, collide
import numpy as np
from scipy.spatial import KDTree
from threading import Lock


class CollisionDetectionNode(Node):

    def __init__(self) -> None:
        super().__init__('collision_detection')

        # Declare parameters
        self.declare_parameter('tb3_model',
                               descriptor=ParameterDescriptor(
                                   type=ParameterType.PARAMETER_STRING))
        self.declare_parameter('robot_frame_id',
                               value='base_footprint',
                               descriptor=ParameterDescriptor(
                                   type=ParameterType.PARAMETER_STRING))
        self.declare_parameter('occupancy_threshold',
                               value=50,
                               descriptor=ParameterDescriptor(
                                   type=ParameterType.PARAMETER_INTEGER))
        self.declare_parameter('use_map_topic',
                               value=False,
                               descriptor=ParameterDescriptor(
                                   type=ParameterType.PARAMETER_BOOL))
        self.declare_parameter('map_conversions_implemented',
                               value=False,
                               descriptor=ParameterDescriptor(
                                   type=ParameterType.PARAMETER_BOOL))
        self.declare_parameter('rate',
                               value=10.0,
                               descriptor=ParameterDescriptor(
                                   type=ParameterType.PARAMETER_DOUBLE))

        # Set required data
        self.have_map = False
        self.have_transform = False

        # Check if MapConversions is implemented
        self.map_conversions_implemented = \
            self.get_parameter('map_conversions_implemented').get_parameter_value().bool_value

        # Load TB3 Params
        self.params = TB3Params(self.get_parameter('tb3_model').get_parameter_value().string_value)
        self.robot_frame_id = \
            self.get_parameter('robot_frame_id').get_parameter_value().string_value

        # Robot polygon
        self.robot_poly = Poly(Vector(0, 0), [Vector(x, y) for x, y in self.params.footprint], 0)
        self.radius = np.max([np.linalg.norm(np.array(p)) for p in self.params.footprint])

        # Params
        self.occupancy_threshold = \
            self.get_parameter('occupancy_threshold').get_parameter_value().integer_value

        # Map
        self.lock = Lock()
        self.map_frame_id = None
        if self.get_parameter('use_map_topic').get_parameter_value().bool_value:
            latching_qos = QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
            self.map_sub = self.create_subscription(
                OccupancyGrid,
                'map',
                self.map_callback,
                qos_profile=latching_qos)
        else:
            map_client = self.create_client(GetMap, 'map_server/map')
            map_client.wait_for_service()
            self.get_clock().sleep_for(Duration(seconds=1.0))
            future = map_client.call_async(GetMap.Request())
            rclpy.spin_until_future_complete(self, future)
            self.prepare_map(future.result().map)

        # Timer
        rate = self.get_parameter('rate').get_parameter_value().double_value
        self.timer = self.create_timer(1/rate, self.timer_callback)

        # TF
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Make sure the transformation between the map and robot exists
        while rclpy.ok() and self.map_frame_id is None:
            self.get_logger().info('Waiting for map to be received', once=True)
            self.get_clock().sleep_for(Duration(seconds=0.1))
        future = self.tf_listener.buffer.wait_for_transform_async(
            self.robot_frame_id,
            self.map_frame_id,
            Time())
        rclpy.spin_until_future_complete(self, future)
        self.have_transform = True
        self.get_logger().info(f'Got transform from {self.map_frame_id} to {self.robot_frame_id}')

    def map_callback(self, msg: OccupancyGrid) -> None:
        self.prepare_map(msg)

    def prepare_map(self, msg: OccupancyGrid) -> None:
        """Prepare the map for collision detection."""
        with self.lock:
            self.get_logger().info('Received map')
            # Store map info
            self.map_frame_id = msg.header.frame_id
            self.resolution = msg.info.resolution
            if not self.map_conversions_implemented:
                return
            # Find objects in map
            objs = np.argwhere(np.array(msg.data) > self.occupancy_threshold)
            mc = MapConversions.from_msg(msg)
            xx, yy = mc.ind2xy(objs)
            # Build KD tree
            self.kdtree = KDTree(np.hstack((xx, yy)))
            # Build collision polygons
            v = Vector
            r = self.resolution/2
            box = [v(-r, -r), v(-r, r), v(r, r), v(r, -r)]
            self.map_boxes = [Poly(v(x, y), box) for x, y in zip(xx, yy)]
            self.have_map = True

    def timer_callback(self):
        """Check for collisions."""
        # Ensure we have a map
        with self.lock:
            if not self.map_conversions_implemented:
                self.get_logger().warning('Map conversions not implemented', once=True)
                return
            if not self.have_map:
                self.get_logger().warning('No map received yet', once=True)
                return
            if not self.have_transform:
                self.get_logger().warning(
                    f'Waiting for transform from {self.map_frame_id} to {self.robot_frame_id}',
                    once=True)
                return
            self.get_logger().info('Checking for collisions with the map', once=True)
            # Get pose of the robot in the map frame
            try:
                x, y, t = lookup_transform(
                    self.tf_buffer,
                    self.map_frame_id,
                    self.robot_frame_id,
                    format='xyt')
            except Exception as err:
                self.get_logger().warn(err)
                return

            # Get objects nearby the robot's current position
            ii = self.kdtree.query_ball_point((x, y), self.radius+self.resolution)
            # Check for collisions with those nearby boxes
            self.robot_poly.pos.x = x
            self.robot_poly.pos.y = y
            self.robot_poly.angle = t
            for i in ii:
                if collide(self.robot_poly, self.map_boxes[i]):
                    raise SystemExit


def main(args=None):
    # Start up ROS2
    rclpy.init(args=args)

    # Create the node
    cdnode = CollisionDetectionNode()

    # Let the node run until it is killed
    try:
        rclpy.spin(cdnode)
    except SystemExit:
        cdnode.get_logger().error('Turtlebot collided with something in the map')

    # Clean up the node and stop ROS2
    cdnode.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
    