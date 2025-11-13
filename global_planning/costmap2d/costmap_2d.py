from rclpy import ok, spin_once
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.duration import Duration
from rclpy.lifecycle import (
    LifecycleNode,
    State,
    TransitionCallbackReturn,
)
from rclpy import qos
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
from tf2_ros import Buffer, TransformListener

from geometry_msgs.msg import Point, Polygon, PolygonStamped
from lifecycle_msgs.msg import State as StateMsg
from lifecycle_msgs.srv import GetState
from nav_msgs.msg import OccupancyGrid
from nav_msgs.srv import GetMap
from map_msgs.msg import OccupancyGridUpdate
from nav2_msgs.msg import Costmap
from nav2_msgs.srv import (
    ClearCostmapExceptRegion,
    ClearCostmapAroundRobot,
    ClearEntireCostmap,
    GetCostmap)

from tb3_utils import TB3Params

import cv2 as cv
from threading import Lock
import numpy as np


class Costmap2D(LifecycleNode):
    def __init__(self, use_sim_time: bool = False) -> None:
        super().__init__(
            'global_costmap',
            namespace='global_costmap',
            use_global_arguments=False)
        self.active = False

        # Declare parameters
        self.declare_parameter(
            'tb3_model',
            'burger',
            descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description='TurtleBot3 model type (burger, waffle, waffle_pi)'))
        self.declare_parameter(
            'occ_threshold',
            50,
            descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_INTEGER,
                description='Occupancy threshold for binarizing the map'))
        self.declare_parameter(
            'always_send_full_costmap',
            False,
            descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_BOOL,
                description='Whether to always send the full costmap on updates'))
        self.declare_parameter(
            'footprint_padding',
            0.0,
            descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_DOUBLE,
                description='Padding to add to the robot footprint'))
        self.declare_parameter(
            'robot_base_frame',
            'base_footprint',
            descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description='The robot base frame for the costmap'))
        self.declare_parameter(
            'publish_frequency',
            1.0,
            descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_DOUBLE,
                description='Frequency at which to publish the costmap'))

    def on_configure(self, state: State) -> TransitionCallbackReturn:
        super().on_configure(state)
        self.get_logger().info("Configuring Costmap2D node...")

        # Get parameters
        self.get_logger().info(f"{self.get_parameter('tb3_model').value}")
        self.robot_params = TB3Params(self.get_parameter('tb3_model').value)
        self.always_send_full_costmap = self.get_parameter('always_send_full_costmap').value
        self.footprint_padding = self.get_parameter('footprint_padding').value
        self.robot_base_frame = self.get_parameter('robot_base_frame').value
        self.publish_frequency = self.get_parameter('publish_frequency').value
        self.occ_threshold = self.get_parameter('occ_threshold').value

        # Create robot footprint polygon
        self.robot_footprint = Polygon()
        for p in self.robot_params.footprint:
            self.robot_footprint.points.append(Point(x=p[0], y=p[1], z=0.0))
        # Adjust footprint for padding
        if self.footprint_padding != 0.0:
            centroid = np.mean(np.array(self.robot_params.footprint), axis=0)
            for p in self.robot_footprint.points:
                p.x = centroid[0] + (p.x - centroid[0]) * (1.0 + self.footprint_padding)
                p.y = centroid[1] + (p.y - centroid[1]) * (1.0 + self.footprint_padding)

        # Set up publishers and subscribers
        latching_qos = qos.QoSProfile(
            history=qos.QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            durability=qos.QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=qos.QoSReliabilityPolicy.RELIABLE)
        self.footprint_pub = self.create_lifecycle_publisher(
            PolygonStamped,
            'robot_footprint',
            10)
        self.costmap_pub = self.create_lifecycle_publisher(
            OccupancyGrid,
            'costmap',
            latching_qos)
        self.costmap_update_pub = self.create_lifecycle_publisher(
            OccupancyGridUpdate,
            'costmap_updates',
            latching_qos)
        self.costmap_raw_pub = self.create_lifecycle_publisher(
            Costmap,
            'costmap_raw',
            latching_qos)

        # tf_buffer and tf_listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Create services
        self.get_costmap_server = self.create_service(
            GetCostmap,
            'get_costmap',
            self.get_costmap_callback)
        self.clear_costmap_around_robot_server = self.create_service(
            ClearCostmapAroundRobot,
            'clear_around_costmap',
            self.clear_around_robot_callback)
        self.clear_costmap_except_region_server = self.create_service(
            ClearCostmapExceptRegion,
            'clear_except_costmap',
            self.clear_except_region_callback)
        self.clear_entire_costmap_server = self.create_service(
            ClearEntireCostmap,
            'clear_entirely_global_costmap',
            self.clear_entire_callback)

        self.get_logger().info("Costmap2D configured.")
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: State) -> TransitionCallbackReturn:
        super().on_activate(state)
        self.get_logger().info("Activating Costmap2D node...")
        # Get map
        self.lock = Lock()
        self.have_map = False

        # Make sure map client is active
        if not self.ensure_server_active('/map_server'):
            self.get_logger().error("Map server is not active.")
            return TransitionCallbackReturn.FAILURE

        # Make sure GetMap service is available
        map_client = self.create_client(
            GetMap,
            '/map_server/map',
            callback_group=MutuallyExclusiveCallbackGroup())
        map_client.wait_for_service()

        # Get the map
        future = map_client.call_async(GetMap.Request())
        future.add_done_callback(lambda f: self.map_callback(f.result().map))
        with self.lock:
            ready = self.have_map
        while ok() and not ready:
            self.get_logger().info('Waiting for map to be received...', once=True)
            self.get_clock().sleep_for(Duration(seconds=0.01))
            spin_once(future)
            with self.lock:
                ready = self.have_map

        # Set active flag
        self.active = True

        self.get_logger().info("Costmap2D activated.")
        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state: State) -> TransitionCallbackReturn:
        self.active = False
        self.get_logger().info("Deactivating Costmap2D node.")
        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state: State) -> TransitionCallbackReturn:
        self.costmap = None
        self.get_logger().info("Cleaning up Costmap2D node.")
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("Shutting down Costmap2D node.")
        return TransitionCallbackReturn.SUCCESS

    def ensure_server_active(self, node: str) -> bool:
        """Ensure that the given lifecycle node is active."""
        client = self.create_client(
            GetState,
            node + '/get_state',
            callback_group=MutuallyExclusiveCallbackGroup())
        while not client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info(f'{client.srv_name} not available, waiting...', once=True)
        # Initialize ready dictionary for callback
        self.ready = False
        future = client.call_async(GetState.Request())
        future.add_done_callback(self.is_active)
        with self.lock:
            ready = self.ready
        while ok() and not ready:
            self.get_clock().sleep_for(Duration(seconds=0.01))
            spin_once(self)
            with self.lock:
                ready = self.ready
        return True

    def is_active(self, future):
        """Ensure that the given lifecycle node is active."""
        self.get_logger().info(f'Future done for {future.result()}...')
        if future.result() is not None:
            current_state = future.result().current_state
            if current_state.id == StateMsg.PRIMARY_STATE_ACTIVE:
                with self.lock:
                    self.ready = True
                return
            else:
                self.get_logger().warn(
                    f'Not not ready yet (current state: {current_state.label}), waiting...',  # noqa: E501
                    once=True)
                return
        else:
            self.get_logger().error('Service call failed')
            return

    def map_callback(self, msg: OccupancyGrid) -> None:
        """
        Convert the input map to a costmap to use for planning.

        Inputs:
            map: OccupancyGrid map from ROS
        Outputs:
            costmap: Costmap created from the input map

        Key steps:
            1) Binarize the map
                a) Keep all unknown values (-1) in place
                b) Make all values < self.occ_threshold to 0
                c) Make all values >= self.occ_threshold to 100
            2) Inflate obstacles by the robot size
            3) Use the inflated map to create a PRM
        """
        self.get_logger().info('Received map for costmap generation.')
        costmap = Costmap()
        costmap.header = msg.header
        costmap.metadata.map_load_time = msg.info.map_load_time
        costmap.metadata.resolution = msg.info.resolution
        costmap.metadata.size_x = msg.info.width
        costmap.metadata.size_y = msg.info.height
        costmap.metadata.origin = msg.info.origin

        # Convert map to numpy array
        img = np.array(msg.data, dtype=np.int8)
        ind_unknown = img == -1  # save indices of unknown cells

        # Shift map values up by 1 to ensure non-negative
        img += 1  # put unknown at 0
        thresh = self.occ_threshold + 1
        max_val = np.uint8(100) + 1
        min_val = np.uint8(0) + 1

        # Binarize occupancy grid
        img = img.astype(np.uint8)  # convert to uint8 to use dilate
        img[np.logical_and(img >= thresh, img <= max_val)] = max_val
        img[np.logical_and(img >= min_val, img < thresh)] = min_val

        # Convert to 2D image
        img = img.reshape((msg.info.height, msg.info.width))

        # Get robot kernel (i.e., shape)
        r = np.int8(np.ceil(self.robot_params.robot_radius / msg.info.resolution))
        robot_img = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2*r+1, 2*r+1), (r, r))

        # Inflate obstacles using dilate function
        img = cv.dilate(img, robot_img)

        # Update map data
        img = img.flatten().astype(np.int8) - 1  # shift values back down
        img[np.logical_and(ind_unknown, img == 0)] = -1  # put unknown back to -1
        img[img < 0] = 127  # unknown to 127 for costmap
        costmap.data = img.astype(np.uint8).tolist()

        with self.lock:
            self.costmap = costmap
            self.have_map = True
            self.get_logger().info('Costmap generated successfully.')

    def get_costmap(self) -> Costmap:
        """Get the current costmap."""
        with self.lock:
            return self.costmap

    def get_map_frame_id(self) -> str:
        """Get the frame ID of the map."""
        with self.lock:
            if self.have_map:
                return self.costmap.header.frame_id
            else:
                return ''

    def get_costmap_callback(self, request, response):
        """Return costmap."""
        if not self.active:
            self.get_logger().error('Costmap2D node is not active')
            return response
        if not self.have_map:
            self.get_logger().error('No map received yet')
            return response
        # Return the costmap
        response.costmap = self.costmap
        return response

    def clear_around_robot_callback(self, request, response):
        """Clear the costmap around the robot."""
        # Placeholder for ClearCostmapAroundRobot service callback
        pass
        return response

    def clear_except_region_callback(self, request, response):
        """Clear the costmap except for a region."""
        pass
        return response

    def clear_entire_callback(self, request, response):
        """Clear the entire costmap."""
        pass
        return response
