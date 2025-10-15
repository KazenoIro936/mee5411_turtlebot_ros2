#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid  # noqa: F401
from rclpy.qos import QoSProfile, QoSDurabilityPolicy
from .map_conversions import MapConversions  # noqa: F401
from .occupancy_grid_map import OccupancyGridMap  # noqa: F401
import numpy as np


class OccupancyGridNode(Node):

    def __init__(self):
        super().__init__('occupancy_grid')
        ##### YOUR CODE STARTS HERE ##### # noqa: E266
        # TODO Set up the ROS node
        qos = QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        # TODO Set up the ROS publisher for the occupancy grid map
        self.publisher = self.create_publisher(OccupancyGrid, 'map', qos)
        # TODO Read in the map information from the ROS parameter server
        msg = OccupancyGrid()
        self.declare_parameter('frame_id', '')
        self.declare_parameter('resolution', 0.0)
        # self.declare_parameter('filename', '')
        # self.declare_parameter('height', 0)
        # self.declare_parameter('width', 0)
        self.declare_parameter('boundary', [0.0, 0.0, 0.0, 0.0])
        self.declare_parameter('blocks', [0.0, 0.0, 0.0, 0.0])
        # self.declare_parameter('xmin', 0.0)
        # self.declare_parameter('ymin', 0.0)
        # msg.header.frame_id = self.get_parameter('frame_id').value
        # msg.info.resolution = self.get_parameter('resolution').value
        # boundary = self.get_parameter('boundary').value
        # msg.info.width = int(boundary[2] - boundary[0])
        # msg.info.height = int(boundary[3] - boundary[1])
        # msg.info.origin.position.x = boundary[0]
        # msg.info.origin.position.y = boundary[1]
        frame_id = self.get_parameter('frame_id').value
        resolution = self.get_parameter('resolution').value
        boundary = self.get_parameter('boundary').value
        blocks_flat = self.get_parameter('blocks').value
        self.get_logger().info(f"blocks_flat (len={len(blocks_flat)}):{blocks_flat}")
        ogm = OccupancyGridMap(boundary, resolution, frame_id)
        need_x = (abs(boundary[2] - boundary[0]) % resolution) > 1e-6
        need_y = (abs(boundary[3] - boundary[1]) % resolution) > 1e-6
        if need_x:
            ogm.add_block(np.array([boundary[2], boundary[1], boundary[2], boundary[3]]))
        if need_y:
            ogm.add_block(np.array([boundary[0], boundary[3], boundary[2], boundary[3]]))
        if blocks_flat:  # only if non-empty
            blocks = np.array(blocks_flat, dtype=float).reshape(-1, 4)
        for bx0, by0, bx1, by1 in blocks:
            ogm.add_block(np.array([bx0, by0, bx1, by1]))
        # TODO Create an OccupancyGridMap based on the provided data using occupancy_grid_utils
        msg = ogm.to_msg(self.get_clock().now())

        # TODO Create and publish a nav_msgs/OccupancyGrid msg
        self.publisher.publish(msg)
        # self.get_logger().info(f"Publishing OccupancyGrid:
        # "f"frame={msg.header.frame_id},resolution={msg.info.resolution}")
        # self.get_logger().info(f"width: {msg.info.width}, height:{msg.info.height}")
        # self.get_logger().info(f"origin: x={msg.info.origin.position.x},
        # "f"y={msg.info.origin.position.y}")
        # data_np = np.array(msg.data, dtype=np.int8).reshape(msg.info.height,msg.info.width)
        # with np.printoptions(threshold=np.inf, linewidth=10_000):
        # self.get_logger().info(f"Full OccupancyGrid data:\n{data_np}")
        ##### YOUR CODE ENDS HERE   ##### # noqa: E266


def main(args=None):
    # Start up ROS2
    rclpy.init(args=args)

    # Create the node
    og_node = OccupancyGridNode()

    # Let the node run until it is killed
    rclpy.spin(og_node)

    # Clean up the node and stop ROS2
    og_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
