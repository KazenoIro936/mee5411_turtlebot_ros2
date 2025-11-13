from rclpy import (
    init,
    shutdown,
)
from rclpy.action import ActionServer
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.lifecycle import LifecycleNode, State, TransitionCallbackReturn
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
from rclpy.qos import QoSProfile, QoSDurabilityPolicy

import tf2_ros

from nav2_msgs.action import ComputePathToPose, ComputePathThroughPoses
from visualization_msgs.msg import MarkerArray

from datetime import datetime
import numpy as np

from costmap2d import Costmap2D
from prm import PRM
from transform2d_utils import lookup_transform


class PRMNode(LifecycleNode):
    def __init__(self) -> None:
        LifecycleNode.__init__(self, 'prm_node')

        self.costmap_node = Costmap2D()

        # Declare parameters
        self.declare_parameter(
            'robot_frame',
            'base_footprint',
            descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description='Robot frame ID'))
        self.declare_parameter(
            'show_prm',
            True,
            descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_BOOL,
                description='Whether to publish the PRM graph for visualization'))
        self.declare_parameter(
            'publish_every_n',
            20,
            descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_INTEGER,
                description='Publish the PRM graph every n added points'))
        self.declare_parameter(
            'connection_radius',
            1.0,
            descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_DOUBLE,
                description='Connection radius for the PRM'))
        self.declare_parameter(
            'step_size',
            0.01,
            descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_DOUBLE,
                description='Step size for collision checking in the PRM'))
        self.declare_parameter(
            'num_points',
            1000,
            descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_INTEGER,
                description='Number of points to sample in the PRM'))

    def on_configure(self, state: State) -> TransitionCallbackReturn:
        """Configure the node."""
        super().on_configure(state)
        self.costmap_node.on_configure(state)
        self.get_logger().info(f'{self.get_name()}: Configuring...')

        # Robot parameters
        self.robot_frame_id = self.get_parameter('robot_frame').value
        self.get_logger().info(f'Robot frame ID: {self.robot_frame_id}')

        # PRM
        self.show_prm = self.get_parameter('show_prm').value
        self.get_logger().info(f'Show PRM: {self.show_prm}')
        self.prm_params = {}
        self.prm_params['num_points'] = self.get_parameter('num_points').value
        self.prm_params['connection_radius'] = self.get_parameter('connection_radius').value
        self.prm_params['step_size'] = self.get_parameter('step_size').value
        self.prm_params['publish_every_n'] = self.get_parameter('publish_every_n').value
        self.get_logger().info(f'PRM parameters: {self.prm_params}')

        # Publishers and subscribers
        latching_qos = QoSProfile(
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self.marker_pub = self.create_publisher(
            MarkerArray,
            self.get_fully_qualified_name() + '/graph',
            latching_qos) if self.show_prm else None

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.get_logger().info(f'{self.get_name()}: Configured')
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state) -> TransitionCallbackReturn:
        """Activate the node."""
        super().on_activate(state)
        self.costmap_node.on_activate(state)
        self.get_logger().info(f'{self.get_name()}: Activating...')

        # Prepare PRM
        self.prm = PRM(
            costmap=self.costmap_node.get_costmap(),
            clock=self.get_clock(),
            logger=self.get_logger(),
            publisher=self.marker_pub,
            **self.prm_params)

        # Create action server
        self.action_server = ActionServer(
            self,
            ComputePathToPose,
            'compute_path_to_pose',
            self.path_to_pose_callback)
        self.action_server = ActionServer(
            self,
            ComputePathThroughPoses,
            'compute_path_through_poses',
            self.path_through_poses_callback)

        self.get_logger().info(f'{self.get_name()} activated')
        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state) -> TransitionCallbackReturn:
        """Deactivate the node."""
        super().on_deactivate(state)
        self.costmap_node.on_deactivate(state)
        self.get_logger().info('Deactivating PRM Node...')
        self.action_server.destroy()
        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state) -> TransitionCallbackReturn:
        """Cleanup the node."""
        super().on_cleanup(state)
        self.costmap_node.on_cleanup(state)
        self.get_logger().info('Cleaning up PRM Node...')
        # Reset variables
        self.prm = None

        # Destroy publishers and subscribers
        if self.marker_pub is not None:
            self.marker_pub.destroy()

        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state) -> TransitionCallbackReturn:
        """Shutdown the node."""
        super().on_shutdown(state)
        self.costmap_node.on_shutdown(state)
        self.get_logger().info('Shutting down PRM Node...')
        return TransitionCallbackReturn.SUCCESS

    def path_to_pose_callback(self, goal_handle) -> ComputePathToPose.Result:
        """
        Use the PRM to plan a path from the current pose of the robot to the goal pose.

        Inputs:
            request: Action request (nav2_msgs/Action/ComputePathToPose/Request)
        Outputs:
            response: Action response (nav2_msgs/Action/ComputePathToPose/Response)
        """
        ##### YOUR CODE STARTS HERE ##### # noqa: E266
        # TODO Read the information in the action request, paying attention to initial pose
        # NOTE The use_start field coming in should always be False
        pass

        # NOTE No need to publish feedback since the action definition does not include it

        # TODO Use prm.query to plan a path from the current robot position to the goal
        # NOTE You can optionally choose to add points to the PRM if no path is found
        # NOTE fill in the action response with the planned path and planning time
        result = ComputePathToPose.Result()
        pass
        ##### YOUR CODE ENDS HERE   ##### # noqa: E266
        return result

    def path_through_poses_callback(self, goal_handle) -> ComputePathThroughPoses.Result:
        """
        Use the PRM to plan a path through multiple goal poses.

        Inputs:
            request: Action request (nav2_msgs/Action/ComputePathThroughPoses/Request)
        Outputs:
            response: Action response (nav2_msgs/Action/ComputePathThroughPoses/Response)
        """
        ##### YOUR CODE STARTS HERE ##### # noqa: E266
        # TODO Read the information in the action request, paying attention to initial pose
        # NOTE The use_start field coming in should always be False
        pass

        # TODO Check that the requested data is valid, if not return the proper error code
        #   (see the action response definition for the list of error codes)
        pass

        # NOTE No need to publish feedback since the action definition does not include it

        # TODO Use prm.query to plan a path from the current robot position to the goal
        # NOTE You can optionally choose to add points to the PRM if no path is found
        # NOTE fill in the action response with the planned path and planning time
        result = ComputePathThroughPoses.Result()
        pass
        ##### YOUR CODE ENDS HERE   ##### # noqa: E266
        return result


def main(args=None):
    # Start up ROS2
    init(args=args)

    # Create the node
    prm_node = PRMNode()

    # Create a multi-threaded executor
    executor = MultiThreadedExecutor()
    executor.add_node(prm_node)

    try:
        # Spin the executor to process callbacks from all added nodes
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        # Shutdown and destroy nodes
        executor.shutdown()
        prm_node.destroy_node()
        shutdown()


if __name__ == '__main__':
    main()
