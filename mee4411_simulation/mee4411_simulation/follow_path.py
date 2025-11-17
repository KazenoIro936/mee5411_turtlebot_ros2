# Adapted from nav2_simple_commander example_follow_path.py example
# https://github.com/ros-navigation/navigation2/blob/main/nav2_simple_commander/nav2_simple_commander/example_follow_path.py

from geometry_msgs.msg import PoseStamped
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult

from rclpy import init, spin
from rcl_interfaces.msg import ParameterDescriptor, ParameterType

"""Basic navigation demo to follow a given path after smoothing"""


class FollowPathNode(BasicNavigator):
    def __init__(self):
        super().__init__(node_name='follow_path_node')

        # Declare parameters
        self.declare_parameter(
            'localizer',
            'amcl',
            descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description='Choose localization method ("amcl", "fake_localization", "icp_node")'
            )
        )
        self.declare_parameter(
            'navigator',
            'bt_navigator',
            descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description='Choose navigation method ("bt_navigator", etc.)'))

        # Subscribers
        self.goal_sub = self.create_subscription(
            PoseStamped,
            'goal_pose',
            self.goal_callback,
            1)

        # Wait for nav2 to fully activate
        self.waitUntilNav2Active(
            localizer=self.get_parameter('localizer').get_parameter_value().string_value,
            navigator=self.get_parameter('navigator').get_parameter_value().string_value)
        self._waitForInitialPose()

    def goal_callback(self, msg: PoseStamped) -> None:
        """Handle a new goal pose from the subscriber."""
        print(f'Received new goal pose: {msg.pose.position.x}, {msg.pose.position.y}')

        # Get the current robot pose
        initial_pose = PoseStamped()

        # Get the path, smooth it
        path = self.getPath(initial_pose, msg)
        smoothed_path = self.smoothPath(path)

        # Follow path
        follow_path_task = self.followPath(smoothed_path)

        while not self.isTaskComplete(task=follow_path_task):
            # Do something with the feedback
            feedback = self.getFeedback(task=follow_path_task)
            if feedback:
                self.get_logger().info(
                    'Estimated distance remaining to goal position: '
                    + f'{feedback.tracking_feedback.distance_to_goal:.3f}'
                    + '\nCurrent speed of the robot: '
                    + f'{feedback.tracking_feedback.speed:.3f}'
                )

        # Do something depending on the return code
        result = self.getResult()
        if result == TaskResult.SUCCEEDED:
            self.get_logger().info('Goal succeeded!')
        elif result == TaskResult.CANCELED:
            self.get_logger().warn('Goal was canceled!')
        elif result == TaskResult.FAILED:
            (error_code, error_msg) = self.getTaskError()
            self.get_logger().error(f'Goal failed! {error_code}: {error_msg}')
        else:
            self.get_logger().error('Goal has an invalid return status!')


def main(args=None):
    # Start up ROS2
    init(args=args)

    # Create the node
    follow_node = FollowPathNode()

    # Let the node run until it is killed
    spin(follow_node)

    # Clean up the node and stop ROS2
    follow_node.lifecycleShutdown()
    exit(0)


if __name__ == '__main__':
    main()
