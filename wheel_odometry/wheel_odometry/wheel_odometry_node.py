import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
import tf2_ros

from sensor_msgs.msg import JointState
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry

import numpy as np

from wheel_odometry import TB3Kinematics


class WheelOdometryNode(Node, TB3Kinematics):
    def __init__(self) -> None:
        # Initialize node
        Node.__init__(self, 'wheel_odometry')

        # Declare parameters
        self.declare_parameter('tb3_model',
                               descriptor=ParameterDescriptor(
                                   type=ParameterType.PARAMETER_STRING))
        self.declare_parameter('odom_frame',
                               value='odom',
                               descriptor=ParameterDescriptor(
                                   type=ParameterType.PARAMETER_STRING))
        self.declare_parameter('base_frame',
                               value='base_footprint',
                               descriptor=ParameterDescriptor(
                                   type=ParameterType.PARAMETER_STRING))

        # Initialize kinematics
        robot_model = self.get_parameter('tb3_model').get_parameter_value().string_value
        if robot_model not in ('burger', 'waffle', 'waffle_pi'):
            self.get_logger().error(f'Turtlebot3 model {robot_model} not defined')
        TB3Kinematics.__init__(self, robot_model)

        # Joint states data
        self.prev_joint_states = None

        # Odometry message
        self.odom_msg = Odometry()
        ##### YOUR CODE STARTS HERE ##### # noqa: E266
        # TODO Initialize the coordinates frames in the odometry message
        self.odom_frame = self.get_parameter('odom_frame').get_parameter_value().string_value
        self.base_frame = self.get_parameter('base_frame').get_parameter_value().string_value
        self.odom_msg.header.frame_id = self.odom_frame
        self.odom_msg.child_frame_id = self.base_frame
        self.odom_msg.pose.pose.position.x = 0.0
        self.odom_msg.pose.pose.position.y = 0.0
        self.odom_msg.pose.pose.position.z = 0.0
        self.odom_msg.pose.pose.orientation.x = 0.0
        self.odom_msg.pose.pose.orientation.y = 0.0
        self.odom_msg.pose.pose.orientation.z = 0.0
        self.odom_msg.pose.pose.orientation.w = 1.0
        self.odom_msg.twist.twist.linear.x = 0.0
        self.odom_msg.twist.twist.linear.y = 0.0
        self.odom_msg.twist.twist.linear.z = 0.0
        self.odom_msg.twist.twist.angular.x = 0.0
        self.odom_msg.twist.twist.angular.y = 0.0
        self.odom_msg.twist.twist.angular.z = 0.0
        ##### YOUR CODE ENDS HERE   ##### # noqa: E266
        self.odom_msg.pose.covariance = np.diag((0.1, 0.1, 1e6, 1e6, 1e6, 0.2)).flatten().tolist()
        self.odom_msg.twist.covariance = np.diag((0.1, 0.1, 1e6, 1e6, 1e6, 0.2)).flatten().tolist()

        # Robot pose in the odom frame in (x, y, theta) format
        self.pose = [0.0, 0.0, 0.0]  # (x, y, theta)

        # Publishers
        self.odom_pub = self.create_publisher(Odometry, 'odom', 100)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # Subscribers
        self.joint_states_sub = self.create_subscription(
            JointState,
            'joint_states',
            self.joint_states_callback,
            100)

    def joint_states_callback(self, msg: JointState) -> None:
        """
        Save the new joint states messages and update the odometry.

        Inputs:
            msg     new joint states message (sensor_msgs/msg/JointState)
        Outputs:
            None

        This function should:
            1. If this is the first message received, store it and return
            2. Otherwise, update the odometry and tf messages based on the new and previous
               joint states messages
            3. Store the new message for future use
        """
        # Check if first message received
        if self.prev_joint_states is None:
            self.prev_joint_states = msg
            return

        # Update and send messages
        self.update_odometry(msg)
        self.update_tf()

        # Save message for future use
        self.prev_joint_states = msg

    def update_odometry(self, new_joint_states: JointState) -> None:
        """
        Update the odometry message based on the new and previous joint states.

        Inputs:
            new_joint_states    new joint states (sensor_msgs/msg/JointState)
        Outputs:
            None

        This function should:
            1. Calculate the change in wheel angles and time using the new
               and previous joint states
            2. Calculate the displacement of the robot based on the change
               in wheel angles
            3. Compute the new pose of the robot based on the previous pose
               and the displacement
            4. Update the odometry message (stored in self.odom) based on
               the new time, pose, and velocity
            5. Publish the odometry message
        """
        ##### YOUR CODE STARTS HERE ##### # noqa: E266
        # TODO Calculate change in wheel angles using new msg and self.prev_joint_states
        delta_wheel_l, delta_wheel_r, delta_time = self.calculate_wheel_change(
            new_joint_states, self.prev_joint_states)

        # TODO Calculate displacement
        delta_s, delta_theta = self.calculate_displacement(delta_wheel_l, delta_wheel_r)

        # TODO Compute new pose
        self.pose = self.calculate_pose(self.pose, delta_s, delta_theta)

        # TODO Update odometry message (stored in self.odom) based on new time, pose, and velocity
        self.odom_msg.header.stamp = new_joint_states.header.stamp
        self.odom_msg.pose.pose.position.x = self.pose[0]
        self.odom_msg.pose.pose.position.y = self.pose[1]
        self.odom_msg.pose.pose.position.z = 0.0
        self.odom_msg.pose.pose.orientation.x = 0.0
        self.odom_msg.pose.pose.orientation.y = 0.0
        self.odom_msg.pose.pose.orientation.z = np.sin(self.pose[2] / 2.0)
        self.odom_msg.pose.pose.orientation.w = np.cos(self.pose[2] / 2.0)
        self.odom_msg.twist.twist.linear.x = delta_s / delta_time
        self.odom_msg.twist.twist.linear.y = 0.0
        self.odom_msg.twist.twist.linear.z = 0.0
        self.odom_msg.twist.twist.angular.x = 0.0
        self.odom_msg.twist.twist.angular.y = 0.0
        self.odom_msg.twist.twist.angular.z = delta_theta / delta_time
        ##### YOUR CODE ENDS HERE   ##### # noqa: E266
        # Publish odometry message
        self.odom_pub.publish(self.odom_msg)

    def update_tf(self) -> None:
        """
        Update and publish the transformation from odom to base frames.

        Inputs:
            None
        Outputs:
            None

        This function should:
            1. Fill in a TransformStamped message based on the current time,
               the odom_frame and base_frame parameters, and the current pose
               of the robot (stored in self.pose)
            2. Broadcast the transform using self.tf_broadcaster
        """
        # Initialize empty transform
        odom_tf = TransformStamped()

        ##### YOUR CODE STARTS HERE ##### # noqa: E266
        # TODO Fill in transform from odom_frame to base_frame
        odom_tf.header.stamp = self.odom_msg.header.stamp
        odom_tf.header.frame_id = self.odom_frame
        odom_tf.child_frame_id = self.base_frame
        odom_tf.transform.translation.x = self.pose[0]
        odom_tf.transform.translation.y = self.pose[1]
        odom_tf.transform.translation.z = 0.0
        odom_tf.transform.rotation.x = 0.0
        odom_tf.transform.rotation.y = 0.0
        odom_tf.transform.rotation.z = np.sin(self.pose[2] / 2.0)
        odom_tf.transform.rotation.w = np.cos(self.pose[2] / 2.0)
        ##### YOUR CODE ENDS HERE   ##### # noqa: E266

        # Broadcast transform
        self.tf_broadcaster.sendTransform(odom_tf)


def main(args=None):
    # Start up ROS2
    rclpy.init(args=args)

    # Create the node
    wo_node = WheelOdometryNode()

    # Let the node run until it is killed
    rclpy.spin(wo_node)

    # Clean up the node and stop ROS2
    wo_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
