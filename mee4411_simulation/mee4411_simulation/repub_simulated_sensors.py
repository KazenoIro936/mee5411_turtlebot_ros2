#!/usr/bin/env python3
import rclpy
from rclpy.duration import Duration
from rclpy.time import Time
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
from sensor_msgs.msg import LaserScan, JointState

from tb3_utils import TB3Params

from copy import deepcopy
import numpy as np
import os


class SimRepub(Node, TB3Params):

    def __init__(self) -> None:
        Node.__init__(self, node_name='repub_simulated_sensors')
        # Declare paramters
        self.declare_parameter('namespace_remove',
                               value='',
                               descriptor=ParameterDescriptor(
                                   type=ParameterType.PARAMETER_STRING))
        self.declare_parameter('noise_std',
                               value=0.0,
                               descriptor=ParameterDescriptor(
                                   type=ParameterType.PARAMETER_DOUBLE))
        self.declare_parameter('alpha',
                               value=0.9,
                               descriptor=ParameterDescriptor(
                                   type=ParameterType.PARAMETER_DOUBLE))
        self.declare_parameter('tb3_model',
                               value=os.getenv('TURTLEBOT3_MODEL', 'burger'),
                               descriptor=ParameterDescriptor(
                                   type=ParameterType.PARAMETER_STRING))

        # Namespace to remove from frames
        self.namespace_remove_ = \
            self.get_parameter('namespace_remove').get_parameter_value().string_value
        # Strip any leading or trailing slashes
        self.namespace_remove_ = self.namespace_remove_.strip('/')

        # Other Parameters
        self.noise_std_ = self.get_parameter('noise_std').get_parameter_value().double_value
        self.alpha_ = \
            self.get_parameter('alpha').get_parameter_value().double_value  # running average param
        self.tb3_model_ = \
            self.get_parameter('tb3_model').get_parameter_value().string_value

        # Get TB3 parameters
        TB3Params.__init__(self, robot_model=self.tb3_model_)

        # ROS publishers and subscribers
        self.scan_pub_ = self.create_publisher(LaserScan, '~/scan_out', 10)
        self.joint_states_pub_ = self.create_publisher(JointState, '~/joint_states_out', 10)
        self.prev_joint_state_ = None  # previous joint states
        self.joint_state_rate_ = None  # rate at which messages are being published
        self.position_ = [0.0, 0.0]
        self.velocity_ = [0.0, 0.0]
        self.get_clock().sleep_for(Duration(seconds=0.01))
        self.scan_sub_ = self.create_subscription(LaserScan, 'scan', self.scanCallback, 1)
        self.joint_states_sub_ = self.create_subscription(
            JointState,
            'joint_states',
            self.jointStateCallback,
            1)

    def stripName(self, name: str) -> str:
        # Strip "self.namespace_remove_" from "name"
        parts = name.split('/')
        parts = [p for p in parts if p not in ['', self.namespace_remove_]]
        return "/".join(parts)

    def scanCallback(self, msg: LaserScan) -> None:
        msg.header.frame_id = self.stripName(msg.header.frame_id)
        self.scan_pub_.publish(msg)

    def jointStateCallback(self, msg: JointState) -> None:
        msg_out = deepcopy(msg)

        # Change names
        msg_out.header.frame_id = self.stripName(msg.header.frame_id)
        msg_out.name = [self.stripName(n) for n in msg.name]

        # Add noise (only if moving)
        if self.noise_std_ > 0.0:
            if self.prev_joint_state_ is not None:
                # Get timing information
                dt = (Time.from_msg(msg.header.stamp).nanoseconds -
                      Time.from_msg(self.prev_joint_state_.header.stamp).nanoseconds) / 1e9
                if self.joint_state_rate_ is None:
                    self.joint_state_rate_ = dt
                else:
                    self.joint_state_rate_ = self.alpha_ * self.joint_state_rate_ \
                        + (1-self.alpha_) * dt

                # Get true information displacement
                self.position_ = [pos + (j1 - j0) for pos, j1, j0 in
                                  zip(self.position_,
                                      msg.position,
                                      self.prev_joint_state_.position)]
                self.velocity_ = [v for v in msg.velocity]

                # Add noise proportional to wheel velocity
                w_max = self.v_max / self.wheel_radius  # max wheel angular velocity
                for i, (w) in enumerate(self.velocity_):
                    std_dev = np.sqrt(dt) * self.noise_std_ * np.abs(w) / w_max
                    self.position_[i] += std_dev * np.random.randn()
                    self.velocity_[i] += std_dev * np.random.randn()

                # Update position and velocity
                msg_out.position = deepcopy(self.position_)
                msg_out.velocity = deepcopy(self.velocity_)
            self.prev_joint_state_ = msg

        # Publish message
        self.joint_states_pub_.publish(msg_out)


def main(args=None):
    # Start up ROS2
    rclpy.init(args=args)

    # Create the node
    srnode = SimRepub()

    # Let the node run until it is killed
    rclpy.spin(srnode)

    # Clean up the node and stop ROS2
    srnode.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
