from sensor_msgs.msg import JointState

import numpy as np
from typing import List

from tb3_utils import TB3Params


class TB3Kinematics(TB3Params):

    def __init__(self, robot_model) -> None:
        super().__init__(robot_model)

    def calculate_wheel_change(
            self,
            new_joint_states: JointState,
            prev_joint_states: JointState) -> List[float]:
        """
        Calculate the change in wheel angles and time between two joint state messages.

        Inputs:
            new_joint_states    new joint states (sensor_msgs/msg/JointState)
            prev_joint_states   previous joint states (sensor_msgs/msg/JointState)
        Outputs:
            delta_wheel_l       change in left wheel angle [rad]
            delta_wheel_r       change in right wheel angle [rad]
            delta_time          change in time [s]
        """
        ##### YOUR CODE STARTS HERE ##### # noqa: E266
        # TODO Calculate the output values
        t_new = new_joint_states.header.stamp.sec + new_joint_states.header.stamp.nanosec * 1e-9
        t_prev = prev_joint_states.header.stamp.sec + prev_joint_states.header.stamp.nanosec * 1e-9
        delta_time = t_new - t_prev
        delta_wheel_l = new_joint_states.position[0] - prev_joint_states.position[0]
        delta_wheel_r = new_joint_states.position[1] - prev_joint_states.position[1]
        ##### YOUR CODE ENDS HERE   ##### # noqa: E266

        # Data validation
        if np.isnan(delta_wheel_l):
            delta_wheel_l = 0.0
        if np.isnan(delta_wheel_r):
            delta_wheel_r = 0.0

        return (delta_wheel_l, delta_wheel_r, delta_time)

    def calculate_displacement(
            self,
            delta_wheel_l: float,
            delta_wheel_r: float) -> List[float]:
        """
        Calculate the displacement of the robot based on the change in wheel angles.

        Inputs:
            delta_wheel_l       change in left wheel angle [rad]
            delta_wheel_r       change in right wheel angle [rad]
        Outputs:
            delta_s         linear displacement [m]
            delta_theta     angular displacement [rad]
        """
        ##### YOUR CODE STARTS HERE ##### # noqa: E266
        # TODO Calculate the output values
        delta_s_l = delta_wheel_l * self.wheel_radius
        delta_s_r = delta_wheel_r * self.wheel_radius
        delta_s = (delta_s_r + delta_s_l) / 2.0
        delta_theta = (delta_s_r - delta_s_l) / self.wheel_separation
        ##### YOUR CODE ENDS HERE   ##### # noqa: E266

        return (delta_s, delta_theta)

    def calculate_pose(
            self,
            prev_pose: List[float],
            delta_s: float,
            delta_theta: float) -> List[float]:
        """
        Calculate the new pose of the robot based on the previous pose and the displacement.

        Inputs:
            prev_pose       input pose in format (x, y, theta) [m, m, rad]
            delta_s         linear displacement [m]
            delta_theta     angular displacement [rad]
        Outputs:
            pose            output pose in format (x, y, theta) [m, m, rad]
        """
        ##### YOUR CODE STARTS HERE ##### # noqa: E266
        # TODO Calculate the output values
        x_prev, y_prev, theta_prev = prev_pose
        x_new = x_prev + delta_s * np.cos(theta_prev + delta_theta / 2.0)
        y_new = y_prev + delta_s * np.sin(theta_prev + delta_theta / 2.0)
        theta_new = theta_prev + delta_theta
        pose = [x_new, y_new, theta_new]
        ##### YOUR CODE ENDS HERE   ##### # noqa: E266

        return pose
