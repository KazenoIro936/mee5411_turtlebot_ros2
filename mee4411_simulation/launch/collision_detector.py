#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, RegisterEventHandler
from launch.events import Shutdown
from launch.event_handlers import OnProcessExit
from launch.substitutions import EnvironmentVariable
from launch_ros.actions import Node


def generate_launch_description():
    arg_map_conversions_implemented = DeclareLaunchArgument(
        'map_conversions_implemented',
        default_value='False',
        description='Bool: whether map conversions have been implemented'
    )
    arg_use_sim_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value='False',
        description='Bool: whether to use simulation time'
    )

    collision_detector = Node(
        package='mee4411_simulation',
        executable='collision_detector',
        output='screen',
        emulate_tty=True,  # for colored output
        parameters=[
            {'tb3_model': EnvironmentVariable('TURTLEBOT3_MODEL')},
            {'map_conversions_implemented': True}
        ]
    )

    return LaunchDescription([
        arg_map_conversions_implemented,
        arg_use_sim_time,
        collision_detector,
        RegisterEventHandler(
            OnProcessExit(
                target_action=collision_detector,
                on_exit=[
                    EmitEvent(event=Shutdown()),
                ]
            )
        ),
    ])
