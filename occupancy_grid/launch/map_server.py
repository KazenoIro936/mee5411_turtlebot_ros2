#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, RegisterEventHandler, TimerAction
from launch.event_handlers import OnProcessStart
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # Arguments
    map_filename = LaunchConfiguration('map_filename')
    map_filename_arg = DeclareLaunchArgument(
        'map_filename',
        default_value=PathJoinSubstitution([
            FindPackageShare('occupancy_grid'),
            'maps',
            'levine-4.yaml'
        ]),
        description='Default map to load.'
    )

    delay = LaunchConfiguration('delay')
    delay_arg = DeclareLaunchArgument(
        'delay',
        default_value='1.0',
        description='Default delay to start map_server.'
    )

    use_sim_time = LaunchConfiguration('use_sim_time')
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='False',
        description='Whether to use simulation time.'
    )

    # Parameters
    lifecycle_nodes = ['map_server']
    use_sim_time = True
    autostart = True  # Nodes launching commands

    start_map_server = Node(
        package='nav2_map_server',
        executable='map_server',
        output='screen',
        # emulate_tty=True,  # https://github.com/ros2/launch/issues/188
        parameters=[
            {'frame_id': "map"},
            {'topic_name': "map"},
            {'yaml_filename': map_filename},
        ],
    )

    start_lifecycle_manager_cmd = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager',
        output='screen',
        # emulate_tty=True,  # https://github.com/ros2/launch/issues/188
        parameters=[
            {'use_sim_time': use_sim_time},
            {'autostart': autostart},
            {'node_names': lifecycle_nodes},
        ],
    )

    return LaunchDescription([
        map_filename_arg,
        delay_arg,
        use_sim_time_arg,
        start_map_server,
        RegisterEventHandler(
            OnProcessStart(
                target_action=start_map_server,
                on_start=[
                    LogInfo(msg='Map server started, initializing'),
                    TimerAction(
                        period=delay,  # Delay in seconds
                        actions=[start_lifecycle_manager_cmd]
                    )
                ]
            )
        ),
    ])
