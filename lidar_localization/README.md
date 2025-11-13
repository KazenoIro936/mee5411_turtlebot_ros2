# Overview

This package contains a simple 2D iterative closest point (ICP) based localization node that uses a map (an occupancy grid) and laser scans to estimate the robot pose in the map frame. The node converts incoming laser scans to point clouds, aligns them with the map using ICP, and publishes the resulting transform between the map and odom frames.

# Learning objectives

1. Implement and test a 2D ICP algorithm (least-squares best-fit + iterative refinement).
2. Convert between different 2D transform representations (Transform messages, homogeneous matrices, $x,y,\theta$).
3. Integrate an algorithm with ROS2: subscribe to sensors, use transforms, and publish TFs.

# Pre-Requisites
You need to have completed the `MapConversions` and `OccupancyGridMap` classes from the [occupancy grid map](../occupancy_grid/README.md) module.

# Dependencies

This package requires an additional Python module, [`scikit-learn`](https://scikit-learn.org/stable/).

```bash
pip3 install scikit-learn
pip3 install scipy -U
```

The second line makes sure that `scipy` is up to date (I ran into an error before I did this). Additionally, standard ROS2 packages are required at runtime (rclpy, tf2_ros, nav_msgs, sensor_msgs, geometry_msgs).

# File Layout

This directory contains the following files and folders:

- [`package.xml`](package.xml): ROS 2 package manifest with dependencies and metadata.
- [`pytest.ini`](pytest.ini): Pytest configuration file.
- [`setup.cfg`](setup.cfg): Configuration for Python packaging and linting.
- [`setup.py`](setup.py): Python setup script for installing the package.
- [`README.md`](README.md): This file.
- [`resource/lidar_localization`](resource/lidar_localization): Resource file for ROS 2 package index.
- [`test/`](test) - basic package tests (flake8, pep257, copyright)

## [`lidar_localization`](lidar_localization) Python Package

This package will provide a solution for 2D Iterative Closest Point (ICP). This algorithm will then be used in a ROS node to perform localization to correct for drift in the robot's odometry.

- [`lidar_localization/__init__.py`](lidar_localization/__init__.py): Initializes the `wheel_odometry` Python package.
- [`lidar_localization/icp2d.py`](lidar_localization/icp2d.py): 2D ICP algorithm implementation (class `ICP2D`). Contains `set_map_points`, `best_fit_transform`, and `icp` methods.
- [`lidar_localization/icp_node.py`](lidar_localization/icp_node.py): ROS2 node that subscribes to `scan` and (`map` or map server), runs ICP, and publishes TFs.

# Instructions

Your task is to implement functions to perform ICP and then to use that algorithm to perform localization.

## [`icp2d.py`](lidar_localization/icp2d.py)

This file implements the `ICP2D` class. There are three methods to this.

### [`set_map_points`](lidar_localization/icp2d.py#L12-L32)
This method stores the map points and builds a NearestNeighbors object for fast matching. There is nothing that you have to do here.

### [`best_fit_transform`](lidar_localization/icp2d.py#L34-L80)
This method computes the least-squares best-fit 2D transform between two point sets. You are expected to implement the ideas in [slides 5-14](https://www.ipb.uni-bonn.de/html/teaching/msr2-2020/sse2-03-icp.pdf). The function should output the 3x3 homogeneous transformation matrix and mean matching error.

### [`icp`](lidar_localization/icp2d.py#L82-L138)
This method runs the full ICP loop, described in [slides 15-19](https://www.ipb.uni-bonn.de/html/teaching/msr2-2020/sse2-03-icp.pdf). In this, you should find the nearest neighbor for each point to determine correspondences, use the `best_fit_transform` to calculate the transformation, and update the points for the next iteration.

Notes on expected behavior and common pitfalls:
- Inputs are expected in 2xN or Nx2 layouts as documented in the code—follow the function docstrings.
- Be careful with numpy shapes and right indexing when building homogeneous transforms.

## [`icp_node.py`](lidar_localization/icp_node.py)

This file implements the `ICPLocalizationNode` class, which will use ICP to perform localization and perform all the ROS input/output. The ROS components of this node are:

- Parameters
    - `initial_pose_x`, `initial_pose_y`, `initial_pose_a`: initial pose estimate of odom relative to map
    - `odom_frame`: child frame id for published TF (default `odom`)
    - `time_travel`: optional time offset to use when querying TFs
    - `use_map_topic`: whether to subscribe to a latched `map` topic or call the map server
- Subscriptions:
    - `scan` (sensor_msgs/msg/LaserScan): Input lidar scans.
    - `initialpose` (geometry_msgs/msg/PoseWithCovarianceStamped): User-provided initial pose estimate (e.g., from `rviz`).
	- `map` (nav_msgs/msg/OccupancyGrid): The node gets the map to localize the robot in from the `map` topic or the `map_server/map` service.
    - `tf`: Listens for transform from odom -> sensor frame, broadcasts map -> odom transform when pose is updated (per [REP 105](https://ros.org/reps/rep-0105.html#frame-authorities)). 

Note that the node requires a map and the odom -> sensor frame transformations before it can perform ICP.

### [`initialize_icp`](lidar_localization/icp_node.py#L148-L175)

The purpose of this method is to take in an `OccupancyGrid` map and use that to initialize the point cloud for the map. *You must implement this.*

The idea here is to use your `OccupancyGridMap` code from the [`occupancy_grid` package](../occupancy_grid/occupancy_grid/occupancy_grid_map.py) and the [ICP class](icp2d.py) to do this.

### [`map_callback`](lidar_localization/icp_node.py#L177-L189)

This method is the callback function used to process `OccupancyGrid` messages. This initializes an `OccupancyGridMap` objects then calls the `initialize_icp` method.

### [`publish_map_odom_tf`](lidar_localization/icp_node.py#L191-L215)

The purpose of this method is to publish the transformation from the `map` to `odom` coordinate frames to the `tf` topic. *You must implement this.*

The class has the message already saved within the class in the `self.tf_map_odom` field. It is [initialized here](lidar_localization/icp_node.py#L80), the header frame is [set here](lidar_localization/icp_node.py#L167), and the child frame is [set here](lidar_localization/icp_node.py#L81). Look at the message definition using `ros2 interface show geometry_msgs/msg/TransformStamped` to see what is left for you to fill in here.

### [`initialpose_callback`](lidar_localization/icp_node.py#L217-L225)

This method is the callback function for the `initialpose` topic. This allows you to update the estimate of the robot's initial pose using a topic, which is typically done through [rviz](https://docs.ros.org/en/humble/Tutorials/Intermediate/RViz/RViz-User-Guide/RViz-User-Guide.html#d-pose-estimate).

### [`scan_callback`](lidar_localization/icp_node.py#L227-L275)

The purpose of this method is to use the incoming lidar points and the pre-existing map points to perform ICP to localize the robot. *You must implement this.* 

Based on [REP 105](https://ros.org/reps/rep-0105.html#frame-authorities), this node should publish a transformation from the `map` frame to the `odom` frame. This means we need the data from the lidar to be put into the `odom` frame (and you need to convert the `LaserScan` message into a point cloud). From there you can pass that point cloud into ICP to find the pose of the `odom` frame in the `map` frame before publishing it.

## Launch File

You also need to update the [`simulation.xml` launch file](../mee4411_simulation/launch/simulation.xml) by removing the `static_transform_publisher` (which publishes a constant transform from the `map` to `odom` frame) and replacing it with your new node.

# Testing

This package includes only basic automated style tests. Run them from the workspace root after building with colcon:

```bash
# from workspace root
colcon build --packages-select lidar_localization
source install/setup.bash
colcon test --packages-select lidar_localization
colcon test-result --verbose
```

Unit tests provided by ROS in `test/` are primarily style and metadata checks (flake8, pep257).

# How to run the node (quick guide)

1. Build and source your workspace:

```bash
colcon build
source install/setup.bash
```

2. Run the node (after the package is installed into the workspace):

```bash
ros2 launch mee4411_simulation simulation.xml
```

3. Once you have it running, try adding noise to the odometry (assuming you have `wheel_odometry` completed). You can do this with:

```bash
ros2 launch mee4411_simulation simulation.xml joint_noise_std:=30.0
```

The expected behavior is to see the two robots stay aligned, but the `odom` frame will move away from the origin. This indicates that your localization node is publishing corrections to this transformation.

# Suggested development & debugging order

1. Implement and unit-test `ICP2D.best_fit_transform` separately with small synthetic point sets (translate/rotate a set and ensure transform recovers it).
2. Implement `set_map_points` and test nearest neighbors behavior.
3. Implement the `icp` loop and test convergence on noisy data.
4. Implement `ICPLocalizationNode.initialize_icp` to extract points from an `OccupancyGrid` and call `set_map_points`.
5. Implement `scan_callback`: convert ranges to (x,y) in the lidar frame, transform to odom frame, run ICP, and update `self.pose`.
6. Implement `publish_map_odom_tf` to broadcast the computed transform.

# Edge cases and tests to run manually

- Verify the node handles empty scans and ranges with NaN/Inf.
- Test with different map resolutions (map.info.resolution) - the node uses the resolution as ICP tolerance.
- Test with errors in the initial pose to ensure ICP either converges or fails cleanly.
- Large rotations (near $pi$) and reversed correspondences - ensure SVD-based best-fit handles reflections correctly (determinant check).

# Contributing

If you add tests, please place them under the `test/` directory and update `pytest.ini` if needed. Keep style consistent with the rest of the repository (PEP8 and PEP257).
