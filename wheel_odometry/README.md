# Setup

This package implements wheel odometry for the TurtleBot3 models used in the course. There are no external Python dependencies beyond the usual ROS 2 Python packages. If you are missing ROS 2 or TurtleBot3 system packages, install them with your system package manager (for ROS Humble on Ubuntu 22.04 as an example):

```bash
sudo apt install ros-humble-turtlebot3
```

# Pre-Requisites
None

# File Layout

This directory contains the following files and folders:

- [`package.xml`](package.xml): ROS 2 package manifest with dependencies and metadata.
- [`pytest.ini`](pytest.ini): Pytest configuration file.
- [`setup.cfg`](setup.cfg): Configuration for Python packaging and linting.
- [`setup.py`](setup.py): Python setup script for installing the package.
- [`README.md`](README.md): This file.
- [`resource/wheel_odometry`](resource/wheel_odometry): Resource file for ROS 2 package index.

## Python Package

This package provides a small Python package that implements the TB3 wheel kinematics and a node that publishes odometry using those kinematics.

- [`wheel_odometry/__init__.py`](wheel_odometry/__init__.py): Initializes the `wheel_odometry` Python package.
- [`wheel_odometry/tb3_kinematics.py`](wheel_odometry/tb3_kinematics.py): Kinematics utilities for TurtleBot3 wheel odometry (forward/inverse kinematics and helper functions).
- [`wheel_odometry/wheel_odometry_node.py`](wheel_odometry/wheel_odometry_node.py): ROS node that computes and/or publishes wheel odometry (example usage in simulation or on real robot setups).

## Resources

- [`resource/wheel_odometry`](resource/wheel_odometry): Resource entry used by the ROS 2 package index.

## Tests

Unit tests and style checks are included to validate the implementation:

- [`test/test_kinematics.py`](test/test_kinematics.py): Unit tests for the kinematics functions.
- [`test/test_flake8.py`](test/test_flake8.py): Flake8 style check.
- [`test/test_pep257.py`](test/test_pep257.py): Docstring style check.
- [`test/test_copyright.py`](test/test_copyright.py): Copyright checks.

You can run the tests for this package with:
```bash
colcon test --packages-select wheel_odometry
```
If you do not want to see the output of the stylistic tests, you can just run the kinematics tests with:
```bash
colcon test --packages-select wheel_odometry --pytest-args test/test_kinematics.py
```
Either way, you can view the results with:

```bash
colcon test-result --verbose
```

## Usage

This package is intended to be imported by other nodes or run as a ROS 2 node via a launch file or direct invocation. The `wheel_odometry_node.py` file contains an example node implementation; adapt it to your launch setup or call the functions in `tb3_kinematics.py` from your own code.

## Notes

- The implementation assumes standard TB3 parameter conventions (wheelbase, wheel radius) — check the parameter values if you use a custom robot configuration.

# Instructions

Your task is to implement functions to describe the kinematics of the turtlebot3 (i.e., for a differential drive robot) and then to use those kinematics, along with sensor data, to perform wheel odometry. The code you need to implement is inside the following files:

## [`tb3_kinematics.py`](wheel_odometry/tb3_kinematics.py)

In this file, you will step through the process of calculating the displacement of a differential drive robot based on measurements of wheel rotations. This is done inside of a class called `TB3Kinematics`, which inherits from the [`TB3Params`](../mee4411_simulation/tb3_utils/tb3_params.py) class. This means a `TB3Kinematics` object will have all the parameters that a `TB3Params` object has and more.

### Steps

1. `calculate_wheel_change`: The first step is to take in the current and previous `sensor_msgs/msg/JointState` messages and use this to calculate the change in wheel angles and time. You need to look at these messages to understand how to "unpack" them, i.e., how to access the information you need. I recommend doing this while the simulation is running by using
```bash
ros2 topic echo /joint_states --once
```
which will publish one message to the command line. If you remove the `--once` flag then you can continue to see messages as you drive the robot around.

2. `calculate_displacement`: The next step is to take the output of the `calculate_wheel_change` function and use that to calculate the displacement of the robot (in the body frame). This will be two values, the distance traveled and the angle turned. Here you have to use the kinematics of the turtlebot3 robot and the parameters of the robot.

3. `calculate_pose`: Finally, you need to take in the current pose and the outputs of `calculate_displacement` and use that to calculate the next pose (formatted as a list of $x, y, \theta$ values).

### Testing 

Once you have finished this, you can test the code using
```bash
colcon test --packages-select wheel_odometry
```
I expect your code to pass all the tests in [`test_kinematics.py`](test/test_kinematics.py). The other tests are checks for style, so it is recommended, but not required, that you pass all of those.

## [`wheel_odometry_node.py`](wheel_odometry/wheel_odometry_node.py)

Once you have the kinematics implemented, you can use them to implement odometry in the `WheelOdometryNode` class. This is a ROS2 node -- it inherits from `Node` -- and it also has all the methods of the `TB3Kinematics` class. The input/output structure of the node is to subscribe to `joint_states` and publishes `odom` and TF transforms.

### Steps

1. [Initialization](wheel_odometry/wheel_odometry_node.py#L44-L47): One element of the node is an [`nav_msgs/msg/Odometry`](wheel_odometry/wheel_odometry_node.py#L43) object. The node will continually update and publish this. Here you need to initialize the coordinate frames within the message. You can use
```bash
ros2 interface show nav_msgs/msg/Odometry
```
to see the definition of the message structure.

2. [`JointStates` callback](wheel_odometry/wheel_odometry_node.py#L65-L90): This method runs every time you get a new `JointStates` message. It saves the previous message in the `self.prev_joint_states` field and updates/sends the odometry and TF. You do not need to do anything here.

3. [`JointStates` callback](wheel_odometry/wheel_odometry_node.py#L92-L126): This function should use your kinematics functions to update the pose of the robot and then use this to update the `Odometry` message before publishing it.

4. [`JointStates` callback](wheel_odometry/wheel_odometry_node.py#L128-L152): This uses the same information as before to update and send the transformation to the `tf` topic. Remember to use the frame conventions from [REP 105](https://www.ros.org/reps/rep-0105.html).

### Testing

To test how your approach works you need to use the simulation. Before you test your code, make sure the simulation itself is fully operational.

Bring this up using 
```bash
ros2 launch mee4411_simulation simulation.xml
```
In a separate terminal, run
```bash
ros2 run turtlebot3_teleop teleop_keyboard
```
Use the second terminal to drive the robot around in the simulation environment.

Now, go into the launch file for the simulation, specifically looking at the section for [odometry](../mee4411_simulation/launch/simulation.xml#L55-L61). The out-of-the-box solution for odometry comes from the `turtlebot3_fake_node`. You need to take this node out (I recommend commenting it out to begin with) and replace it with your node from the `wheel_odometry` package. This will be your first experience modifying a launch file for the project. When adding your node, make sure you set all the necessary parameters for the node (including those needed by the classes that `WheelOdometryNode` inherits from). If everything is set up correctly, when you relaunch the simulation then it should work exactly as it did before.

The next test to run is to add random noise to the `JointState` messages. To do this, you can call the simulation using
```bash
ros2 launch mee4411_simulation simulation.xml joint_noise_std:=10.0
```
This will add roughly $10^\circ$ of error per second to the joint angles. Once you change this, when you drive the robot around you should see the simulated robot (solid) and the "real" robot (transparent) diverge from one another. Adding more noise will speed up the divergence. (Note, `turtlebot3_fake_node` is not set to work with this, so changing that parameter will have no effect if you are using it.)
