# Summary

You will create a node that can generate occupancy grid maps from text file descriptions.

# Learning Objectives

1. Get experience programming in Python
2. Practice reading files in code
3. Create and publish a ROS message

# Pre-Requisites
None

# Starter Code Structure

The `occupancy_grid` package contains the following folders and files:

- [`package.xml`](package.xml) A file holding information about what this package does and what other code is needed to run it
- [`setup.py`](setup.py) A file holding information about the files in this package and what will get installed when you run `colcon build`
- [`setup.cfg`](setup.cfg) A file holding more information used by `colcon` to install the code
- [`pytest.ini`](pytest.ini) A file that configures options used to test your code
- [`LICENSE`](LICENSE) A file holding the software license for this package
- [`launch`](launch) A directory holding the launch files for this package
    - [`occ_grid.xml`](launch/occ_grid.xml) This is the main launch file you will use in this part of the project. Change the argument `filename` to change which map you are using. Make sure to test your code against all the provided maps. Launching with this will bring up three nodes:
        - [`block_visualizer`](launch/occ_grid.xml#L22-L30) This is a node that I wrote to visualize the true map.
        - [`rviz`](launch/occ_grid.xml#L32-L35) This is ROS’s built-in visualization tool (rviz = ROS Visualization)
        - [`create_occupancy_grid`](launch/occ_grid.xml#L13-L20) This is the node you will write to create and publish an occupancy grid representation of the map.
        - The launch file also takes in a few arguments: [`frame_id`](launch/occ_grid.xml#L3-L5) to set the name of the map’s coordinate frame, [`resolution`](launch/occ_grid.xml#L6-L8) to set the size of each grid cell (m/cell), and [`filename`](launch/occ_grid.xml#L9-L11) to select the map description to load.
- [`maps`](maps) A directory holding the text files that describe the maps
    - [`map0.yaml`](maps.map0.yaml) , [`map1.yaml`](maps.map1.yaml) , [`map2.yaml`](maps.map2.yaml), [`map3.yaml`](maps.map3.yaml) These files hold the map descriptions. All objects, including the map itself, are assumed to be rectangles that are aligned with the xy axes. Each file has the following structure (`map0.yaml` is shown here):
        ```yaml
        /**:
          ros__parameters:
            boundary: [-5.0, -5.0, 5.0, 15.0]
            blocks: [-3.0, 1.0,   3.0, 1.25,
                     -3.0, 1.0, -2.75, 10.0,
                      2.75, 1.0,   3.0, 10.0]
        ```
        
        - The first line `/**` tells ROS to use this parameters in whatever node calls the file
        - The second line `ros__parameters` tells the system that everything after that are ROS parameters
        - The third line `boundary: <1x4 ARRAY>` holds information about the boundary of the environment in the format `xmin, ymin, xmax, ymax`. This specifies the lower left and upper right corner of the environment rectangle.
        - The fourth line `blocks: <1x4N ARRAY>` holds information about the `N`  rectangular obstacles (blocks) in the environment. The array is formatted so that each set of 4 numbers corresponds to a single block, in the order `xmin, ymin, xmax, ymax`. Note that the line breaks are just visual to make it easier to keep track of.
- [`occupancy_grid`](occupancy_grid) A directory holding the Python code for this package
    - [`__init__.py`](occupancy_grid/__init__.py) A file that is used to create a [Python package](https://www.geeksforgeeks.org/python/what-is-__init__-py-file-in-python/)
    - [`block_visualizer.py`](occupancy_grid/block_visualizer.py) A node that visualizes the actual blocks and boundary of the environment. This is a helper node that I wrote to help you visually debug your code.
    - [`map_conversions.py`](occupancy_grid/map_conversions.py) A set of helper functions to convert between different map coordinates for an occupancy grid map. You will need to fill this in (see details below).
    - [`occupancy_grid_map.py`](occupancy_grid/occupancy_grid_map.py) A class that builds on top of the map conversions code to actually create a `nav_msgs.msg.OccupancyGrid` message. You will need to fill this in (see details below).
    - [`occupancy_grid_node.py`](occupancy_grid/occupancy_grid_node.py) A node that reads in the map `yaml` file and publishes a `nav_msgs.msg.OccupancyGrid` message. You will need to fill this in (see details below).
- [`resource`](resource) A directory holding a file used by `colcon`
    - [`occupancy_grid`](resource/occupancy_grid) An empty file that serves as a marker that the package gets installed.
- [`rviz`](rviz) A directory holding the configuration file used for the visualization tool `rviz`
    - [`occupancy_grid.rviz`](rviz/occupancy_grid.rviz) A configuration file that sets up the visualizer `rviz` to subscribe to and show data from the desired topics. You do not need to change anything in this file. However, if you update anything in `rviz`, including the viewing angle, you can save the configuration over this original one to automatically load in your new changes. Two items that are particularly helpful to change are the `Plane Cell Count` and `Cell Size` parameters for the `Grid` object in `rviz`. The `Grid` object just shows a uniform grid in the visualization window (it is completely separate from the occupancy grid). This `Plane Cell Count`controls how many grid cells are displayed and the second controls the size of the grid. As you test different resolutions in your launch file, be sure to update the `Cell Size` parameter to match it to more easily see if things are working correctly.
- [`test`](test) A directory holding tests you can run against your code
    - [`test_copyright.py`](test/test_copyright.py) A test that ROS automatically adds to ensure that your code has a license set.
    - [`test_flake8.py`](test/test_flake8.py) A test that ROS automatically adds to enforce [PEP 8 style guidelines](https://peps.python.org/pep-0008/) for your code, detect programming errors, and identify overly complex code statements. See more in the [ROS documentation](https://docs.ros.org/en/humble/Tutorials/Advanced/Ament-Lint-For-Clean-Code.html#ament-flake8).
    - [`test_pep257.py`](test/test_pep257.py) A test that ROS automatically adds to check your code against the [PEP 257 style guidelines](https://peps.python.org/pep-0257/) for code documentation.
    - [`test_map_conversions.py`](test/test_map_conversions.py) A test suite that I wrote to check your [`map_conversions.py`](occupancy_grid/map_conversions.py) code against some test cases.

# Task

Your primary task in this phase of the project is to fill in the three files [`map_conversions.py`](occupancy_grid/map_conversions.py), [`occupancy_grid_map.py`](occupancy_grid/occupancy_grid_map.py), and [`occupancy_grid_node.py`](occupancy_grid/occupancy_grid_node.py) (all of which are in the [`occupancy_grid`](occupancy_grid) folder).

## [`map_conversions.py`](occupancy_grid/map_conversions.py)

This file contains a class called [`MapConversions`](occupancy_grid/map_conversions.py#L7) with a number of different methods (i.e., functions) that you need to fill in. There are functions to initialize the object, extract information from an occupancy grid message, and six functions to convert between the different representations: $(x,y)$ coordinates, (row, column) subscripts, and linear indices (position in the 1D array holding the map). Two of the functions are written for you already, [`xy2ind`](occupancy_grid/map_conversions.py#L112-L125) and [`ind2xy`](occupancy_grid/map_conversions.py#L127-L139). Your job is to fill in the remainder of these functions.

Remember, we are assuming the following conversions:

| $(r,c) \to (x,y)$ | $(x,y) \to (r,c)$ | $i \to (r,c)$ | $(r,c) \to i$ |
| --- | --- | --- | --- |
| $x=x_0+s(c+0.5)$ | $r = \lfloor \frac{y-y_0}{x} \rfloor$ | $r = \lfloor \frac{i}{N_c} \rfloor$ | $i = r N_c + c$ |
| $y=y_0+s(r+0.5)$ | $c = \lfloor \frac{x-x_0}{x} \rfloor$ | $c = \mod(i, N_c)$ |  |

where $s$ is the cell size, $N_c$ is the number of columns in the map, and $(x_0, y_0)$ are the coordinates of the lower left corner of the map. You also need to be careful about points that are exactly on the right and/or top boundaries of the environment as these are not, by default, included within the cells due to our choices in rounding. You must add in the necessary logic to achieve this behavior.

See the code for detailed specifications about inputs and outputs. This includes instructions what to do about points that are not valid (i.e. are outside of the map). Remember to test your code as you go by using the test script provided with the starter code.

## [`occupancy_grid_map.py`](occupancy_grid/occupancy_grid_map.py)

This file contains another class that builds on (i.e., inherits from) the `MapConversions` class from the previous file. You need to fill in the functions to create an object from an `OccupancyGrid` message, to add a block to the map, to convert the object into an `OccupancyGrid` message, and to check occupied locations in the map. You can use the conversion functions that you wrote in the last file to help you out here.

Make sure you understand the structure of the `OccupancyGrid` message, which you can show in the command line using `ros2 interface show nav_msgs/msg/OccupancyGrid` or read the [online documentation](https://docs.ros.org/en/humble/p/nav_msgs/msg/OccupancyGrid.html),

One comment about the implementation: Be careful about using equality tests between real numbers. Due to rounding during computations, you will find that statements like $\frac{2.0 - 1.0}{0.05}$ might not equal exactly 20.0, as one would expect.

Be sure to test your code with different values of the resolution to ensure that it is working properly. The `rviz` window will be very helpful (including changing the `Grid` display) when debugging this code.

## [`occupancy_grid_node.py`](occupancy_grid/occupancy_grid_node.py)

This is the ROS part of the project. This file contains two functions, the [`main`](occupancy_grid/occupancy_grid_node.py#L33-L45) function and the [`OccupancyGridNode`](occupancy_grid/occupancy_grid_node.py#L11-L30) class. You do not need to do anything to the `main` function. `OccupancyGridNode` should do all of the work here. You need to do the following:

1. Initialize the ROS node (the name will be `occupancy_grid`). This is done for you [here](occupancy_grid/occupancy_grid_node.py#L14).
2. Create a publisher to publish your map. You need to set the QoS (Quality of Service) for this publisher to use the [transient local option for durability](https://docs.ros.org/en/humble/Concepts/Intermediate/About-Quality-of-Service-Settings.html#qos-policies). This will allow you to just publish the message once, and any future nodes that come up will also still receive the message.
3. Read in the map parameters from the parameter server (see the launch file or use the command line tools to see the parameters that you need to load in).
4. Create an empty `nav_msgs/msg/OccupancyGrid` message
5. Fill in this message using the parameters and your `occupancy_grid_map.py` code from the previous file (this is already imported for you at the top).
6. Publish the message (your node should only publish a single message, but should *not* shut down after it sends the message).

Feel free to use the code from the tutorials as a starting point as you go about writing your ROS node.

# Testing

You should test your code as you work on it. Here is what I recommend

## `map_conversions.py`

Use the Python test file to make sure this is working properly. To run this, go to your ROS workspace and run the command `colcon test`. This will run all tests against the packages in your workspace and output the results to the command line. You want to make sure that your code passes all the tests, particularly in [`test_map_conversions.py`](test/test_map_conversions.py).

## `occupancy_grid_map.py`

There is no test set up to directly ensure that this is working properly.

## `occupancy_grid_node.py`

Use the launch file and `rviz` to test this. You want to make sure that your occupancy grid is an *over-approximation* of all the blocks in the map, that your map is the right size, and that your code works for a variety of different cell resolutions. This will implicitly test your `occupancy_grid_map.py` code, since you will use that here.

You can also use the ROS command line tools to check that the topic used to publish the map exists and is properly connected to `rviz`.

## Suggested Order

While working on this phase of the project, I have the following recommendations on the order of the tasks as well as suggestions for working on each task:

1. First write your ROS node in `occupancy_grid_node.py`
    1. Use `ros2 launch` to make sure your node successfully starts
    2. Add some logging information to easily print information to the screen during run time to help debug. You can read more about this in the [documentation](https://docs.ros.org/en/humble/Concepts/Intermediate/About-Logging.html) and see [examples](https://github.com/ros2/examples/blob/humble/rclpy/services/minimal_client/examples_rclpy_minimal_client/client.py#L29).
    3. Start by publishing an empty message, then fill in the `Header` and `MapMetaData` pieces. Leave the `data` field for later.
    4. Use `ros2 topic echo /map` to see the contents of your message and to make sure it successfully publishes
2. Next write your conversion functions in `map_conversions.py`
    1. Test your code early and often using `rostest`
    2. Test each function before moving on the next
3. Finally, write the code to fill in the map in `occupancy_grid_map.py` and use this function to fill in the `data` field in the map message
    1. Use the functions you wrote in `map_conversions.py` and tested earlier to help fill in the map
    2. Use this function to finish filling in the message in `occ_grid_node.py`
    3. Test with multiple maps and different resolution values by changing the `arg`s in your launch file
    4. Debug using the `rviz` visualization
    5. Make sure the Cell Size value for the Grid in `rviz` matches the resolution in your launch file
