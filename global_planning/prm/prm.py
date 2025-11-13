from rclpy.clock import Clock
from rclpy.publisher import Publisher

from geometry_msgs.msg import Point
from nav2_msgs.msg import Costmap
from nav_msgs.msg import OccupancyGrid, Path
from visualization_msgs.msg import Marker, MarkerArray

import networkx as nx
import numpy as np
from scipy.spatial import KDTree
from tqdm import tqdm

from occupancy_grid import OccupancyGridMap


Z = 0.0  # z coordinate for the graph display
ALPHA = 0.25  # alpha value for graph transparency


class PRM:
    def __init__(
            self,
            costmap: Costmap,
            num_points: int,
            connection_radius: float,
            step_size: float,
            *,
            logger = None,
            publisher: Publisher = None,
            publish_every_n: int = 20,
            clock: Clock = None) -> None:
        """
        Initialize the probabilistic roadmap in the given occupancy grid.

        Inputs:
            map: An Costmap object representing the environment
            num_points: Number of points to sample in the PRM
            connection_radius: Radius in which to check for connections between nodes
            step_size: Step size to use when checking for collisions between nodes
            publisher: ROS publisher for visualization
            publish_every_n: Publish the graph every n added points
            clock: ROS clock for timestamps
        Outputs:
            None
        """
        # Convert the costmap to an occupancy grid map
        og = OccupancyGrid()
        og.header = costmap.header
        og.info.map_load_time = costmap.metadata.map_load_time
        og.info.resolution = costmap.metadata.resolution
        og.info.width = costmap.metadata.size_x
        og.info.height = costmap.metadata.size_y
        og.info.origin = costmap.metadata.origin
        og.data = np.array(costmap.data).astype(np.int8).flatten().tolist()
        og.data[og.data == 127] = -1  # unknown
        self.ogm = OccupancyGridMap.from_msg(og)

        # Check inputs
        self.logger = logger
        if publisher is not None and clock is None:
            raise Exception('Clock is required when publishing')
        self.publisher = publisher
        self.publish_every_n = publish_every_n
        self.clock = clock

        # Parameters
        self.connection_radius = connection_radius  # radius in which to check for connections
        self.step_size = step_size  # size of step to take for collision checking

        # Data structures
        self.graph = None  # networkx graph
        self.kdtree = None  # KDTree for quickly finding nearest node

        # Build the PRM
        self.build_prm(num_points)

    def build_prm(self, num_points: int) -> None:
        """
        Build a PRM graph consisting of num_points.

        Inputs:
            num_points: Number of points to sample in the PRM
        Outputs:
            None
        """
        # Initialize empty graph and add points
        self.graph = nx.Graph()  # intialize empty graph
        self.kdtree = None  # initialize empty KD tree
        self.add_to_prm(num_points)

    def add_to_prm(self, num_points: int) -> None:
        """
        Add num_points to the PRM graph.

        Inputs:
            num_points: Number of points to sample in the PRM
        Outputs:
            None

        All points should be in the free space of the map.
        Points should be connected if they are within self.connection_radius and
            the edge between them is valid (i.e., not in collision).
        """
        # First, add points to the graph
        for i in tqdm(range(num_points)):  # Wrap in tqdm for progress bar
            ##### YOUR CODE STARTS HERE ##### # noqa: E266
            # TODO Generate valid point in free space
            pass

            # TODO Add the point to the graph node list
            #   Include an attribute called 'location' holding the 2D position
            #   'location' can be formatted as a list or as a numpy array
            #   see documentation here:
            #   https://networkx.org/documentation/stable/tutorial.html#adding-attributes-to-graphs-nodes-and-edges
            pass
            ##### YOUR CODE ENDS HERE   ##### # noqa: E266
            # Display graph as it is being built
            if self.publisher is not None and i % self.publish_every_n == self.publish_every_n - 1:
                self.publisher.publish(self.to_msg())

        # Show final set of nodes
        if (self.publisher is not None) and (i % self.publish_every_n != self.publish_every_n - 1):
            self.publisher.publish(self.to_msg())

        # Initialize KD tree for quickly finding nearest node
        pts = np.array([p for _, p in self.graph.nodes.data('location')])
        assert pts.shape[1] == 2
        self.kdtree = KDTree(pts)

        # Next, add edges between the points within the connection radius
        ##### YOUR CODE STARTS HERE ##### # noqa: E266
        # TODO For each point in the graph, do the following:
        #   Find other points within the connection radius
        #   Check to see if the path from the new point to a previous point is obstacle free
        #     If it is, add an edge between the two points in the graph
        # NOTE Read the documentation for the KDTree class to find points within a certain radius
        #    https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.KDTree.html
        pass
        ##### YOUR CODE ENDS HERE   ##### # noqa: E266
        # Show final graph with edges
        if self.publisher is not None:
            self.publisher.publish(self.to_msg())

    def sample_free_point(self) -> np.array:
        """
        Draw a random points from within the free space of the map.

        Inputs:
            None
        Outputs:
            2D point within the map as a numpy.array
        """
        ##### YOUR CODE STARTS HERE ##### # noqa: E266
        # TODO Draw a random point within the boundary
        # NOTE You can use np.random.rand to draw random numbers between 0 and 1
        pass

        # TODO Check if point is valid (i.e., not in collision based on the map)
        #      If it is not then try again, if it is valid then return the point
        pass

        return np.array([0.0, 0.0])  # placeholder return
        ##### YOUR CODE ENDS HERE   ##### # noqa: E266

    def valid_edge(self, p0: np.array, p1: np.array) -> bool:
        """
        Check to see if an edge connecting p0 to p1 is in collision with the map.

        Inputs:
            p0: 2D point as a numpy.array
            p1: 2D point as a numpy.array
        Outputs:
            True if the edge is valid (i.e., not in collision), False otherwise
        """
        ##### YOUR CODE STARTS HERE ##### # noqa: E266
        # TODO Create a series of points starting at p0 and ending at p1 in steps of self.step_size
        pass

        # TODO Check to make sure none of the points collide with the map
        pass

        return False
        ##### YOUR CODE ENDS HERE   ##### # noqa: E266

    def query(self, start: np.array, goal: np.array) -> Path:
        """
        Query the PRM to get a path from a start point to a goal point.

        Inputs:
            start: 2D point as a numpy.array
            goal: 2D point as a numpy.array
        Outputs:
            Return a nav_msgs/msg/Path object from start to goal
        """
        # Make sure the PRM is initialized
        if self.graph is None:
            raise Exception('PRM not initialized')
        ##### YOUR CODE STARTS HERE ##### # noqa: E266
        # TODO Add the start and goal points to the PRM graph
        # NOTE You need to connect the start and goal points to existing nodes in the PRM
        pass

        # TODO Plan path using A*
        # NOTE Use networkx library to call A* to find a path from the start to goal nodes.
        #   See documentation here:
        #   https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.shortest_paths.astar.astar_path.html
        pass

        # TODO Convert the path returned by networkx to a nav_msgs/msg/Path message
        # NOTE Make sure to include the start and goal points
        path = Path()
        pass

        return path
        ##### YOUR CODE ENDS HERE   ##### # noqa: E266

    def to_msg(self) -> None:
        """Convert the PRM graph to a visualization_msgs/msg/MarkerArray."""
        if self.clock is None:
            raise Exception('Clock is required for to_msg')

        # Create marker array
        ma = MarkerArray()

        # Create marker to show the nodes in the graph
        points_marker = Marker()
        points_marker.header.frame_id = self.ogm.frame_id
        points_marker.header.stamp = self.clock.now().to_msg()
        points_marker.ns = 'points'
        points_marker.type = Marker.SPHERE_LIST
        points_marker.pose.orientation.w = 1.0
        points_marker.scale.x = 0.1
        points_marker.scale.y = 0.1
        points_marker.scale.z = 0.1
        points_marker.color.r = 1.0
        points_marker.color.a = ALPHA

        for _, loc in self.graph.nodes.data('location'):
            points_marker.points.append(Point(x=loc[0], y=loc[1], z=Z))

        ma.markers.append(points_marker)

        # Create marker to show the edges in the graph
        edges_marker = Marker()
        edges_marker.header.frame_id = self.ogm.frame_id
        edges_marker.header.stamp = points_marker.header.stamp
        edges_marker.ns = 'edges'
        edges_marker.type = Marker.LINE_LIST
        edges_marker.pose.orientation.w = 1.0
        edges_marker.scale.x = 0.05
        edges_marker.color.b = 1.0
        edges_marker.color.a = ALPHA

        for e in self.graph.edges:
            p0 = self.graph.nodes[e[0]]['location']
            p1 = self.graph.nodes[e[1]]['location']
            edges_marker.points.append(Point(x=p0[0], y=p0[1], z=Z))
            edges_marker.points.append(Point(x=p1[0], y=p1[1], z=Z))

        ma.markers.append(edges_marker)

        return ma
