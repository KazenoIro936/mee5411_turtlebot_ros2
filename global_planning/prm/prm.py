from rclpy.clock import Clock
from rclpy.publisher import Publisher


from geometry_msgs.msg import Point, PoseStamped
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
            logger=None,
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
        data = np.array(costmap.data, dtype=np.int8).reshape(
            costmap.metadata.size_y,
            costmap.metadata.size_x
        )
        data[data == 127] = -1
        og.data = data.flatten().tolist()

        # og.data = np.array(costmap.data).astype(np.int8).flatten().tolist()
        # og.data[og.data == 127] = -1  # unknown
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
            pt = self.sample_free_point()
            node_id = self.graph.number_of_nodes()
            # TODO Add the point to the graph node list
            #   Include an attribute called 'location' holding the 2D position
            #   'location' can be formatted as a list or as a numpy array
            #   see documentation here:
            #   https://networkx.org/documentation/stable/tutorial.html#adding-attributes-to-graphs-nodes-and-edges
            self.graph.add_node(node_id, location=pt)
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
        num_nodes = len(self.graph.nodes)
        if num_nodes == 0:
            return
        for i in range(num_nodes):
            p_i = pts[i]
            neighbor_indices = self.kdtree.query_ball_point(p_i, self.connection_radius)
            for j in neighbor_indices:
                if j == i or self.graph.has_edge(i, j):
                    continue
                p_j = pts[j]
                if self.valid_edge(p_i, p_j):
                    self.graph.add_edge(i, j)
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
        xmin, ymin, xmax, ymax = self.ogm.boundary
        while True:
            x = np.random.uniform(xmin, xmax)
            y = np.random.uniform(ymin, ymax)
            occ = self.ogm.is_occupied(np.array([x]), np.array([y]))[0]
            if not occ:
                return np.array([x, y])

        # TODO Check if point is valid (i.e., not in collision based on the map)
        #      If it is not then try again, if it is valid then return the point
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
        diff = p1 - p0
        dist = np.linalg.norm(diff)
        if dist == 0.0:
            occ = self.ogm.is_occupied(
                np.array([p0[0]]),
                np.array([p0[1]])
            )[0]
            return not occ
        # TODO Check to make sure none of the points collide with the map
        num_steps = max(1, int(dist / self.step_size))
        for k in range(1, num_steps + 1):
            alpha = k / num_steps
            p = p0 + alpha * diff  # interpolated point
            occ = self.ogm.is_occupied(np.array([p[0]]), np.array([p[1]]))[0]
            if occ:
                return False
        return True
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
        if self.ogm.is_occupied(np.array([start[0]]), np.array([start[1]]))[0]:
            raise Exception('Start point is in an occupied cell')
        if self.ogm.is_occupied(np.array([goal[0]]), np.array([goal[1]]))[0]:
            raise Exception('Goal point is in an occupied cell')

        start_id = self.graph.number_of_nodes()
        self.graph.add_node(start_id, location=start)
        goal_id = self.graph.number_of_nodes()
        self.graph.add_node(goal_id, location=goal)
        # Connect start node to nearby nodes
        if self.kdtree is None:
            raise Exception('KDTree not initialized')
        neighbor_indices = self.kdtree.query_ball_point(start, self.connection_radius)
        for j in neighbor_indices:
            p_j = self.graph.nodes[j]['location']
            if self.valid_edge(start, p_j):
                self.graph.add_edge(start_id, j)
        # Connect goal node to nearby nodes
        neighbor_indices = self.kdtree.query_ball_point(goal, self.connection_radius)
        for j in neighbor_indices:
            p_j = self.graph.nodes[j]['location']
            if self.valid_edge(goal, p_j):
                self.graph.add_edge(goal_id, j)
        # TODO Plan path using A*
        # NOTE Use networkx library to call A* to find a path from the start to goal nodes.
        #   See documentation here:
        #   https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.shortest_paths.astar.astar_path.html
        try:
            path_node_ids = nx.astar_path(
                self.graph,
                start_id,
                goal_id,
                heuristic=lambda a, b: np.linalg.norm(
                    self.graph.nodes[a]['location'] - self.graph.nodes[b]['location']
                )
            )
        except nx.NetworkXNoPath:
            self.graph.remove_node(start_id)
            self.graph.remove_node(goal_id)
            raise Exception('No path found between start and goal')
        # Remove start and goal nodes from the graph to restore original PRM

        # TODO Convert the path returned by networkx to a nav_msgs/msg/Path message
        # NOTE Make sure to include the start and goal points
        path = Path()
        if self.clock is not None:
            path.header.stamp = self.clock.now().to_msg()
        path.header.frame_id = self.ogm.frame_id
        for node_id in path_node_ids:
            loc = self.graph.nodes[node_id]['location']
            pose = PoseStamped()
            pose.header = path.header
            pose.pose.position.x = loc[0]
            pose.pose.position.y = loc[1]
            pose.pose.position.z = Z
            pose.pose.orientation.w = 1.0  # neutral orientation
            path.poses.append(pose)

        self.graph.remove_node(start_id)
        self.graph.remove_node(goal_id)
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
