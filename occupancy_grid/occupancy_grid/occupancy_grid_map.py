from rclpy.time import Time
from nav_msgs.msg import OccupancyGrid

import numpy as np

from .map_conversions import MapConversions


class OccupancyGridMap(MapConversions):
    def __init__(self, boundary, resolution, frame_id) -> None:
        super(OccupancyGridMap, self).__init__(boundary, resolution)
        # Set coordinate frame ID
        self.frame_id = frame_id
        # Initialize empty data array (2D array holding values)
        #   In the range [0, 100], representing the probability of occupancy
        #   If a cell is unknown, set to -1
        self.data = np.zeros(self.array_shape)

    @classmethod
    def from_msg(cls, msg: OccupancyGrid):
        """Create an object from an OccupancyGrid msg."""
        ##### YOUR CODE STARTS HERE ##### # noqa: E266
        # TODO Extract boundary, resolution, and frame_id from input message
        resolution = msg.info.resolution
        xmin = msg.info.origin.position.x
        ymin = msg.info.origin.position.y
        xmax = xmin + msg.info.width * resolution
        ymax = ymin + msg.info.height * resolution
        boundary = [xmin, ymin, xmax, ymax]
        frame_id = msg.header.frame_id
        ##### YOUR CODE ENDS HERE   ##### # noqa: E266

        # Initialize object
        ogm = cls(boundary, resolution, frame_id)

        ##### YOUR CODE STARTS HERE ##### # noqa: E266
        # TODO Update data array in ogm, based on conventions in the __init__ method
        ogm.data = np.array(msg.data).reshape((msg.info.height, msg.info.width))
        ##### YOUR CODE ENDS HERE   ##### # noqa: E266
        return ogm

    def add_block(self, block: np.array) -> None:
        """
        Add a block to the map stored in self.data.

        Inputs:
            block   np.array in the format (xmin, ymin, xmax, ymax)
        """
        ##### YOUR CODE STARTS HERE ##### # noqa: E266
        # TODO Fill in all the cells that overlap with the block
        # bx0, by0, bx1, by1 = map(float, block)
        # if np.isnan([bx0, by0, bx1, by1]).any():
        # return
        # if bx0 > bx1:
        # bx0, bx1 = bx1, bx0
        # if by0 > by1:
        # by0, by1 = by1, by0
        # xmin, ymin, xmax, ymax = self.boundary
        # res = self.resolution
        # nrows, ncols = self.array_shape
        # if (bx1 <= xmin) or (bx0 >= xmax) or (by1 <= ymin) or (by0 >= ymax):
        # return
        # bx0 = max(bx0, xmin)
        # by0 = max(by0, ymin)
        # bx1 = min(bx1, xmax)
        # by1 = min(by1, ymax)
        # j_min = int(np.floor((bx0 - xmin) / res))
        # j_max = int(np.ceil((bx1 - xmin) / res) - 1)
        # i_min = int(np.floor((by0 - ymin) / res))
        # i_max = int(np.ceil((by1 - ymin) / res) - 1)
        # j_min = max(0, min(j_min, ncols - 1))
        # j_max = max(0, min(j_max, ncols - 1))
        # i_min = max(0, min(i_min, nrows - 1))
        # i_max = max(0, min(i_max, nrows - 1))
        # if (j_min > j_max) or (i_min > i_max):
        # return
        bx0, by0, bx1, by1 = block
        X = np.array([bx0, bx1])
        Y = np.array([by0, by1])
        Rows, Cols = self.xy2sub(X, Y)
        self.data[Rows[0]:Rows[1] + 1, Cols[0]:Cols[1] + 1] = 100
        ##### YOUR CODE ENDS HERE   ##### # noqa: E266

    def to_msg(self, time: Time) -> OccupancyGrid:
        """
        Convert the OccupancyGridMap object into an OccupancyGrid ROS message.

        Inputs:
            time    current ROS time
        Outputs:
            msg     OccupancyGrid ROS message
        """
        msg = OccupancyGrid()
        ##### YOUR CODE STARTS HERE ##### # noqa: E266
        # TODO Fill in all the fields of the msg using the data from the class
        msg.header.stamp = time.to_msg()
        msg.header.frame_id = self.frame_id
        msg.info.resolution = self.resolution
        msg.info.width = self.array_shape[1]
        msg.info.height = self.array_shape[0]
        msg.info.origin.position.x = self.boundary[0]
        msg.info.origin.position.y = self.boundary[1]
        msg.info.origin.orientation.x = 0.0
        msg.info.origin.orientation.y = 0.0
        msg.info.origin.orientation.z = 0.0
        msg.info.origin.orientation.w = 1.0
        msg.data = self.data.flatten().astype(int).tolist()
        ##### YOUR CODE ENDS HERE   ##### # noqa: E266
        return msg

    def is_occupied(self, x: np.array, y: np.array, *, threshold=50) -> bool:
        """
        Check whether the given cells are occupied.

        Inputs:
            x           numpy array of x values
            y           numpy array of y values
        Optional Inputs:
            threshold   minimum value to consider a cell occupied (default 50)
        Outputs:
            occupied    np.array of bool values
        """
        ##### YOUR CODE STARTS HERE ##### # noqa: E266
        # TODO Check for occupancy in the map based on the input type
        occupied = np.zeros_like(x, dtype='bool')
        rows, cols = self.xy2sub(x, y)
        valid_rows = (rows >= 0) & (rows < self.array_shape[0])
        valid_cols = (cols >= 0) & (cols < self.array_shape[1])
        valid = valid_rows & valid_cols
        occupied[valid] = self.data[rows[valid], cols[valid]] >= threshold
        ##### YOUR CODE ENDS HERE   ##### # noqa: E266
        return occupied

    def where_occupied(self, *, format='xy', threshold=50) -> np.array:
        """
        Find the locations of all cells that are occupied.

        Optional Inputs:
            format      requested format of the returned data: 'xy', 'rc', 'ind' (default 'xy')
            threshold   minimum value to consider a cell occupied (default 50)
        Outputs:
            locations   np.array with the locations of occupied cells in the requested format
        """
        # Check that requested format is valid
        if format not in ('xy', 'rc', 'ind'):
            raise Exception(f'Requested format {format} invalid, must be xy, rc, or ind')
        ##### YOUR CODE STARTS HERE ##### # noqa: E266
        # TODO Check for occupancy in the map based on the input type
        locations = np.zeros(2)
        rows, cols = np.where(self.data >= threshold)
        if format == 'rc':
            locations = np.vstack((rows, cols)).T
        elif format == 'ind':
            locations = self.sub2ind(rows, cols)
        elif format == 'xy':
            x, y = self.sub2xy(rows, cols)
            locations = np.vstack((x, y)).T
        ##### YOUR CODE ENDS HERE   ##### # noqa: E266
        return locations
