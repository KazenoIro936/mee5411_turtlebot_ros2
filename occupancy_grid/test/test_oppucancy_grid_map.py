import numpy as np
from occupancy_grid import OccupancyGridMap

from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Pose
from std_msgs.msg import Header
from builtin_interfaces.msg import Time as RosTime


def make_og_msg_2x3(frame_id="map"):
    """
    Build a 2x3 OccupancyGrid.

      height=2 rows (r=0..1), width=3 cols (c=0..2), resolution=1.0
      origin at (0,0) with identity orientation
      data is row-major (r0c0, r0c1, r0c2, r1c0, r1c1, r1c2)
      Set 2nd and 4th cells to 100: indices 1 and 3
      Set last cell (index 5) to -1 (unknown, which we now treat as occupied):

         data = [0,   100, 0,
                 100, 0,   -1]
    """
    msg = OccupancyGrid()
    msg.header = Header()
    msg.header.frame_id = frame_id
    msg.header.stamp = RosTime(sec=123, nanosec=456)  # arbitrary

    msg.info.resolution = 1.0
    msg.info.width = 3
    msg.info.height = 2

    msg.info.origin = Pose()
    msg.info.origin.position.x = 0.0
    msg.info.origin.position.y = 0.0
    msg.info.origin.position.z = 0.0
    msg.info.origin.orientation.x = 0.0
    msg.info.origin.orientation.y = 0.0
    msg.info.origin.orientation.z = 0.0
    msg.info.origin.orientation.w = 1.0

    # row-major: [r0c0, r0c1, r0c2, r1c0, r1c1, r1c2]
    msg.data = [0,   100, 0,
                100, 0,   -1]  # <- last cell unknown
    return msg


def test_from_msg_basic_fields():
    msg = make_og_msg_2x3("test_map")
    ogm = OccupancyGridMap.from_msg(msg)

    # boundary = [xmin, ymin, xmax, ymax] ; with res=1, width=3, height=2
    # -> [0, 0, 3, 2]
    assert np.allclose(ogm.boundary, [0.0, 0.0, 3.0, 2.0]), f"Boundary wrong: {ogm.boundary}"
    assert ogm.resolution == 1.0, "Resolution not copied"
    assert ogm.frame_id == "test_map", "Frame id not copied"
    assert ogm.data.shape == (2, 3), f"Data shape wrong: {ogm.data.shape}"

    # Data reshaped (height, width) with values as provided
    expected = np.array([[0,   100, 0],
                         [100,   0, -1]])
    assert np.array_equal(ogm.data, expected), f"Data mismatch:\n{ogm.data}\n!=\n{expected}"


# ---------- where_occupied ----------------------------------------------------
def test_where_occupied_formats():
    msg = make_og_msg_2x3()
    ogm = OccupancyGridMap.from_msg(msg)

    # Occupied cells are (r,c) = (0,1), (1,0), and (1,2) (unknown=-1 treated as occupied)
    rows_cols_true = np.array([
        [0, 1],
        [1, 0],
        [1, 2],
    ])

    # rc format
    rc = ogm.where_occupied(format='rc', threshold=50)
    # Order is not guaranteed; sort for comparison
    rc_sorted = rc[np.lexsort((rc[:, 1], rc[:, 0]))]
    rc_true_sorted = rows_cols_true[np.lexsort((rows_cols_true[:, 1], rows_cols_true[:, 0]))]
    assert np.array_equal(rc_sorted, rc_true_sorted), f"rc occupied wrong: {rc}"

    # ind format (linear indices in row-major)
    # indices: (0,1)->1, (1,0)->3, (1,2)->5
    inds = ogm.where_occupied(format='ind', threshold=50)
    inds_sorted = np.sort(inds)
    inds_true = np.array([1, 3, 5])
    assert np.array_equal(inds_sorted, inds_true), f"ind occupied wrong: {inds}"

    # xy format (use MapConversions sub2xy mapping)
    xy = ogm.where_occupied(format='xy', threshold=50)
    # For resolution 1.0, cell centers at (c+0.5, r+0.5) from origin (0,0):
    # (r,c)=(0,1) -> (1.5,0.5)
    # (r,c)=(1,0) -> (0.5,1.5)
    # (r,c)=(1,2) -> (2.5,1.5)
    xy_sorted = xy[np.lexsort((xy[:, 1], xy[:, 0]))]
    xy_true = np.array([
        [0.5, 1.5],
        [1.5, 0.5],
        [2.5, 1.5],
    ])
    assert np.allclose(xy_sorted, xy_true), f"xy occupied wrong:\n{xy_sorted}\n!=\n{xy_true}"


# ---------- is_occupied -------------------------------------------------------
def test_is_occupied_points():
    msg = make_og_msg_2x3()
    ogm = OccupancyGridMap.from_msg(msg)

    # Query some points near cell centers:
    # (r,c)=(0,1) occupied (100)  -> center (1.5, 0.5)
    # (r,c)=(1,0) occupied (100)  -> center (0.5, 1.5)
    # (r,c)=(1,2) occupied (-1)   -> center (2.5, 1.5)  # unknown treated as occupied
    # (r,c)=(0,2) free (0)        -> center (2.5, 0.5)
    xs = np.array([1.5, 0.5, 2.5, 2.5])
    ys = np.array([0.5, 1.5, 1.5, 0.5])
    occ = ogm.is_occupied(xs, ys, threshold=50)

    # Expect first three True (two 100's and one -1), last one False (free cell)
    expected = np.array([True, True, True, False])
    assert np.array_equal(occ, expected), f"is_occupied wrong: {occ} vs {expected}"


# ---------- add_block ---------------------------------------------------------
def test_add_block_marks_cells():
    msg = make_og_msg_2x3()
    ogm = OccupancyGridMap.from_msg(msg)

    # Start with three occupied cells: (0,1), (1,0), (1,2=-1)
    # Add a block covering x:[2.0, 3.0], y:[0.0, 1.0]
    # This should mark (r,c)=(0,2) as occupied (center at x=2.5,y=0.5)
    ogm.add_block(np.array([2.0, 0.0, 3.0, 1.0]))

    # Now occupied should include (0,2) in addition to original ones
    rc = ogm.where_occupied(format='rc', threshold=50)
    rc_set = set(map(tuple, rc))
    assert (0, 2) in rc_set, f"add_block failed to mark (0,2); got {rc_set}"


# ---------- to_msg ------------------------------------------------------------
def test_to_msg_roundtrip():
    msg = make_og_msg_2x3(frame_id="map_frame")
    ogm = OccupancyGridMap.from_msg(msg)

    # set a known timestamp to check it passes through (nanosec may be lost if mocked)
    from rclpy.time import Time
    t = Time(seconds=10, nanoseconds=20)
    out = ogm.to_msg(t)

    # header
    assert out.header.frame_id == "map_frame", "frame_id mismatch"
    # stamp passed (exact equality depends on rclpy version; we check fields exist)
    assert out.header.stamp.sec == 10 and out.header.stamp.nanosec == 20, "timestamp mismatch"

    # info
    assert out.info.resolution == 1.0, "resolution mismatch"
    assert out.info.width == 3 and out.info.height == 2, "width/height mismatch"
    assert out.info.origin.position.x == 0.0, "origin x mismatch"
    assert out.info.origin.position.y == 0.0, "origin y mismatch"
    assert out.info.origin.orientation.w == 1.0, "origin orientation mismatch"

    # data flatten should match ogm.data row-major (including the -1)
    flat_expected = ogm.data.flatten().astype(int).tolist()
    data_list = list(out.data)
    assert data_list == flat_expected, f"data flatten mismatch: {data_list} vs {flat_expected}"
