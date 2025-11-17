import numpy as np
from PIL import Image
import itertools

from nav2_msgs.msg import Costmap
from prm import PRM


def save_prm_graph_image(prm,
                         filename_prefix="PRM_graph",
                         img_size=(400, 400),
                         node_radius=2,
                         path=None,
                         path_radius=3):
    """
    Save a visualization of:
      - occupancy map (background)
      - PRM edges (black)
      - PRM nodes (blue)
      - optionally: a path (red polyline) + start/goal (red disks)

    `path` can be:
      - a nav_msgs/Path message, or
      - an iterable of (x, y) pairs.
    """
    ogm = prm.ogm          # OccupancyGridMap
    G = prm.graph          # networkx graph

    height, width = ogm.data.shape
    res = ogm.resolution
    xmin, ymin, xmax, ymax = ogm.boundary

    img_w, img_h = img_size

    # base image: white background
    img = [[[255, 255, 255] for _ in range(img_w)] for _ in range(img_h)]

    # --- helpers ---------------------------------------------------------

    def world_to_pixel(x, y):
        # Normalize x,y to [0,1] based on map bounds
        u = (x - xmin) / (xmax - xmin + 1e-9)
        v = (y - ymin) / (ymax - ymin + 1e-9)
        px = int(u * (img_w - 1))
        py = int((1.0 - v) * (img_h - 1))  # flip y for image coordinates
        return px, py

    def draw_pixel(px, py, color):
        if 0 <= px < img_w and 0 <= py < img_h:
            img[py][px] = list(color)

    # Simple Bresenham-style line
    def draw_line(p0, p1, color):
        x0, y0 = p0
        x1, y1 = p1
        dx = abs(x1 - x0)
        dy = -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx + dy
        while True:
            draw_pixel(x0, y0, color)
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x0 += sx
            if e2 <= dx:
                err += dx
                y0 += sy

    def draw_disk(px, py, radius, color):
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if dx * dx + dy * dy <= radius * radius:
                    draw_pixel(px + dx, py + dy, color)

    # --- 1) draw occupancy background -----------------------------------

    for r in range(height):
        for c in range(width):
            val = ogm.data[r, c]
            if val < 0:      # unknown
                color = (200, 200, 200)
            elif val >= 50:  # occupied
                color = (0, 0, 0)
            else:            # free
                color = (255, 255, 255)

            # cell center
            x = xmin + (c + 0.5) * res
            y = ymin + (r + 0.5) * res
            px, py = world_to_pixel(x, y)
            draw_pixel(px, py, color)

    # --- 2) draw PRM edges (black) --------------------------------------

    for u, v in G.edges():
        loc_u = G.nodes[u]["location"]
        loc_v = G.nodes[v]["location"]
        loc_u = loc_u.tolist() if isinstance(loc_u, np.ndarray) else loc_u
        loc_v = loc_v.tolist() if isinstance(loc_v, np.ndarray) else loc_v
        p0 = world_to_pixel(*loc_u)
        p1 = world_to_pixel(*loc_v)
        draw_line(p0, p1, (0, 0, 0))  # black

    # --- 3) draw PRM nodes (blue) ---------------------------------------

    for _, attrs in G.nodes(data=True):
        loc = attrs["location"]
        loc = loc.tolist() if isinstance(loc, np.ndarray) else loc
        px, py = world_to_pixel(*loc)
        draw_disk(px, py, node_radius, (30, 144, 255))  # dodger blue

    # --- 4) optionally draw path (red) ----------------------------------

    if path is not None:
        # Extract list of (x, y) points
        if hasattr(path, "poses"):  # nav_msgs/Path
            pts = []
            for pose_stamped in path.poses:
                pts.append(
                    (pose_stamped.pose.position.x,
                     pose_stamped.pose.position.y)
                )
        else:
            # assume iterable of (x, y) pairs
            pts = [(float(p[0]), float(p[1])) for p in path]

        if len(pts) >= 1:
            # draw red polyline
            for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
                p0 = world_to_pixel(x0, y0)
                p1 = world_to_pixel(x1, y1)
                draw_line(p0, p1, (255, 0, 0))  # red

            # draw start & goal as bigger red disks
            sx, sy = world_to_pixel(*pts[0])
            gx, gy = world_to_pixel(*pts[-1])
            draw_disk(sx, sy, path_radius, (255, 0, 0))
            draw_disk(gx, gy, path_radius, (255, 0, 0))

    # --- 5) save as PNG via Pillow --------------------------------------

    flat = list(itertools.chain.from_iterable(
        itertools.chain.from_iterable(img)
    ))
    im = Image.new("RGB", (img_w, img_h))
    im.putdata([tuple(flat[i:i+3]) for i in range(0, len(flat), 3)])
    out_name = filename_prefix + ".png"
    im.save(out_name)
    print(f"Saved PRM image to {out_name}")


def make_test_costmap_H(width: int = 7, height: int = 7, resolution: float = 1.0) -> Costmap:
    """
    Create a tiny synthetic costmap with an 'H'-shaped free corridor.

    - All cells start as occupied (100)
    - Free cells (0) form an 'H' shape:
        left leg  at x = 1, y = 1..5
        right leg at x = 5, y = 1..5
        crossbar  at y = 3, x = 1..5
    """
    cm = Costmap()
    cm.header.frame_id = "map"
    cm.metadata.resolution = resolution
    cm.metadata.size_x = width
    cm.metadata.size_y = height
    cm.metadata.origin.position.x = 0.0
    cm.metadata.origin.position.y = 0.0
    cm.metadata.origin.position.z = 0.0

    # Start fully occupied
    data = np.full((height, width), 100, dtype=np.uint8)

    # Carve the H-shaped free corridor (0 = free)
    # Note: rows = y, cols = x (row 0 is bottom or top depending on implementation,
    # but we just need consistency here)
    # Left vertical leg
    data[0:7, 1] = 0
    # Right vertical leg
    data[0:7, 5] = 0
    # Crossbar
    data[3, 0:7] = 0

    cm.data = data.flatten().tolist()
    return cm


def test_valid_edge():

    costmap = make_test_costmap_H()
    prm = PRM(costmap, num_points=10, connection_radius=1.5, step_size=0.1)
    ogm = prm.ogm
    print("OGM data (rows=y, cols=x):")
    print(ogm.data)
    # Edge that clearly crosses obstacle -> should be invalid
    p_block_left = np.array([1.5, 1.5])
    p_block_right = np.array([5.5, 1.5])
    assert not prm.valid_edge(p_block_left, p_block_right), \
        "Edge crossing obstacles should be invalid"

    # Edge that stays entirely in a known free corridor -> should be valid
    p_free_left = np.array([1.5, 3.5])
    p_free_right = np.array([5.5, 3.5])
    assert prm.valid_edge(p_free_left, p_free_right), \
        "Edge along free corridor should be valid"


def test_build_prm_graph():

    costmap = make_test_costmap_H()
    num_points = 1000
    prm = PRM(costmap, num_points, connection_radius=1, step_size=0.2)
    save_prm_graph_image(
        prm,
        filename_prefix="PRM_build_graph",   # will create PRM_build_graph.png
        img_size=(400, 400),
        node_radius=2,
        path=None
    )
    # Graph should have exactly num_points nodes
    assert prm.graph.number_of_nodes() == num_points, \
        f"PRM graph has {prm.graph.number_of_nodes()} nodes, expected {num_points}"

    # All nodes should lie in free space
    for node_id, loc in prm.graph.nodes.data("location"):
        occ = prm.ogm.is_occupied(np.array([loc[0]]), np.array([loc[1]]))[0]
        assert not occ, f"Node {node_id} at {loc} lies in occupied cell"

    # There should be at least some edges
    assert prm.graph.number_of_edges() > 0, "PRM graph has no edges; expected some connectivity"


def test_prm_query_path():

    costmap = make_test_costmap_H()
    num_points = 300
    prm = PRM(costmap, num_points, connection_radius=0.5, step_size=0.1)

    # Start on left leg of H, goal on right leg
    start = np.array([1.5, 1.5])
    goal = np.array([5.5, 5.5])

    path = prm.query(start, goal)
    save_prm_graph_image(
        prm,
        filename_prefix="PRM_query_path",    # -> PRM_query_path.png
        img_size=(400, 400),
        node_radius=2,
        path=path,                       # this triggers red overlay
        path_radius=4                        # a bit larger than node dots
    )
    # Path should contain at least a start and end pose
    assert len(path.poses) >= 2, f"PRM query returned path with {len(path.poses)} poses"

    sx = path.poses[0].pose.position.x
    sy = path.poses[0].pose.position.y
    gx = path.poses[-1].pose.position.x
    gy = path.poses[-1].pose.position.y

    # Start of path should be close to requested start
    start_err = np.hypot(sx - start[0], sy - start[1])
    # End of path should be close to requested goal
    goal_err = np.hypot(gx - goal[0], gy - goal[1])

    assert start_err < 1.0, f"Path start too far from requested start: {start_err:.3f} m"
    assert goal_err < 1.0, f"Path end too far from requested goal: {goal_err:.3f} m"
