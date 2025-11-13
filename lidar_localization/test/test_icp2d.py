import numpy as np

from lidar_localization.icp2d import ICP2D


def _theta_from_T(T: np.ndarray) -> float:
    """Return yaw (radians) from a 3x3 SE(2) matrix."""
    return float(np.arctan2(T[1, 0], T[0, 0]))


def _block(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Homogeneous 3x3 from R(2x2), t(2x1)."""
    T = np.eye(3, dtype=np.float64)
    T[:2, :2] = R
    T[:2, 2:3] = t
    return T


def test_icp2d():
    np.random.seed(0)
    A = np.random.rand(2, 200).astype(np.float64)

    theta = np.deg2rad(15.0)
    R_true = np.array([[np.cos(theta), -np.sin(theta)],
                       [np.sin(theta),  np.cos(theta)]], dtype=np.float64)
    t_true = np.array([[1.0], [0.5]], dtype=np.float64)
    B = R_true @ A + t_true

    icp = ICP2D()
    icp.set_map_points(B.astype(np.float64))
    T_est, err, iters = icp.icp(A.astype(np.float64), max_iterations=50, tolerance=1e-6)

    T_true = _block(R_true, t_true)

    # tolerances
    ang_err = abs(_theta_from_T(T_est) - _theta_from_T(T_true))
    ang_err = min(ang_err, 2*np.pi - ang_err)  # wrap
    trans_err = np.linalg.norm(T_est[:2, 2] - T_true[:2, 2])

    assert ang_err < 1e-3, f"Angle error too large: {ang_err:.6f} rad"
    assert trans_err < 1e-3, f"Translation error too large: {trans_err:.6f} m"
    assert err < 1e-2, f"Mean NN error should be small, got {err:.6f}"
