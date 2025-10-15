from geometry_msgs.msg import Transform

import numpy as np
from typing import Tuple


def transform2xyt(T: Transform) -> Tuple[float, float, float]:
    """
    Convert geometry_msgs/msg/Transform to (x, y, theta).

    Inputs:
        T: geometry_msgs/Transform object
    Outputs:
        x: position [m]
        y: position [m]
        theta: angle [rad]
    """
    ##### YOUR CODE STARTS HERE ##### # noqa: E266
    # TODO fill x, y, theta in with correct values
    x = T.translation.x
    y = T.translation.y
    theta = np.arctan2(T.rotation.z, T.rotation.w) * 2.0
    ##### YOUR CODE ENDS HERE ##### # noqa: E266
    return (x, y, theta)


def xyt2transform(x: float, y: float, theta: float) -> Transform:
    """
    Convert (x, y, theta) to geometry_msgs/Transform.

    Inputs:
        x: position [m]
        y: position [m]
        theta: angle [rad]
    Outputs:
        T: geometry_msgs/Transform object
    """
    T = Transform()
    ##### YOUR CODE STARTS HERE ##### # noqa: E266
    # TODO fill in the transform
    T.translation.x = x
    T.translation.y = y
    T.translation.z = 0.0
    T.rotation.w = np.cos(theta / 2.0)
    T.rotation.x = 0.0
    T.rotation.y = 0.0
    T.rotation.z = np.sin(theta / 2.0)
    ##### YOUR CODE ENDS HERE ##### # noqa: E266
    return T


def homogeneous2xyt(T: np.ndarray) -> Tuple[float, float, float]:
    """
    Convert homogeneous transformation matrix to (x, y, theta).

    Inputs:
        T: 3x3 numpy.array of the transformation matrix
    Outputs:
        x: position [m]
        y: position [m]
        theta: angle [rad]
    """
    ##### YOUR CODE STARTS HERE ##### # noqa: E266
    # TODO fill in x, y, theta with correct values
    x = T[0, 2]
    y = T[1, 2]
    theta = np.arctan2(T[1, 0], T[0, 0])
    ##### YOUR CODE ENDS HERE ##### # noqa: E266
    return (x, y, theta)


def xyt2homogeneous(x: float, y: float, theta: float) -> Tuple[np.ndarray]:
    """
    Convert (x, y, theta) to homogeneous transformation.

    Inputs:
        x: position [m]
        y: position [m]
        theta: angle [rad]
    Outputs:
        T: 3x3 numpy.array of the transformation matrix
    """
    ##### YOUR CODE STARTS HERE ##### # noqa: E266
    # TODO fill in with the correct formula
    T = np.zeros((3, 3))
    T[0, 0] = np.cos(theta)
    T[0, 1] = -np.sin(theta)
    T[0, 2] = x
    T[1, 0] = np.sin(theta)
    T[1, 1] = np.cos(theta)
    T[1, 2] = y
    T[2, 2] = 1.0
    ##### YOUR CODE ENDS HERE ##### # noqa: E266
    return T


def transform2homogeneous(T: Transform) -> np.ndarray:
    """
    Convert geometry_msgs/Transform to homogeneous transformation.

    Inputs:
        T: geometry_msgs/Transform object
    Outputs:
        T: 3x3 numpy.array of the transformation matrix
    """
    return xyt2homogeneous(*transform2xyt(T))


def homogeneous2transform(T: np.ndarray) -> Transform:
    """
    Convert homogeneous transformation to geometry_msgs/Transform.

    Inputs:
        T: 3x3 numpy.array of the transformation matrix
    Outputs:
        T: geometry_msgs/Transform object
    """
    return xyt2transform(*homogeneous2xyt(T))
