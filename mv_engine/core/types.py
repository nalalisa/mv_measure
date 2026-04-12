from __future__ import annotations

from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]
UInt8Array: TypeAlias = NDArray[np.uint8]
Point3D: TypeAlias = NDArray[np.float64]
Point2D: TypeAlias = NDArray[np.float64]
Vector3D: TypeAlias = NDArray[np.float64]
PointCloud3D: TypeAlias = NDArray[np.float64]
PointCloud2D: TypeAlias = NDArray[np.float64]
Matrix3D: TypeAlias = NDArray[np.float64]
