# Python to C++ Porting Guide

## Mapping policy

- `NRRD` I/O is intentionally isolated in `mv_engine/io/nrrd_reader.py` so the rest of the engine only sees typed point clouds.
- `numpy.ndarray` with shape `(3,)` maps to `Eigen::Vector3d`.
- `numpy.ndarray` with shape `(N, 3)` maps to `Eigen::MatrixXd` or `std::vector<Eigen::Vector3d>`.
- Pure functions under `mv_engine.math` and `mv_engine.algo` should become namespace functions or `static` helpers in C++.
- Dataclasses under `mv_engine.core` should become `struct` or light POD-like classes.
- `MitralValveAnalyzer` should become the facade or controller entry point of the engine.
- Grouped result models (`AnnulusMetrics`, `ShapeMetrics`, `CoaptationMetrics`, `LeafletMetrics`, `MiscMetrics`) should remain grouped in C++ for API stability.

## Current external library targets

- `numpy` / `scipy.linalg` -> `Eigen3`
- `scipy.spatial.cKDTree` -> `nanoflann` or `PCL`
- `scipy.spatial.Delaunay` -> `CGAL` or a custom triangulation helper
- voxel downsampling stub -> `PCL::VoxelGrid` or custom grid filter
- `pynrrd` -> ITK, Teem, or an internal NRRD parser

## Testing policy

- Keep deterministic inputs in Python tests.
- Store expected numeric outputs at practical tolerances.
- Reuse identical fixtures in C++ tests after porting.
- Preserve the same 128 x 128 x 128 label mapping used in Python tests:
  - `0`: background
  - `1`: anterior leaflet
  - `2`: posterior leaflet
  - `3`: MV annulus
  - `4`: aortic annulus
