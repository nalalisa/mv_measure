# Python to C++ Porting Guide

## Mapping policy

- `numpy.ndarray` with shape `(3,)` maps to `Eigen::Vector3d`.
- `numpy.ndarray` with shape `(N, 3)` maps to `Eigen::MatrixXd` or `std::vector<Eigen::Vector3d>`.
- Pure functions under `mv_engine.math` and `mv_engine.algo` should become namespace functions or `static` helpers in C++.
- Dataclasses under `mv_engine.core` should become `struct` or light POD-like classes.
- `MitralValveAnalyzer` should become the facade or controller entry point of the engine.

## Current external library targets

- `numpy` / `scipy.linalg` -> `Eigen3`
- `scipy.spatial.cKDTree` -> `nanoflann` or `PCL`
- voxel downsampling stub -> `PCL::VoxelGrid` or custom grid filter

## Testing policy

- Keep deterministic inputs in Python tests.
- Store expected numeric outputs at practical tolerances.
- Reuse identical fixtures in C++ tests after porting.
