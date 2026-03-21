# Refactor Summary — RGB-D → Monocular HW3 SLAM

## Overview

This project was refactored from a 1912-line, single-file RGB-D SLAM prototype into a
modular monocular visual-odometry pipeline suited for Homework 3.
The old codebase used depth images at every step: to build 3D-2D correspondences for
`solvePnPRansac`, to backproject a dense map from each frame's depth grid, and to reject
frames whose keypoints had no valid depth reading.
The new architecture removes that dependency entirely: two-view epipolar geometry
(`findEssentialMat` + `recoverPose`) is the primary motion estimator, triangulation builds
the 3D map, and PnP is retained only as an optional secondary step when existing
map-point observations are available.

---

## Old Design Problems

| # | Problem | Detail |
|---|---------|--------|
| 1 | Single 1912-line monolith | All structs, algorithms, and I/O in `main.cpp`; nothing reusable |
| 2 | RGB-D backbone | Primary motion estimation was `solvePnPRansac` driven by depth-derived 3D points |
| 3 | Epipolar geometry diagnostic-only | `findFundamentalMat` was called purely to compute an error number, never to drive pose |
| 4 | No triangulation | 3D map was built by depth backprojection (every 14th pixel of the depth image) |
| 5 | `MapPoint` had no observations | Just `{id, position, valid}` — no way to find which frame/keypoint created a point |
| 6 | `MapData` was two plain vectors | No observation links, no managed ownership, no lookup helpers |
| 7 | Depth required to process any frame | Loader skipped frames with no matching depth timestamp; impossible to run monocular |
| 8 | Duplicate key-event blocks | Identical pause/speed/quit handling copied three times inside `main()` |
| 9 | All tracking state in `main()` | `consecutive_rejects`, `last_accepted_frame`, `recent_accepted_steps` as raw locals |

---

## New Module Structure

| File | Responsibility |
|------|----------------|
| `pose.h` | `Pose` struct (R_wc, t_wc convention) with compose / inverse / from_relative / from_pnp_cv helpers |
| `frame.h` | Monocular `Frame` — image data, keypoints, descriptors, pose; **no depth fields** |
| `map_point.h` | `MapPoint` with `Observation{frame_id, keypoint_idx}` list for back-reference |
| `slam_map.h/.cpp` | `SlamMap` class owning all frames and map points; `build_kp_to_mp_index()` lookup |
| `feature_matcher.h/.cpp` | `extract_features()`, `match_features()`, `MatchInfo` (BF + Lowe + distance filter) |
| `epipolar_geometry.h/.cpp` | `estimate_epipolar()` via `findEssentialMat` RANSAC; `recover_pose_from_essential()`; `compute_mean_epipolar_error()` |
| `triangulation.h/.cpp` | `triangulate_points()` — DLT via `cv::triangulatePoints`, projection matrices from absolute poses, cheirality check |
| `pnp_relocalizer.h/.cpp` | `solve_pnp()` wrapping `solvePnPRansac` (OpenCV 4.13 crash guard kept); `compute_mean_reprojection_error()` |
| `diagnostics.h/.cpp` | `StepDiagnostics` (monocular fields), `evaluate_motion_acceptance()`, CSV / jump-candidates / summary output |
| `optimization.h/.cpp` | Scaffold stubs: `local_bundle_adjustment()` and `full_bundle_adjustment()` — **no-op for now** |
| `frontend.h/.cpp` | `Frontend::process()` — 7-step monocular pipeline per frame pair; owns ORB, K, tracking state |
| `viewer.h/.cpp` | `Viewer` class — Pangolin 3D window, OpenCV keypoint window, `save_top_down_map_image()` |
| `main.cpp` | Orchestration only (~160 lines): dataset load, loop, key events, final file output |
| `CMakeLists.txt` | Updated to multi-TU build, version 0.2.0 |

---

## What Is Already Aligned with HW3

- No depth sensor or depth image is required anywhere in the normal pipeline
- Dataset loading reads RGB-only (`rgb.txt`); `depth.txt` is not touched
- `findEssentialMat` + `recoverPose` is the **primary motion-estimation path**
- The 3D map is produced entirely by **triangulation**, not depth backprojection
- `MapPoint` stores full observation lists — every frame and keypoint index that saw it
- PnP is isolated to `pnp_relocalizer.cpp` and invoked as an optional secondary step
  only when existing map-point observations are available in the current frame pair
- `optimization.h/.cpp` scaffolding is in place so bundle adjustment can be added
  without touching any other module
- Pangolin 3D viewer and top-down map PNG export continue to work unchanged
- Diagnostics CSV, jump-candidates file, and summary file are preserved with
  updated monocular field names
- Code is modular: 14 translation units, each with a single clearly named responsibility
- Zero compiler warnings on a clean build (AppleClang, OpenCV 4.13, Pangolin)

---

## What Still Remains To Implement

| Item | Status | Notes |
|------|--------|-------|
| Bundle adjustment | Scaffold only | `optimization.cpp` stubs; add g2o / Ceres / manual Gauss-Newton |
| Metric scale | Not addressed | `recoverPose` always returns `‖t‖ = 1`; map is in arbitrary scale |
| Loop closure | Not implemented | Add a `LoopDetector` module calling `full_bundle_adjustment()` |
| Keyframe selection | Not implemented | Every frame is processed; add parallax-threshold gating |
| Map point culling | Not implemented | No removal of low-observation or high-reproj-error points |
| PnP spatial balancing | Not ported | The 4×3 grid balancer from the old code can be re-added in `pnp_relocalizer.cpp` |
| Reprojection optimization | Not implemented | Per-step pose refinement via direct minimisation of reprojection error |
