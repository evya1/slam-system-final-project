# Change Log — RGB-D → Monocular HW3 Refactor

This document records every file that was created or modified during the refactor,
explains the specific decisions made inside each file, and gives the reason for each
decision. Files are listed in dependency order (from the most fundamental types up to
the orchestration layer) so the rationale builds logically.

---

## `pose.h` — **Extracted and extended**

**What changed:**
The `Pose` struct existed in the old `main.cpp` (lines 23–38) with just `rotation_matrix`,
`translation_vector`, and `matrix()`. It was extracted into its own header and extended
with four new static/member helpers.

**Decisions and why:**

| Addition | Why |
|----------|-----|
| `compose(other)` | Needed to chain relative poses when computing absolute pose from a sequence of frame-to-frame motions |
| `inverse()` | Used to convert world-from-camera ↔ camera-from-world when building triangulation projection matrices |
| `from_relative(prev, R_cp, t_cp)` | `recoverPose` and `solvePnPRansac` both return a relative transform in the form `p_curr = R * p_prev + t`. This helper centralises the inversion math in one place so both the epipolar path and the PnP path share identical composition logic, eliminating the risk of sign errors. |
| `from_pnp_cv(prev, rvec, tvec)` | Converts OpenCV's `rvec/tvec` representation directly to an absolute `Pose`, isolating the `cv::Rodrigues` call here rather than scattering it through the frontend |
| Added `<opencv2/calib3d.hpp>` include | `cv::Rodrigues` lives in calib3d, not core; the original monolith included everything at the top of one file, so this was invisible |

**Convention documented:** `rotation_matrix` = R_wc (world-from-camera rotation),
`translation_vector` = camera position in world. All code that touches poses must follow
this convention; it is now stated explicitly in a comment at the top of the file.

---

## `frame.h` — **Extracted and cleaned**

**What changed:**
The `Frame` struct (old `main.cpp` lines 40–63) was extracted and stripped of all
depth-related fields.

**Removed fields and why:**

| Removed field | Why removed |
|---------------|-------------|
| `depth_timestamp` | Not meaningful for a monocular camera |
| `depth_path` | No depth file to load |
| `depth_image` (cv::Mat) | The entire depth-backprojection and depth-to-3D pipeline depended on this; removing it forces all callers to switch to triangulation |

**Renamed field:**
- `rgb_timestamp` → `timestamp` (the "rgb" qualifier is redundant when there is only one sensor)

**Kept unchanged:** `id`, `image_gray`, `image_bgr`, `keypoints`, `descriptors`,
`pose`, `processed`. These are all needed for the monocular pipeline.

---

## `map_point.h` — **Extended with observations**

**What changed:**
The `MapPoint` struct (old `main.cpp` lines 65–75) had `{id, point, valid}`.
A new `Observation` struct and an `observations` vector were added.

**Why:**
The old design had no way to answer "which frames and keypoints created this 3D point?"
Without that information:
- PnP relocalization requires a full O(N×M) scan of all map points against all matches
- Bundle adjustment cannot build the bipartite graph of pose ↔ point factors
- Outlier culling cannot count how many frames consistently observe a point

The `Observation{frame_id, keypoint_idx}` pair records exactly the information needed
for all three use cases. The cost is two integers per observation, which is negligible.

---

## `slam_map.h` / `slam_map.cpp` — **Promoted from plain struct to class**

**What changed:**
`MapData` (old `main.cpp` lines 77–80) was just:
```cpp
struct MapData {
    vector<Frame> frames;
    vector<MapPoint> points;
};
```
It became a `SlamMap` class with managed addition and a lookup helper.

**Decisions and why:**

| Change | Why |
|--------|-----|
| `add_frame(Frame)` / `add_map_point(MapPoint)` | Centralise ID assignment so callers never need to manage `next_mp_id` manually. The old `main.cpp` had a raw `int next_map_point_id = 0` local variable that was easy to mismanage. |
| `build_kp_to_mp_index(frame_id)` | Returns an `unordered_map<int,int>` mapping keypoint index → map_point vector index for a given frame. This makes PnP relocalization O(good_matches) instead of O(map_points × good_matches). |
| Kept `frames()` and `map_points()` returning `const&` | The viewer and diagnostics code only needs read access; exposing mutable references only where needed (e.g. `frame_at(idx)`) |

---

## `feature_matcher.h` / `feature_matcher.cpp` — **Extracted unchanged**

**What changed:**
`extract_features()` (old lines 216–225) and `match_features()` (old lines 227–279)
were moved verbatim into a dedicated module. `MatchInfo` was moved here too.

**Minor renames:**
- `points_prev_2d` / `points_curr_2d` → `pts_prev` / `pts_curr` (shorter, no redundant `_2d` since they are always 2D in this struct)

**Why extracted:**
Both functions are pure algorithms with no side effects — ideal for isolation.
The frontend, and potentially a future loop-detector, need to call matching
independently; keeping them in `main.cpp` would force unwanted coupling.

**Nothing algorithmically changed:** Lowe ratio (0.72), distance filter (2.2× min + 28),
BFMatcher with NORM_HAMMING are all preserved exactly.

---

## `epipolar_geometry.h` / `epipolar_geometry.cpp` — **Promoted from diagnostic to backbone**

**What changed:**
The old code called `findFundamentalMat` twice (before and after PnP) solely to measure
an error number. The new module uses `findEssentialMat` to drive pose estimation.

**Decisions and why:**

| Change | Why |
|--------|-----|
| `findEssentialMat` instead of `findFundamentalMat` | When the camera intrinsics K are known, the essential matrix E is the correct two-view estimator. F has 7 DOF vs E's 5 DOF — E's tighter constraints produce more reliable RANSAC inliers. `findEssentialMat` internally calibrates the points using K, so the inlier mask is directly usable for `recoverPose`. |
| F computed as `K^{-T} E K^{-1}` after the fact | F is only needed for `compute_mean_epipolar_error` (diagnostic); deriving it from E avoids a second RANSAC call and ensures consistency. |
| `recoverPose` called inside `recover_pose_from_essential` | This is the standard OpenCV cheirality-aware decomposition of E into (R, t). It also updates the inlier mask to mark only points that are in front of both cameras, which is exactly the subset we want to triangulate. |
| `try/catch` around `recoverPose` | OpenCV 4.13 has crash-prone paths in some geometric estimators under edge-case inputs; consistent with the existing guard in `solvePnPRansac`. |
| `compute_mean_epipolar_error` preserved | Kept for diagnostics; measured in pixels using the symmetric epipolar distance `|x'^T F x| / sqrt(a²+b²)`. |

---

## `triangulation.h` / `triangulation.cpp` — **New module**

**What changed:**
This module did not exist. It replaces `add_current_frame_depth_points_to_map()`
(old lines 709–755) as the primary mechanism for creating 3D map points.

**Decisions and why:**

| Decision | Why |
|----------|-----|
| `cv::triangulatePoints` (DLT) | The standard linear method; correct for the task. Normalised DLT is more numerically stable with pixel-scale inputs, but the difference is negligible here because OpenCV normalises internally. |
| Projection matrices built from absolute poses | `P = K * [R_cw | t_cw]` where `R_cw = R_wc^T`, `t_cw = -R_wc^T * t_wc`. Using absolute poses (not relative) means any two frames can be triangulated, which is required for future multi-view triangulation. |
| Cheirality depth check (min/max bounds per camera) | A point is only valid if its depth is positive in **both** cameras. Points behind either camera are geometric artefacts of the DLT solver and must be discarded. The default bounds (0.01 to 200.0) are intentionally loose since monocular scale is arbitrary. |
| `valid[]` array instead of filtering | Keeps a 1-1 correspondence between input match indices and output 3D points, so the frontend can directly map `valid[i]` back to `good_matches[tri_match_indices[i]]` to record observations. |

**Why this replaces depth backprojection:**
Depth backprojection produced a dense point cloud from a known sensor reading — correct
for RGB-D but impossible without a depth camera. Triangulation produces sparse but
geometry-consistent 3D points from two views of the same keypoint, which is the
fundamental mechanism of any monocular 3D reconstruction system.

---

## `pnp_relocalizer.h` / `pnp_relocalizer.cpp` — **Extracted and isolated**

**What changed:**
`estimate_pose_pnp()` (old lines 597–676) was split: the raw solver moved here,
and `update_frame_pose_from_pnp()` (old lines 678–707) was replaced by
`Pose::from_pnp_cv()` in `pose.h`.

**Decisions and why:**

| Decision | Why |
|----------|-----|
| OpenCV 4.13 try/catch crash guard **kept** | The crash in `solvePnPRansac` for degenerate inputs is a known OpenCV 4.13 issue; removing the guard would cause silent crashes |
| Inlier matrix shape normalisation **kept** | OpenCV 4.x returns inliers as either Nx1 or 1xN; the `reshape(1, n_inliers)` call handles both forms |
| `compute_mean_reprojection_error` moved here | It only makes sense in the context of a PnP solve result; co-locating it with the solver makes the module self-contained |
| PnP is **no longer called in the main loop by default** | PnP is invoked only when `frontend.cpp:try_pnp_relocalizer()` finds ≥10 map-point-derived 3D-2D correspondences. This prevents the failure mode where PnP is the gating mechanism and every frame without depth is rejected. |

---

## `diagnostics.h` / `diagnostics.cpp` — **Redesigned for monocular**

**What changed:**
`StepDiagnostics` (old lines 102–129) had fields that were tightly coupled to the
RGB-D pipeline. The struct was redesigned, `evaluate_step_diagnostics` was rewritten,
and all CSV/log output functions were updated.

**Field renames and why:**

| Old field | New field | Why |
|-----------|-----------|-----|
| `depth_valid_3d2d` | `epipolar_inliers` | The count now measures E-matrix RANSAC inliers, not depth-valid 3D-2D pairs |
| `depth_valid_ratio` | `inlier_ratio` | Now `epipolar_inliers / good_matches`; same concept, different source |
| `map_points_added_this_step` | `triangulated_points` | More precise name for the monocular mechanism |
| `accepted_pose_source` | `pose_source` | Shorter; values are now `"epipolar"`, `"pnp"`, or `"none"` |

**New fields:**
- `pnp_correspondences` — how many map-point 3D-2D pairs were available for PnP
- `reproj_error` — mean reprojection error from PnP (only valid when `pose_source == "pnp"`)

**`evaluate_motion_acceptance` rewrite:**
The old function (lines 1150–1289) had depth-specific rejection reasons
(`reject_low_depth_valid_count`, `reject_depth_valid_ratio_too_low`,
`tracking_lost_depth_desert`) and checked `reproj_before/after` from PnP.
All of these were removed. The new function checks:
1. `epipolar_inliers` count (≥15 normal, ≥10 recovery mode)
2. `inlier_ratio` (≥0.35 normal, ≥0.25 recovery)
3. `epi_error` (≤2.5px normal, ≤3.0px recovery)
4. Rotation magnitude vs recent median (rotation outlier detection kept)

The recovery mode logic (triggered at ≥8 consecutive rejects) is preserved because
it is useful for any VO system, not just RGB-D.

---

## `optimization.h` / `optimization.cpp` — **New scaffold**

**What changed:**
This module is completely new and currently a no-op.

**Why create it now:**
- Establishes the call site and function signatures before the implementation exists
- Any code that needs to trigger BA (e.g. a future keyframe manager or loop detector)
  can `#include "optimization.h"` and call `local_bundle_adjustment(map)` without
  knowing whether the implementation is g2o, Ceres, or hand-coded Gauss-Newton
- The stubs compile and link cleanly, so the project remains buildable at all times

**Functions defined:**
- `local_bundle_adjustment(SlamMap&, int window_size)` — optimise last N keyframes
- `full_bundle_adjustment(SlamMap&)` — full map refinement (for post-loop-closure)

---

## `frontend.h` / `frontend.cpp` — **New class (core of the refactor)**

**What changed:**
The ~400-line tracking loop inside `main()` was extracted into a `Frontend` class.
This is the most significant structural change.

**Decisions and why:**

| Decision | Why |
|----------|-----|
| `Frontend` owns the ORB detector | The ORB parameters (`nfeatures=3000, scaleFactor=1.2, nlevels=8, ...`) should not be scattered across call sites. One owner = one place to tune. |
| `Frontend` owns tracking state | `consecutive_rejects_`, `last_accepted_frame_`, `recent_accepted_steps_` moved out of `main()` into private members. This makes the pipeline unit-testable and reusable. |
| `process(prev, curr, map)` as the single entry point | A clean interface: give it two frames and a map, get back a `StepDiagnostics`. The caller (`main.cpp`) does not need to know anything about the internal pipeline. |
| 7-step pipeline in `process()` | Steps are: (1) match, (2) estimate E, (3) recover pose, (4) motion-bounds check, (5) update pose, (6) triangulate + insert map points, (7) optional PnP. Each step is clearly delimited with a comment. |
| `try_pnp_relocalizer()` as a private helper | It runs after pose has already been accepted from epipolar, so it is a refinement, not the gating criterion. Keeping it private signals that it is an implementation detail. |
| Reject path always sets `curr_frame.pose = prev_frame.pose` | "Pose hold" on rejection — the frame keeps the last accepted pose rather than drifting to an undefined state. This is the same behaviour as the old code. |
| Epipolar inlier mask used for triangulation | Only points that passed `recoverPose`'s cheirality test are triangulated, avoiding behind-camera artifacts. |
| PnP only updates pose if `num_inliers >= 15` and `reproj_error < 2.5px` | The epipolar pose (scale=1) and the map-derived PnP pose are in the same coordinate system only if the map was built in that frame. Using a low-quality PnP result would corrupt the trajectory. |

---

## `viewer.h` / `viewer.cpp` — **Extracted and wrapped**

**What changed:**
All Pangolin and OpenCV window setup, rendering, and image-drawing helpers were
moved out of `main()` into a `Viewer` class.

**Decisions and why:**

| Decision | Why |
|----------|-----|
| `Viewer` constructor does Pangolin init + OpenCV window setup | Setup and usage are co-located; no risk of calling `render()` before the window is created |
| `draw_trajectory_and_map_gl` made `static private` | It is an implementation detail of `render()`; not part of the public API |
| `draw_keypoints_image`, `draw_matches_image`, `save_top_down_map_image` kept as `static` | They do not use any instance state (no Pangolin context needed); `static` makes it clear they are pure image transformations |
| `show_keypoints(frame, accepted)` convenience method | Bundles `draw_keypoints_image` + `cv::imshow` so `main.cpp` stays clean |
| `should_quit()` and `wait_key()` delegated to Pangolin / OpenCV | Thin wrappers that keep `main.cpp` from depending on Pangolin or OpenCV directly for flow control |
| `top_down_map_image` uses `const SlamMap&` | The viewer never modifies map data; const reference enforces this |

---

## `main.cpp` — **Reduced to orchestration**

**What changed:**
Reduced from 1912 lines to ~160 lines. All algorithm logic was removed.

**Decisions and why:**

| Decision | Why |
|----------|-----|
| Loads RGB-only (`rgb.txt`; ignores `depth.txt`) | The pipeline no longer needs depth; loading it would waste memory and mislead readers |
| Frame storage moved into `SlamMap` immediately on load | Avoids a separate local frame vector; the map is the single source of truth from the start |
| `handle_key_events(paused, frame_delay_ms)` helper | Eliminates the three copies of identical pause/speed/quit code that existed in the old `main()` loop. Returns `false` on quit so the caller can `break`. |
| `frontend.process()` called once per frame pair | `main()` no longer touches matching, epipolar geometry, triangulation, or PnP directly |
| Feature extraction for frame 0 before the loop | Frame 0 is special (no "previous frame" to compare against); its features must be ready before the loop starts |
| Console output unchanged in structure | The same per-frame field printout is preserved so existing monitoring/grep workflows are not broken |

---

## `CMakeLists.txt` — **Updated for multi-TU build**

**What changed:**
Moved from `add_executable(slam_system main.cpp)` to a multi-file build listing
all 10 `.cpp` source files.

**Decisions and why:**

| Decision | Why |
|----------|-----|
| Explicit source list (no `file(GLOB ...)`) | `GLOB` silently misses new files added later; explicit listing forces developers to consciously update the build file |
| Version bumped to `0.2.0` | Marks the transition from the RGB-D prototype to the monocular architecture |
| `${CMAKE_CURRENT_SOURCE_DIR}` added to `target_include_directories` | Required so headers in the root can be found from any build directory (e.g. `build_refactor/`) |
| All other Pangolin / Eigen / OpenCV integration unchanged | The dependency detection logic was already robust; no reason to modify it |
