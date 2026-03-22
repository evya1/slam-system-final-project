#pragma once

#include "pose.h"
#include "slam_map.h"

#include <opencv2/core.hpp>
#include <vector>

// =============================================================================
// Pose-only Levenberg-Marquardt optimizer
//
// Minimises the sum of squared reprojection errors over a set of 3D-2D
// correspondences, optimising only the current-frame pose.
//
// State:  δξ = [δφ (3), δρ (3)]  — left perturbation of T_cw
//   T_cw_new = exp([δφ]_×) * T_cw_old
//   ⟹  R_cw_new = δR * R_cw_old
//       t_cw_new = δR * t_cw_old + δρ
//
// Residual (2×1):  r = projected − observed  (sign consistent with J above)
//
// Jacobian (2×6) of [u, v] w.r.t. δξ:
//   J = J_π * [ -[x_c]_×  |  I_3 ]
// where J_π = [fx/z, 0, -fx*xc/z²; 0, fy/z, -fy*yc/z²] (2×3)
// =============================================================================

struct OptimResult {
    Pose   optimized_pose;
    double reproj_before = -1.0;   // RMS reprojection error before (pixels)
    double reproj_after  = -1.0;   // RMS reprojection error after  (pixels)
    int    iterations    = 0;
    bool   converged     = false;
};

// Pose-only LM optimisation.
// pts3d — world-frame 3D points
// pts2d — corresponding observed 2D image points
// K     — camera intrinsics (3×3)
OptimResult optimize_pose_lm(
    const Pose&                      initial_pose,
    const std::vector<cv::Point3f>&  pts3d,
    const std::vector<cv::Point2f>&  pts2d,
    const cv::Mat&                   K,
    int    max_iter      = 20,
    double lambda_init   = 1e-3,
    double converge_eps  = 1e-7);

// RMS reprojection error (pixels) for a given pose and correspondences.
double compute_reproj_error(
    const Pose&                      pose,
    const std::vector<cv::Point3f>&  pts3d,
    const std::vector<cv::Point2f>&  pts2d,
    const cv::Mat&                   K);
