#pragma once

#include "pose.h"
#include "slam_map.h"

#include <opencv2/core.hpp>
#include <vector>

// Pose-only Levenberg-Marquardt optimizer.
// Minimises sum-of-squared reprojection errors over world-frame 3D-2D correspondences.
// State: left perturbation of T_cw as [delta_phi (3), delta_rho (3)].
// Residual: projected - observed.
// Jacobian (2x6): J = J_pi * [-[x_c]_x | I_3]
struct OptimResult {
    Pose   optimized_pose;
    double reproj_before = -1.0;   // RMS reprojection error before (pixels)
    double reproj_after  = -1.0;   // RMS reprojection error after  (pixels)
    int    iterations    = 0;
    bool   converged     = false;
};

// Pose-only LM optimisation.
// pts3d -- world-frame 3D points
// pts2d -- corresponding observed 2D image points
// K     -- camera intrinsics (3x3)
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
