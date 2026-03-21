#pragma once

#include "pose.h"

#include <opencv2/core.hpp>
#include <vector>

// Result of PnP-RANSAC pose estimation.
struct PnPResult {
    cv::Mat rvec, tvec;          // raw OpenCV output (camera-from-world)
    std::vector<int> inlier_indices;  // indices into input pts3d/pts2d
    double mean_reproj_error = -1.0;
    int    num_inliers       = 0;
    bool   success           = false;
};

// Solve PnP with RANSAC.
// objectPoints — 3D world points
// imagePoints  — corresponding 2D observations in the current frame
// K            — camera intrinsics
// Returns rvec/tvec in the OpenCV convention (camera-from-world).
PnPResult solve_pnp(
    const std::vector<cv::Point3f>& object_points,
    const std::vector<cv::Point2f>& image_points,
    const cv::Mat& K,
    int   ransac_iters            = 150,
    float reprojection_threshold  = 3.0f,
    double confidence             = 0.995,
    int   min_inliers             = 10);

// Mean reprojection error (pixels) for a given pose.
// If selected_indices is non-null, only those indices are used.
double compute_mean_reprojection_error(
    const std::vector<cv::Point3f>& object_points,
    const std::vector<cv::Point2f>& image_points,
    const cv::Mat& rvec,
    const cv::Mat& tvec,
    const cv::Mat& K,
    const std::vector<int>* selected_indices = nullptr);
