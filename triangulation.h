#pragma once

#include "pose.h"
#include "map_point.h"

#include <opencv2/core.hpp>
#include <vector>

// Result of triangulating a set of matched point pairs.
struct TriangulationResult {
    std::vector<Eigen::Vector3d> points_3d;  // one per input pair
    std::vector<bool>            valid;      // whether the point passed all checks
    int num_valid = 0;
};

// Triangulate N matched pairs using the linear DLT method (cv::triangulatePoints).
//
// pose1, pose2 — absolute world poses of the two cameras (R_wc, t_wc convention).
// K            — 3×3 camera intrinsic matrix (same for both cameras assumed).
// min/max_depth — cheirality bounds in each camera frame (in the same unit as
//                  the pose translation, which is 1 for normalised monocular).
TriangulationResult triangulate_points(
    const std::vector<cv::Point2f>& pts1,
    const std::vector<cv::Point2f>& pts2,
    const Pose& pose1,
    const Pose& pose2,
    const cv::Mat& K,
    double min_depth = 0.01,
    double max_depth = 200.0);
