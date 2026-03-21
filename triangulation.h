#pragma once

#include "pose.h"
#include "map_point.h"

#include <opencv2/core.hpp>
#include <vector>

// Result of triangulating a set of matched point pairs.
struct TriangulationResult {
    std::vector<Eigen::Vector3d> points_3d;  // one per input pair (invalid if !valid[i])
    std::vector<bool>            valid;
    int num_valid = 0;
};

// Triangulate N matched pairs using the linear DLT method (cv::triangulatePoints).
//
// pose1, pose2   — absolute T_wc poses (R_wc, t_wc convention).
// K              — 3×3 intrinsic matrix (same for both cameras).
// min/max_depth  — cheirality bounds in each camera frame.
// min_parallax_deg — minimum parallax angle; points with less parallax are
//                    rejected because triangulation is ill-conditioned.
TriangulationResult triangulate_points(
    const std::vector<cv::Point2f>& pts1,
    const std::vector<cv::Point2f>& pts2,
    const Pose& pose1,
    const Pose& pose2,
    const cv::Mat& K,
    double min_depth        = 0.01,
    double max_depth        = 200.0,
    double min_parallax_deg = 0.5);
