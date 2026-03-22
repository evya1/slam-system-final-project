#pragma once

#include "pose.h"

#include <opencv2/core.hpp>
#include <vector>

// Result of estimating the essential matrix between two views.
struct EpipolarResult {
    cv::Mat E;                       // 3×3 essential matrix
    cv::Mat F;                       // 3×3 fundamental matrix  (K^{-T} E K^{-1})
    std::vector<uchar> inlier_mask;  // 1 = inlier, 0 = outlier (same length as input)
    int    num_inliers     = 0;
    double mean_epi_error  = -1.0;
    bool   success         = false;
};

// Result of recovering relative pose from an essential matrix.
struct PoseRecoveryResult {
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    Eigen::Vector3d t = Eigen::Vector3d::Zero();
    // Updated inlier mask (recoverPose further removes points behind cameras)
    std::vector<uchar> inlier_mask;
    int  num_inliers = 0;
    bool success     = false;
};

// Estimate the essential matrix using findEssentialMat (5-pt RANSAC).
// Returns inlier_mask aligned with pts1/pts2.
EpipolarResult estimate_epipolar(
    const std::vector<cv::Point2f>& pts1,
    const std::vector<cv::Point2f>& pts2,
    const cv::Mat& K,
    double ransac_threshold = 1.0,
    double confidence       = 0.999);

// Recover relative pose (R, t) from E using cheirality check.
// Uses the inlier_mask from EpipolarResult as a starting mask and updates it.
PoseRecoveryResult recover_pose_from_essential(
    const EpipolarResult& epi,
    const std::vector<cv::Point2f>& pts1,
    const std::vector<cv::Point2f>& pts2,
    const cv::Mat& K);

// Mean point-to-epipolar-line distance in image 2: |x2^T F x1| / ||(Fx1)_{0:1}||.
double compute_mean_epipolar_error(
    const std::vector<cv::Point2f>& pts1,
    const std::vector<cv::Point2f>& pts2,
    const cv::Mat& F);