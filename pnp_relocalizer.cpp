#include "pnp_relocalizer.h"

#include <opencv2/calib3d.hpp>
#include <iostream>
#include <cmath>

PnPResult solve_pnp(
    const std::vector<cv::Point3f>& object_points,
    const std::vector<cv::Point2f>& image_points,
    const cv::Mat& K,
    int    ransac_iters,
    float  reprojection_threshold,
    double confidence,
    int    min_inliers)
{
    PnPResult result;

    if (static_cast<int>(object_points.size()) < min_inliers ||
        object_points.size() != image_points.size()) {
        return result;
    }

    cv::Mat inliers_mat;
    cv::Mat dist_coeffs = cv::Mat::zeros(4, 1, CV_64F);

    bool ok = false;
    try {
        ok = cv::solvePnPRansac(
            object_points, image_points,
            K, dist_coeffs,
            result.rvec, result.tvec,
            false,
            ransac_iters, reprojection_threshold, confidence,
            inliers_mat,
            cv::SOLVEPNP_ITERATIVE);
    } catch (const cv::Exception& e) {
        std::cerr << "solvePnPRansac exception: " << e.what() << "\n";
        return result;
    }

    // OpenCV 4.x may return inliers as Nx1 or 1xN — normalise to flat list
    int n_inliers = inliers_mat.rows * inliers_mat.cols;
    if (!ok || n_inliers < min_inliers) return result;

    cv::Mat flat = inliers_mat.reshape(1, n_inliers);
    result.inlier_indices.reserve(n_inliers);
    for (int i = 0; i < n_inliers; ++i)
        result.inlier_indices.push_back(flat.at<int>(i, 0));
    result.num_inliers = n_inliers;

    result.mean_reproj_error = compute_mean_reprojection_error(
        object_points, image_points,
        result.rvec, result.tvec, K,
        &result.inlier_indices);

    result.success = true;
    return result;
}

double compute_mean_reprojection_error(
    const std::vector<cv::Point3f>& object_points,
    const std::vector<cv::Point2f>& image_points,
    const cv::Mat& rvec,
    const cv::Mat& tvec,
    const cv::Mat& K,
    const std::vector<int>* selected_indices)
{
    if (object_points.empty() || image_points.empty()) return -1.0;

    cv::Mat dist_coeffs = cv::Mat::zeros(4, 1, CV_64F);
    std::vector<cv::Point2f> projected;
    cv::projectPoints(object_points, rvec, tvec, K, dist_coeffs, projected);

    double total = 0.0;
    int    count = 0;

    auto accumulate = [&](int idx) {
        if (idx < 0 || idx >= static_cast<int>(projected.size())) return;
        double dx = projected[idx].x - image_points[idx].x;
        double dy = projected[idx].y - image_points[idx].y;
        total += std::sqrt(dx * dx + dy * dy);
        ++count;
    };

    if (!selected_indices) {
        for (int i = 0; i < static_cast<int>(projected.size()); ++i) accumulate(i);
    } else {
        for (int idx : *selected_indices) accumulate(idx);
    }

    return count > 0 ? total / count : -1.0;
}
