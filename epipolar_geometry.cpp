#include "epipolar_geometry.h"

#include <opencv2/calib3d.hpp>
#include <iostream>
#include <cmath>

EpipolarResult estimate_epipolar(
    const std::vector<cv::Point2f>& pts1,
    const std::vector<cv::Point2f>& pts2,
    const cv::Mat& K,
    double ransac_threshold,
    double confidence)
{
    EpipolarResult result;

    if (pts1.size() < 8 || pts1.size() != pts2.size()) {
        return result;
    }

    result.inlier_mask.assign(pts1.size(), 0);

    try {
        result.E = cv::findEssentialMat(
            pts1, pts2, K,
            cv::RANSAC, confidence, ransac_threshold,
            result.inlier_mask);
    } catch (const cv::Exception& e) {
        std::cerr << "findEssentialMat exception: " << e.what() << "\n";
        return result;
    }

    if (result.E.empty() || result.E.rows != 3 || result.E.cols != 3) {
        return result;
    }

    result.num_inliers = cv::countNonZero(result.inlier_mask);
    if (result.num_inliers < 5) return result;

    // Compute F = K^{-T} * E * K^{-1}  for epipolar-error diagnostics
    cv::Mat K_inv;
    cv::invert(K, K_inv);
    result.F = K_inv.t() * result.E * K_inv;

    result.mean_epi_error = compute_mean_epipolar_error(pts1, pts2, result.F);
    result.success = true;
    return result;
}

PoseRecoveryResult recover_pose_from_essential(
    const EpipolarResult& epi,
    const std::vector<cv::Point2f>& pts1,
    const std::vector<cv::Point2f>& pts2,
    const cv::Mat& K)
{
    PoseRecoveryResult result;

    if (!epi.success || epi.E.empty()) return result;

    cv::Mat R_cv, t_cv;
    // Copy the inlier mask so recoverPose can further update it
    cv::Mat mask_mat(epi.inlier_mask);

    int n = 0;
    try {
        n = cv::recoverPose(epi.E, pts1, pts2, K, R_cv, t_cv, mask_mat);
    } catch (const cv::Exception& e) {
        std::cerr << "recoverPose exception: " << e.what() << "\n";
        return result;
    }

    if (n < 5) return result;

    result.num_inliers = n;
    result.inlier_mask.assign(
        mask_mat.data,
        mask_mat.data + mask_mat.total());

    R_cv.convertTo(R_cv, CV_64F);
    t_cv.convertTo(t_cv, CV_64F);

    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c)
            result.R(r, c) = R_cv.at<double>(r, c);
        result.t(r) = t_cv.at<double>(r, 0);
    }

    result.success = true;
    return result;
}

double compute_mean_epipolar_error(
    const std::vector<cv::Point2f>& pts1,
    const std::vector<cv::Point2f>& pts2,
    const cv::Mat& F)
{
    if (pts1.empty() || F.empty()) return -1.0;

    cv::Mat F64;
    F.convertTo(F64, CV_64F);

    double total = 0.0;
    int    count = 0;

    for (size_t i = 0; i < pts1.size(); ++i) {
        cv::Mat x1 = (cv::Mat_<double>(3, 1) << pts1[i].x, pts1[i].y, 1.0);
        cv::Mat line2 = F64 * x1;

        double a = line2.at<double>(0);
        double b = line2.at<double>(1);
        double c = line2.at<double>(2);
        double denom = std::sqrt(a * a + b * b);
        if (denom > 1e-12) {
            total += std::fabs(a * pts2[i].x + b * pts2[i].y + c) / denom;
            ++count;
        }
    }

    return count > 0 ? total / count : -1.0;
}
