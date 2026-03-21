#include "triangulation.h"

#include <opencv2/calib3d.hpp>

// Build a 3×4 projection matrix P = K * [R_cw | t_cw]
// where R_cw = R_wc^T  and  t_cw = -R_wc^T * t_wc
static cv::Mat make_projection_matrix(const Pose& pose, const cv::Mat& K) {
    // Camera-from-world rotation and translation
    Eigen::Matrix3d R_cw = pose.rotation_matrix.transpose();
    Eigen::Vector3d t_cw = -(R_cw * pose.translation_vector);

    cv::Mat Rt(3, 4, CV_64F);
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c)
            Rt.at<double>(r, c) = R_cw(r, c);
        Rt.at<double>(r, 3) = t_cw(r);
    }

    cv::Mat P;
    K.convertTo(P, CV_64F);
    return P * Rt;
}

TriangulationResult triangulate_points(
    const std::vector<cv::Point2f>& pts1,
    const std::vector<cv::Point2f>& pts2,
    const Pose& pose1,
    const Pose& pose2,
    const cv::Mat& K,
    double min_depth,
    double max_depth)
{
    TriangulationResult result;
    const int n = static_cast<int>(pts1.size());
    if (n == 0 || pts1.size() != pts2.size()) return result;

    result.points_3d.resize(n);
    result.valid.resize(n, false);

    cv::Mat P1 = make_projection_matrix(pose1, K);
    cv::Mat P2 = make_projection_matrix(pose2, K);

    cv::Mat points4d;
    cv::triangulatePoints(P1, P2, pts1, pts2, points4d);  // 4×N

    // Camera-from-world transforms for cheirality check
    Eigen::Matrix3d R1_cw = pose1.rotation_matrix.transpose();
    Eigen::Vector3d t1_cw = -(R1_cw * pose1.translation_vector);
    Eigen::Matrix3d R2_cw = pose2.rotation_matrix.transpose();
    Eigen::Vector3d t2_cw = -(R2_cw * pose2.translation_vector);

    for (int i = 0; i < n; ++i) {
        double w = points4d.at<float>(3, i);
        if (std::fabs(w) < 1e-9) continue;

        Eigen::Vector3d p_world(
            points4d.at<float>(0, i) / w,
            points4d.at<float>(1, i) / w,
            points4d.at<float>(2, i) / w);

        if (!p_world.allFinite()) continue;

        // Depth in camera 1
        double z1 = (R1_cw * p_world + t1_cw).z();
        // Depth in camera 2
        double z2 = (R2_cw * p_world + t2_cw).z();

        if (z1 < min_depth || z1 > max_depth) continue;
        if (z2 < min_depth || z2 > max_depth) continue;

        result.points_3d[i] = p_world;
        result.valid[i]     = true;
        ++result.num_valid;
    }

    return result;
}
