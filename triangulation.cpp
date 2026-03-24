#include "triangulation.h"

#include <opencv2/calib3d.hpp>
#include <cmath>

static cv::Mat make_projection_matrix(const Pose& pose, const cv::Mat& K) {
    return pose.projection_matrix(K);
}

// Parallax angle in degrees. Low parallax -> ill-conditioned triangulation.
static double parallax_deg(
    const Eigen::Vector3d& p_world,
    const Pose& pose1,
    const Pose& pose2)
{
    const Eigen::Vector3d& c1 = pose1.translation_vector;
    const Eigen::Vector3d& c2 = pose2.translation_vector;

    Eigen::Vector3d ray1 = (p_world - c1).normalized();
    Eigen::Vector3d ray2 = (p_world - c2).normalized();

    double cos_angle = ray1.dot(ray2);
    cos_angle = std::max(-1.0, std::min(1.0, cos_angle));
    return std::acos(cos_angle) * (180.0 / M_PI);
}

TriangulationResult triangulate_points(
    const std::vector<cv::Point2f>& pts1,
    const std::vector<cv::Point2f>& pts2,
    const Pose& pose1,
    const Pose& pose2,
    const cv::Mat& K,
    double min_depth,
    double max_depth,
    double min_parallax_deg)
{
    TriangulationResult result;
    const int n = static_cast<int>(pts1.size());
    if (n == 0 || pts1.size() != pts2.size()) return result;

    result.points_3d.resize(n);
    result.valid.resize(n, false);

    cv::Mat P1 = make_projection_matrix(pose1, K);
    cv::Mat P2 = make_projection_matrix(pose2, K);

    // triangulatePoints outputs CV_32F regardless of input; convert to CV_64F.
    cv::Mat points4d_f;
    cv::triangulatePoints(P1, P2, pts1, pts2, points4d_f);

    cv::Mat points4d;
    points4d_f.convertTo(points4d, CV_64F);   // 4×N  CV_64F

    Eigen::Matrix3d R1_cw = pose1.R_cw();
    Eigen::Vector3d t1_cw = pose1.t_cw();
    Eigen::Matrix3d R2_cw = pose2.R_cw();
    Eigen::Vector3d t2_cw = pose2.t_cw();

    for (int i = 0; i < n; ++i) {
        double w = points4d.at<double>(3, i);
        if (std::fabs(w) < 1e-9) continue;

        Eigen::Vector3d p_world(
            points4d.at<double>(0, i) / w,
            points4d.at<double>(1, i) / w,
            points4d.at<double>(2, i) / w);

        if (!p_world.allFinite()) continue;

        double z1 = (R1_cw * p_world + t1_cw).z();
        double z2 = (R2_cw * p_world + t2_cw).z();

        if (z1 < min_depth || z1 > max_depth) continue;
        if (z2 < min_depth || z2 > max_depth) continue;

        // Reject near-degenerate triangulation
        if (parallax_deg(p_world, pose1, pose2) < min_parallax_deg) continue;

        result.points_3d[i] = p_world;
        result.valid[i]     = true;
        ++result.num_valid;
    }

    return result;
}
