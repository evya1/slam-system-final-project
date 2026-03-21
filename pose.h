#pragma once

#include <Eigen/Dense>
#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>

// Camera pose convention:
//   rotation_matrix (R_wc):  world-from-camera rotation
//   translation_vector (t_wc): camera origin in world coordinates
//
// World ↔ camera transforms:
//   p_world = R_wc * p_camera + t_wc
//   p_camera = R_wc^T * (p_world - t_wc)
struct Pose {
    Eigen::Matrix3d rotation_matrix    = Eigen::Matrix3d::Identity();
    Eigen::Vector3d translation_vector = Eigen::Vector3d::Zero();

    Pose() = default;
    Pose(const Eigen::Matrix3d& R, const Eigen::Vector3d& t)
        : rotation_matrix(R), translation_vector(t) {}

    // 4×4 homogeneous transform  [R | t]
    Eigen::Matrix4d matrix() const {
        Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
        T.block<3, 3>(0, 0) = rotation_matrix;
        T.block<3, 1>(0, 3) = translation_vector;
        return T;
    }

    // Compose: result = this * other
    // Interprets other as a pose expressed in this frame.
    Pose compose(const Pose& other) const {
        return {rotation_matrix * other.rotation_matrix,
                translation_vector + rotation_matrix * other.translation_vector};
    }

    Pose inverse() const {
        Eigen::Matrix3d Rt = rotation_matrix.transpose();
        return {Rt, -(Rt * translation_vector)};
    }

    Eigen::Vector3d transform_to_world(const Eigen::Vector3d& p_cam) const {
        return rotation_matrix * p_cam + translation_vector;
    }

    Eigen::Vector3d transform_to_camera(const Eigen::Vector3d& p_world) const {
        return rotation_matrix.transpose() * (p_world - translation_vector);
    }

    static Pose identity() { return {}; }

    // Build absolute pose from a previous absolute pose and the relative motion
    // returned by solvePnP / recoverPose.
    //
    // R_cp, t_cp satisfy:  p_curr_cam = R_cp * p_prev_cam + t_cp
    // (This is what OpenCV's solvePnP and recoverPose output.)
    static Pose from_relative(const Pose& prev_pose,
                               const Eigen::Matrix3d& R_cp,
                               const Eigen::Vector3d& t_cp) {
        // Invert relative motion to get "prev-cam as seen in curr-cam" → pose of curr in world
        Pose relative;
        relative.rotation_matrix    =  R_cp.transpose();
        relative.translation_vector = -(R_cp.transpose() * t_cp);
        return prev_pose.compose(relative);
    }

    // Convenience: convert OpenCV rvec/tvec (solvePnP output) to a relative Pose
    // and then compute absolute pose.
    static Pose from_pnp_cv(const Pose& prev_pose,
                              const cv::Mat& rvec,
                              const cv::Mat& tvec) {
        cv::Mat R_cv;
        cv::Rodrigues(rvec, R_cv);

        cv::Mat R64, t64;
        R_cv.convertTo(R64, CV_64F);
        tvec.convertTo(t64, CV_64F);

        Eigen::Matrix3d R_cp;
        Eigen::Vector3d t_cp;
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) R_cp(r, c) = R64.at<double>(r, c);
            t_cp(r) = t64.at<double>(r, 0);
        }

        return from_relative(prev_pose, R_cp, t_cp);
    }
};
