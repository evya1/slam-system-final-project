#pragma once

#include <Eigen/Dense>
#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/calib3d.hpp>

// T_wc convention: rotation_matrix = R_wc, translation_vector = t_wc (camera origin in world).
// p_world = R_wc * p_cam + t_wc;  p_cam = R_wc^T * (p_world - t_wc)
// Projection matrix: P = K * [R_cw | t_cw]  (R_cw = R_wc^T, t_cw = -R_wc^T * t_wc)
struct Pose {
    Eigen::Matrix3d rotation_matrix    = Eigen::Matrix3d::Identity(); // R_wc
    Eigen::Vector3d translation_vector = Eigen::Vector3d::Zero();     // t_wc

    Pose() = default;
    Pose(const Eigen::Matrix3d& R, const Eigen::Vector3d& t)
        : rotation_matrix(R), translation_vector(t) {}

    Eigen::Matrix3d R_cw() const { return rotation_matrix.transpose(); }
    Eigen::Vector3d t_cw() const { return -(rotation_matrix.transpose() * translation_vector); }

    // P = K * [R_cw | t_cw], 3x4 CV_64F
    cv::Mat projection_matrix(const cv::Mat& K) const {
        cv::Mat R_part, t_part;
        cv::eigen2cv(R_cw(), R_part);
        cv::eigen2cv(t_cw(), t_part);

        cv::Mat Rt;
        cv::hconcat(R_part, t_part, Rt);

        cv::Mat K64;
        K.convertTo(K64, CV_64F);
        return K64 * Rt;
    }

    // [R_wc | t_wc; 0 | 1]
    Eigen::Matrix4d matrix() const {
        Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
        T.block<3, 3>(0, 0) = rotation_matrix;
        T.block<3, 1>(0, 3) = translation_vector;
        return T;
    }

    // T_wc_result = T_wc_this * T_wc_other
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

    // For consecutive-frame epipolar motion only.
    // R_21, t_21 from recoverPose: X_cam2 = R_21 * X_cam1 + t_21. |t_21| == 1 (scale unknown).
    static Pose from_relative(const Pose &prev_pose,
                              const Eigen::Matrix3d &R_21,
                              const Eigen::Vector3d &t_21) {
        Pose delta;
        delta.rotation_matrix    =  R_21.transpose();
        delta.translation_vector = -(R_21.transpose() * t_21);
        return prev_pose.compose(delta);
    }

    // Converts PnP output (rvec, tvec in camera-from-world) to stored T_wc.
    // Returns an absolute pose -- do NOT compose with prev_pose.
    static Pose from_world_to_camera_cv(const cv::Mat& rvec, const cv::Mat& tvec) {
        cv::Mat R_cv;
        cv::Rodrigues(rvec, R_cv);

        cv::Mat R64, t64;
        R_cv.convertTo(R64, CV_64F);
        tvec.convertTo(t64, CV_64F);

        Eigen::Matrix3d R_cw;
        Eigen::Vector3d t_cw;
        cv::cv2eigen(R64, R_cw);
        cv::cv2eigen(t64, t_cw);

        const Eigen::Matrix3d R_wc = R_cw.transpose();
        const Eigen::Vector3d t_wc = -(R_wc * t_cw);
        return Pose(R_wc, t_wc);
    }
};
