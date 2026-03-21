#pragma once

#include <Eigen/Dense>
#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>

// =============================================================================
// Camera Pose Convention  (T_wc — world-from-camera)
// =============================================================================
// Stored fields:
//   rotation_matrix    = R_wc  (3×3, columns are camera X/Y/Z axes in world coords)
//   translation_vector = t_wc  (camera origin expressed in world coordinates)
//
// Point transforms:
//   p_world = R_wc * p_cam + t_wc               (camera → world)
//   p_cam   = R_wc^T * (p_world - t_wc)         (world  → camera)
//           = R_cw * p_world + t_cw
//   where:  R_cw = R_wc^T,    t_cw = -R_wc^T * t_wc
//
// Projection (MVG convention, §6.1):
//   P = K * [R_cw | t_cw]          (3×4 projection matrix; use projection_matrix())
//
// Relative-motion convention  (from_relative / recoverPose):
//   R_21, t_21  satisfy:   X_cam2 = R_21 * X_cam1 + t_21
//   (takes points from camera-1 frame to camera-2 frame)
//   t_21 has unit norm in monocular — scale is unknown.
//
// PnP convention  (from_world_to_camera_cv / solvePnPRansac):
//   Object points are in the WORLD frame.
//   solvePnPRansac returns (R_cw, t_cw) such that: p_cam = R_cw * p_world + t_cw
//   Convert to stored T_wc:  R_wc = R_cw^T,  t_wc = -R_wc * t_cw
//   This gives an ABSOLUTE pose — do NOT compose with prev_pose.
// =============================================================================

struct Pose {
    Eigen::Matrix3d rotation_matrix    = Eigen::Matrix3d::Identity(); // R_wc
    Eigen::Vector3d translation_vector = Eigen::Vector3d::Zero();     // t_wc

    Pose() = default;
    Pose(const Eigen::Matrix3d& R, const Eigen::Vector3d& t)
        : rotation_matrix(R), translation_vector(t) {}

    // ---- World-to-camera accessors ------------------------------------------

    // World-to-camera rotation:  R_cw = R_wc^T
    Eigen::Matrix3d R_cw() const { return rotation_matrix.transpose(); }

    // World-to-camera translation:  t_cw = -R_wc^T * t_wc
    Eigen::Vector3d t_cw() const { return -(rotation_matrix.transpose() * translation_vector); }

    // ---- Projection matrix --------------------------------------------------

    // Returns P = K * [R_cw | t_cw] as a 3×4 CV_64F matrix.
    // Use this for triangulation and reprojection.
    cv::Mat projection_matrix(const cv::Mat& K) const {
        Eigen::Matrix3d Rcw = R_cw();
        Eigen::Vector3d tcw = t_cw();

        cv::Mat Rt(3, 4, CV_64F);
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c)
                Rt.at<double>(r, c) = Rcw(r, c);
            Rt.at<double>(r, 3) = tcw(r);
        }

        cv::Mat K64;
        K.convertTo(K64, CV_64F);
        return K64 * Rt;
    }

    // ---- 4×4 homogeneous matrix ---------------------------------------------

    // Returns [R_wc | t_wc; 0 | 1]
    Eigen::Matrix4d matrix() const {
        Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
        T.block<3, 3>(0, 0) = rotation_matrix;
        T.block<3, 1>(0, 3) = translation_vector;
        return T;
    }

    // ---- SE(3) operations ---------------------------------------------------

    // Compose: T_wc_result = T_wc_this * T_wc_other
    // Interprets other as a pose expressed in this frame.
    Pose compose(const Pose& other) const {
        return {rotation_matrix * other.rotation_matrix,
                translation_vector + rotation_matrix * other.translation_vector};
    }

    // T_cw (inverse of stored T_wc)
    Pose inverse() const {
        Eigen::Matrix3d Rt = rotation_matrix.transpose();
        return {Rt, -(Rt * translation_vector)};
    }

    // ---- Point transforms ---------------------------------------------------

    Eigen::Vector3d transform_to_world(const Eigen::Vector3d& p_cam) const {
        return rotation_matrix * p_cam + translation_vector;
    }

    Eigen::Vector3d transform_to_camera(const Eigen::Vector3d& p_world) const {
        return rotation_matrix.transpose() * (p_world - translation_vector);
    }

    static Pose identity() { return {}; }

    // =========================================================================
    // Factory helpers
    // =========================================================================

    // from_relative — for consecutive-frame EPIPOLAR motion only.
    //
    // R_21, t_21 satisfy:  X_cam2 = R_21 * X_cam1 + t_21
    // (output of cv::recoverPose; |t_21| == 1, scale unknown in monocular)
    //
    // Do NOT use when object points are in world frame — use from_world_to_camera_cv.
    static Pose from_relative(const Pose&           prev_pose,
                               const Eigen::Matrix3d& R_21,
                               const Eigen::Vector3d& t_21) {
        // Invert to get camera-2-in-world: R_wc2 = R_21^T, t_wc2 = -R_21^T * t_21
        Pose delta;
        delta.rotation_matrix    =  R_21.transpose();
        delta.translation_vector = -(R_21.transpose() * t_21);
        return prev_pose.compose(delta);
    }

    // from_world_to_camera_cv — for PnP with WORLD-FRAME object points.
    //
    // solvePnPRansac returns (rvec, tvec) encoding T_cw:
    //   p_cam = R_cw * p_world + t_cw
    //
    // Converts to stored T_wc:
    //   R_wc = R_cw^T,    t_wc = -R_wc * t_cw
    //
    // Returns an ABSOLUTE pose — do NOT compose with prev_pose.
    static Pose from_world_to_camera_cv(const cv::Mat& rvec, const cv::Mat& tvec) {
        cv::Mat R_cv;
        cv::Rodrigues(rvec, R_cv);

        cv::Mat R64, t64;
        R_cv.convertTo(R64, CV_64F);
        tvec.convertTo(t64, CV_64F);

        Eigen::Matrix3d R_cw;
        Eigen::Vector3d t_cw;
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) R_cw(r, c) = R64.at<double>(r, c);
            t_cw(r) = t64.at<double>(r, 0);
        }

        // T_wc = inverse of T_cw
        Eigen::Matrix3d R_wc = R_cw.transpose();
        Eigen::Vector3d t_wc = -(R_wc * t_cw);
        return Pose(R_wc, t_wc);
    }
};
