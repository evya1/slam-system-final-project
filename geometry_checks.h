#pragma once

// Lightweight, deterministic sanity checks for the core geometry layer.
// Call run_geometry_self_checks() once at startup in debug builds.
// All checks use synthetic data — no external dependencies.

#include "pose.h"

#include <Eigen/Dense>
#include <opencv2/calib3d.hpp>
#include <cassert>
#include <cmath>
#include <iostream>

// Helper: check that two values are within tolerance
static inline void check_near(double a, double b, double tol,
                               const char* label) {
    if (std::fabs(a - b) > tol) {
        std::cerr << "[geometry_check FAIL] " << label
                  << "  expected ~" << b << "  got " << a
                  << "  (diff=" << std::fabs(a - b) << ")\n";
        assert(false && "geometry self-check failed");
    }
}

// Check 1: T_wc / T_cw round-trip
//
// Given any pose T_wc, verify:
//   R_cw = R_wc^T
//   t_cw = -R_wc^T * t_wc
//   p_cam == R_cw * p_world + t_cw  (same as transform_to_camera)
static void check_twc_tcw_round_trip() {
    // Construct a non-trivial synthetic pose
    Eigen::Matrix3d R_wc = Eigen::AngleAxisd(0.3, Eigen::Vector3d(0, 1, 0).normalized())
                               .toRotationMatrix();
    Eigen::Vector3d t_wc(1.0, 2.0, -0.5);
    Pose pose(R_wc, t_wc);

    Eigen::Vector3d p_world(3.0, -1.0, 4.0);

    // Via Pose helpers
    Eigen::Vector3d p_cam_via_helper = pose.transform_to_camera(p_world);

    // Via explicit R_cw / t_cw
    Eigen::Matrix3d R_cw = pose.R_cw();
    Eigen::Vector3d t_cw = pose.t_cw();
    Eigen::Vector3d p_cam_via_explicit = R_cw * p_world + t_cw;

    for (int k = 0; k < 3; ++k) {
        check_near(p_cam_via_explicit(k), p_cam_via_helper(k), 1e-10,
                   "T_wc/T_cw round-trip");
    }

    // Inverse should recover the world point
    Eigen::Vector3d p_world_recovered = pose.transform_to_world(p_cam_via_helper);
    for (int k = 0; k < 3; ++k) {
        check_near(p_world_recovered(k), p_world(k), 1e-10,
                   "T_wc/T_cw inverse");
    }
}

// Check 2: projection_matrix(K) consistency
//
// Project a 3D point through P = K[R_cw|t_cw] and verify it matches
// a manual projection via transform_to_camera + K application.
static void check_projection_matrix() {
    Eigen::Matrix3d R_wc = Eigen::AngleAxisd(0.5, Eigen::Vector3d(0, 0, 1).normalized())
                               .toRotationMatrix();
    Eigen::Vector3d t_wc(0.5, 0.0, 2.0);
    Pose pose(R_wc, t_wc);

    cv::Mat K = (cv::Mat_<double>(3, 3) <<
        500.0, 0.0, 320.0,
        0.0, 500.0, 240.0,
        0.0, 0.0,   1.0);

    cv::Mat P = pose.projection_matrix(K);

    Eigen::Vector3d p_world(1.0, 0.5, 5.0);

    // Project via P
    cv::Mat pw = (cv::Mat_<double>(4, 1) <<
        p_world.x(), p_world.y(), p_world.z(), 1.0);
    cv::Mat proj = P * pw;
    double u_via_P = proj.at<double>(0) / proj.at<double>(2);
    double v_via_P = proj.at<double>(1) / proj.at<double>(2);

    // Manual projection
    Eigen::Vector3d p_cam = pose.transform_to_camera(p_world);
    double u_manual = 500.0 * p_cam.x() / p_cam.z() + 320.0;
    double v_manual = 500.0 * p_cam.y() / p_cam.z() + 240.0;

    check_near(u_via_P, u_manual, 1e-8, "projection_matrix u");
    check_near(v_via_P, v_manual, 1e-8, "projection_matrix v");
}

// Check 3: from_world_to_camera_cv (PnP conversion)
//
// Construct a T_cw, convert to (rvec, tvec) via Rodrigues, then recover
// the pose via from_world_to_camera_cv and verify R_wc / t_wc match.
static void check_from_world_to_camera_cv() {
    // Build a known T_cw
    Eigen::Matrix3d R_cw = Eigen::AngleAxisd(0.4, Eigen::Vector3d(1, 0, 0).normalized())
                               .toRotationMatrix();
    Eigen::Vector3d t_cw(0.2, -0.3, 0.8);

    // The expected T_wc
    Eigen::Matrix3d R_wc_expected = R_cw.transpose();
    Eigen::Vector3d t_wc_expected = -(R_wc_expected * t_cw);

    // Convert to OpenCV rvec/tvec
    cv::Mat R_cv(3, 3, CV_64F);
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            R_cv.at<double>(r, c) = R_cw(r, c);
    cv::Mat rvec;
    cv::Rodrigues(R_cv, rvec);
    cv::Mat tvec = (cv::Mat_<double>(3, 1) << t_cw.x(), t_cw.y(), t_cw.z());

    Pose recovered = Pose::from_world_to_camera_cv(rvec, tvec);

    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            check_near(recovered.rotation_matrix(r, c),
                       R_wc_expected(r, c), 1e-9,
                       "from_world_to_camera_cv R");
        }
        check_near(recovered.translation_vector(r),
                   t_wc_expected(r), 1e-9,
                   "from_world_to_camera_cv t");
    }
}

// Check 4: from_relative for epipolar motion
//
// Construct a relative R_21, t_21, build the absolute pose for frame 2,
// verify that a point transforms correctly through both.
static void check_from_relative() {
    Pose pose1(Eigen::Matrix3d::Identity(), Eigen::Vector3d(1.0, 0.0, 0.0));

    Eigen::Matrix3d R_21 = Eigen::AngleAxisd(0.1, Eigen::Vector3d(0, 1, 0).normalized())
                               .toRotationMatrix();
    Eigen::Vector3d t_21(0.0, 0.0, 0.5);   // unit translation

    Pose pose2 = Pose::from_relative(pose1, R_21, t_21);

    // A point in frame1 should transform to frame2 via R_21 / t_21
    Eigen::Vector3d p_cam1(1.0, 0.0, 3.0);
    Eigen::Vector3d p_cam2_direct = R_21 * p_cam1 + t_21;

    // Via world
    Eigen::Vector3d p_world   = pose1.transform_to_world(p_cam1);
    Eigen::Vector3d p_cam2_via_world = pose2.transform_to_camera(p_world);

    for (int k = 0; k < 3; ++k) {
        check_near(p_cam2_via_world(k), p_cam2_direct(k), 1e-9,
                   "from_relative compose");
    }
}

// Run all checks
inline void run_geometry_self_checks() {
    check_twc_tcw_round_trip();
    check_projection_matrix();
    check_from_world_to_camera_cv();
    check_from_relative();
    std::cout << "[geometry_checks] all self-checks passed\n";
}
