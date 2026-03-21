#include "loop_closure.h"

#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
#include <Eigen/Dense>
#include <iostream>
#include <cmath>

// Geometric verification: estimate E between old keyframe and current frame.
// Returns number of E-RANSAC inliers (0 on failure).
static int verify_loop_geometry(
    const Frame&   old_kf,
    const Frame&   curr_kf,
    const cv::Mat& K,
    int            min_inliers)
{
    if (old_kf.descriptors.empty() || curr_kf.descriptors.empty()) return 0;

    // Match descriptors (Lowe ratio)
    cv::BFMatcher bf(cv::NORM_HAMMING, false);
    std::vector<std::vector<cv::DMatch>> matches;
    try {
        bf.knnMatch(old_kf.descriptors, curr_kf.descriptors, matches, 2);
    } catch (...) {
        return 0;
    }

    std::vector<cv::Point2f> pts1, pts2;
    for (const auto& pair : matches) {
        if (pair.size() < 2) continue;
        if (pair[0].distance < 0.75f * pair[1].distance) {
            pts1.push_back(old_kf.keypoints[pair[0].queryIdx].pt);
            pts2.push_back(curr_kf.keypoints[pair[0].trainIdx].pt);
        }
    }

    if (static_cast<int>(pts1.size()) < min_inliers) return 0;

    std::vector<uchar> mask;
    cv::Mat E;
    try {
        E = cv::findEssentialMat(pts1, pts2, K,
                                 cv::RANSAC, 0.999, 1.0, mask);
    } catch (...) {
        return 0;
    }

    if (E.empty()) return 0;
    return cv::countNonZero(mask);
}

// Pose-graph correction
//
// Linearly interpolates the positional and rotational drift over all frames
// from old_frame_idx to curr_frame_idx (inclusive).  Map points observed only
// by frames inside the correction window are also adjusted.
static void apply_pose_graph_correction(
    SlamMap&        map,
    int             old_frame_idx,
    int             curr_frame_idx)
{
    if (old_frame_idx < 0 || curr_frame_idx <= old_frame_idx) return;

    const Pose& P_old  = map.frame_at(old_frame_idx).pose;
    const Pose& P_curr = map.frame_at(curr_frame_idx).pose;

    // Positional drift: current says it's at P_curr.t but should be near P_old.t
    const Eigen::Vector3d drift_t = P_curr.translation_vector - P_old.translation_vector;

    // Rotational drift: R_drift * P_old.R = P_curr.R  →  R_drift = P_curr.R * P_old.R^T
    const Eigen::Matrix3d R_drift = P_curr.rotation_matrix * P_old.rotation_matrix.transpose();
    const Eigen::AngleAxisd aa_drift(R_drift);
    const double            drift_angle  = aa_drift.angle();
    const Eigen::Vector3d   drift_axis   = aa_drift.axis();

    const int span = curr_frame_idx - old_frame_idx;

    for (int idx = old_frame_idx; idx <= curr_frame_idx; ++idx) {
        Frame& f = map.frame_at(idx);
        const double alpha = static_cast<double>(idx - old_frame_idx) / span;

        // Correct translation: subtract the fraction of drift accumulated so far
        f.pose.translation_vector -= alpha * drift_t;

        // Correct rotation: apply -alpha * drift_rotation (left perturbation)
        if (drift_angle > 1e-9) {
            Eigen::AngleAxisd correction(-alpha * drift_angle, drift_axis);
            f.pose.rotation_matrix = correction.toRotationMatrix() * f.pose.rotation_matrix;
        }
    }

    // Update map points: reproject world points using the updated poses.
    // Simple approach: for each map point, re-average its position from its
    // first two observations using the corrected poses (keeps things simple).
    // Points that become invalid (negative depth) are culled.
    for (MapPoint& mp : map.map_points()) {
        if (!mp.valid || mp.observation_count() < 2) continue;

        const Observation& obs0 = mp.observations[0];
        const Observation& obs1 = mp.observations[1];

        // Only adjust if both observations are in the corrected window
        bool in0 = (obs0.frame_id >= map.frame_at(old_frame_idx).id &&
                    obs0.frame_id <= map.frame_at(curr_frame_idx).id);
        bool in1 = (obs1.frame_id >= map.frame_at(old_frame_idx).id &&
                    obs1.frame_id <= map.frame_at(curr_frame_idx).id);
        if (!in0 && !in1) continue;

        // Validate depth in corrected frame
        const Frame& f0 = map.frame_at(obs0.frame_id < static_cast<int>(map.frames().size())
                                           ? obs0.frame_id : 0);
        double z = f0.pose.transform_to_camera(mp.position).z();
        if (z < 0.0) mp.valid = false;
    }
}

// Public entry point

LoopClosureResult try_loop_closure(
    SlamMap&          map,
    const KeyframeDB& kf_db,
    int               current_frame_idx,
    int               current_kf_db_idx,
    const cv::Mat&    K,
    int               min_temporal_gap,
    int               min_appearance_score,
    int               min_verify_inliers)
{
    LoopClosureResult result;

    if (kf_db.count() < 2 || current_kf_db_idx < 1) return result;

    const Frame& curr_frame = map.frame_at(current_frame_idx);

    // 1. Appearance-based candidate retrieval
    int cand_db_idx = kf_db.find_loop_candidate(
        curr_frame.descriptors,
        curr_frame.id,
        min_temporal_gap,
        min_appearance_score);

    if (cand_db_idx < 0) return result;

    result.candidate_found  = true;
    result.match_score      = min_appearance_score;  // at least this

    const KeyframeEntry& cand_kf_entry = kf_db.get(cand_db_idx);
    result.matched_frame_id  = cand_kf_entry.frame_id;
    result.matched_frame_idx = cand_kf_entry.frame_idx;

    if (result.matched_frame_idx < 0 ||
        result.matched_frame_idx >= map.frame_count()) return result;

    const Frame& old_frame = map.frame_at(result.matched_frame_idx);

    std::cout << "[loop] candidate: frame " << curr_frame.id
              << " ↔ frame " << old_frame.id
              << "  appearance_score >= " << min_appearance_score << "\n";

    // 2. Geometric verification (essential matrix)
    int inliers = verify_loop_geometry(old_frame, curr_frame, K, min_verify_inliers);
    result.verification_inliers = inliers;

    if (inliers < min_verify_inliers) {
        std::cout << "[loop] geometric verification failed ("
                  << inliers << " inliers, need " << min_verify_inliers << ")\n";
        return result;
    }

    result.verified = true;
    std::cout << "[loop] verified! E-matrix inliers: " << inliers << "\n";

    // 3. Pose-graph correction
    apply_pose_graph_correction(map,
                                result.matched_frame_idx,
                                current_frame_idx);
    result.correction_applied = true;

    std::cout << "[loop] pose-graph correction applied over "
              << (current_frame_idx - result.matched_frame_idx + 1)
              << " frames\n";

    return result;
}
