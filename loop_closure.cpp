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

// Per-frame correction data saved before poses are modified so that
// map-point world positions can be updated consistently afterward.
struct FrameCorrection {
    Eigen::Vector3d old_t_wc;  // camera centre (t_wc) before correction
    Eigen::Matrix3d R_corr;    // rotation correction applied as left-perturbation
    Eigen::Vector3d delta_t;   // translation applied to t_wc (= -alpha * drift_t)
};

// Pose-graph correction
//
// Linearly interpolates the positional and rotational drift over all frames
// from old_frame_idx to curr_frame_idx (inclusive).  Map points whose first
// observation is inside the correction window are geometrically repositioned
// so that the map geometry visibly changes together with the trajectory.
//
// Returns { map_points_updated, avg_displacement, max_displacement }.
static std::tuple<int, double, double> apply_pose_graph_correction(
    SlamMap&        map,
    int             old_frame_idx,
    int             curr_frame_idx)
{
    if (old_frame_idx < 0 || curr_frame_idx <= old_frame_idx)
        return {0, 0.0, 0.0};

    const Pose& P_old  = map.frame_at(old_frame_idx).pose;
    const Pose& P_curr = map.frame_at(curr_frame_idx).pose;

    // Positional drift: current says it's at P_curr.t but should be near P_old.t
    const Eigen::Vector3d drift_t = P_curr.translation_vector - P_old.translation_vector;

    // Rotational drift: R_drift * P_old.R = P_curr.R  →  R_drift = P_curr.R * P_old.R^T
    const Eigen::Matrix3d R_drift = P_curr.rotation_matrix * P_old.rotation_matrix.transpose();
    const Eigen::AngleAxisd aa_drift(R_drift);
    const double            drift_angle = aa_drift.angle();
    const Eigen::Vector3d   drift_axis  = aa_drift.axis();

    const int span = curr_frame_idx - old_frame_idx;

    // Step 1 — Save old camera centres and compute per-frame corrections BEFORE
    //          modifying any pose.  We need the old t_wc to correctly transform
    //          world-space map points later.
    std::vector<FrameCorrection> corrections(curr_frame_idx + 1);
    for (int idx = old_frame_idx; idx <= curr_frame_idx; ++idx) {
        const Frame& f = map.frame_at(idx);
        const double alpha = static_cast<double>(idx - old_frame_idx) / span;

        Eigen::Matrix3d R_corr = Eigen::Matrix3d::Identity();
        if (drift_angle > 1e-9) {
            Eigen::AngleAxisd aa(-alpha * drift_angle, drift_axis);
            R_corr = aa.toRotationMatrix();
        }

        corrections[idx].old_t_wc = f.pose.translation_vector;
        corrections[idx].R_corr   = R_corr;
        corrections[idx].delta_t  = -(alpha * drift_t);  // camera centre moves by this
    }

    // Step 2 — Apply pose corrections to all frames in the window.
    for (int idx = old_frame_idx; idx <= curr_frame_idx; ++idx) {
        Frame& f = map.frame_at(idx);
        f.pose.translation_vector += corrections[idx].delta_t;
        f.pose.rotation_matrix = corrections[idx].R_corr * f.pose.rotation_matrix;
    }

    // Step 3 — Reposition map-point world coordinates.
    //
    // For each map point whose ANCHOR frame (first observation) is inside the
    // correction window, apply the same SE(3) delta that was applied to the
    // anchor camera.  This keeps the bearing ray from the anchor frame
    // invariant in the corrected world frame:
    //
    //   p_cam = R_cw_old * (p_world_old - t_wc_old)
    //   p_world_new = R_corr * (p_world_old - t_wc_old) + t_wc_new
    //               = R_corr * (p_world_old - t_wc_old) + t_wc_old + delta_t
    //
    // All quantities are in already-established map scale — no unit-translation
    // artefact is introduced.

    const int win_id_lo = map.frame_at(old_frame_idx).id;
    const int win_id_hi = map.frame_at(curr_frame_idx).id;

    int    pts_updated = 0;
    double sum_disp    = 0.0;
    double max_disp    = 0.0;

    for (MapPoint& mp : map.map_points()) {
        if (!mp.valid || mp.observations.empty()) continue;

        // Use first observation as the anchor frame.
        const int anchor_id = mp.observations[0].frame_id;
        if (anchor_id < win_id_lo || anchor_id > win_id_hi) continue;

        // In this project frame_id == frame_idx (frames assigned sequentially from 0).
        const int anchor_idx = anchor_id;
        const FrameCorrection& fc = corrections[anchor_idx];

        const Eigen::Vector3d p_old = mp.position;
        mp.position = fc.R_corr * (p_old - fc.old_t_wc) + fc.old_t_wc + fc.delta_t;

        // Sanity checks
        if (!mp.position.allFinite()) {
            mp.valid = false;
            continue;
        }
        const Frame& anchor_frame = map.frame_at(anchor_idx);
        if (anchor_frame.pose.transform_to_camera(mp.position).z() < 0.0) {
            mp.valid = false;
            continue;
        }

        const double disp = (mp.position - p_old).norm();
        sum_disp += disp;
        max_disp  = std::max(max_disp, disp);
        ++pts_updated;
    }

    const double avg_disp = (pts_updated > 0) ? (sum_disp / pts_updated) : 0.0;
    return {pts_updated, avg_disp, max_disp};
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
    auto [pts_updated, avg_disp, max_disp] =
        apply_pose_graph_correction(map,
                                    result.matched_frame_idx,
                                    current_frame_idx);
    result.correction_applied           = true;
    result.corrected_frames             = current_frame_idx - result.matched_frame_idx + 1;
    result.map_points_updated           = pts_updated;
    result.avg_map_point_displacement   = avg_disp;
    result.max_map_point_displacement   = max_disp;

    std::cout << "[loop] pose-graph correction applied over "
              << result.corrected_frames << " frames"
              << " | map pts updated: " << pts_updated
              << " | avg disp: " << avg_disp
              << " | max disp: " << max_disp << "\n";

    return result;
}
