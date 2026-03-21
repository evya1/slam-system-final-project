#include "frontend.h"
#include "epipolar_geometry.h"
#include "triangulation.h"
#include "pnp_relocalizer.h"

#include <opencv2/calib3d.hpp>
#include <iostream>
#include <unordered_map>

Frontend::Frontend(cv::Mat K, int orb_features)
    : K_(std::move(K))
{
    orb_ = cv::ORB::create(
        orb_features,
        1.2f, 8, 31, 0, 2,
        cv::ORB::HARRIS_SCORE, 31, 20);
}

bool Frontend::extract_features(Frame& frame) const {
    return ::extract_features(frame, const_cast<cv::Ptr<cv::ORB>&>(orb_));
}

// ---------------------------------------------------------------------------
// Main processing step
// ---------------------------------------------------------------------------

StepDiagnostics Frontend::process(
    Frame& prev_frame, Frame& curr_frame, SlamMap& map)
{
    StepDiagnostics diag;
    diag.prev_frame_id            = prev_frame.id;
    diag.curr_frame_id            = curr_frame.id;
    diag.consecutive_reject_count = consecutive_rejects_;
    diag.last_accepted_frame      = last_accepted_frame_;
    diag.frames_since_last_accept =
        (last_accepted_frame_ >= 0) ? (curr_frame.id - last_accepted_frame_) : -1;
    diag.recovery_mode = (consecutive_rejects_ >= kRecoveryThreshold);

    auto reject = [&](const std::string& reason) -> StepDiagnostics& {
        diag.accepted = false;
        diag.reason   = reason;
        curr_frame.pose = prev_frame.pose;
        ++consecutive_rejects_;
        return diag;
    };

    // ------------------------------------------------------------------
    // 1. Feature matching
    // ------------------------------------------------------------------
    MatchInfo match = match_features(prev_frame, curr_frame);
    diag.raw_matches  = static_cast<int>(match.raw_matches.size());
    diag.good_matches = static_cast<int>(match.good_matches.size());

    if (diag.good_matches < 12) {
        return reject("reject_low_good_match_count");
    }

    // ------------------------------------------------------------------
    // 2. Epipolar geometry  (essential matrix + inlier mask)
    // ------------------------------------------------------------------
    EpipolarResult epi = estimate_epipolar(match.pts_prev, match.pts_curr, K_);
    diag.epipolar_inliers = epi.num_inliers;
    diag.epi_error        = epi.mean_epi_error;
    if (diag.good_matches > 0)
        diag.inlier_ratio = static_cast<double>(epi.num_inliers) /
                            static_cast<double>(diag.good_matches);

    if (!epi.success) {
        return reject("reject_epipolar_estimation_failed");
    }

    // ------------------------------------------------------------------
    // 3. Pose recovery from E
    // ------------------------------------------------------------------
    PoseRecoveryResult pose_rec = recover_pose_from_essential(
        epi, match.pts_prev, match.pts_curr, K_);

    if (!pose_rec.success) {
        return reject("reject_pose_recovery_failed");
    }

    // Compute step rotation angle for diagnostics / motion check
    cv::Mat R_cv(3, 3, CV_64F);
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            R_cv.at<double>(r, c) = pose_rec.R(r, c);
    cv::Mat rvec_step;
    cv::Rodrigues(R_cv, rvec_step);
    diag.step_r_deg  = cv::norm(rvec_step) * (180.0 / CV_PI);
    diag.step_t_norm = pose_rec.t.norm();  // always ≈1.0 for monocular

    // ------------------------------------------------------------------
    // 4. Motion-bounds acceptance check
    // ------------------------------------------------------------------
    AcceptedMotionStats recent_stats = compute_recent_motion_stats(recent_accepted_steps_);
    bool ok = evaluate_motion_acceptance(diag, recent_stats);
    if (!ok) {
        // reason already set inside evaluate_motion_acceptance
        curr_frame.pose = prev_frame.pose;
        ++consecutive_rejects_;
        return diag;
    }

    // ------------------------------------------------------------------
    // 5. Accept: update pose from epipolar
    // ------------------------------------------------------------------
    curr_frame.pose  = Pose::from_relative(prev_frame.pose, pose_rec.R, pose_rec.t);
    diag.pose_source = "epipolar";

    // ------------------------------------------------------------------
    // 6. Triangulate epipolar inliers → new map points with observations
    // ------------------------------------------------------------------
    std::vector<cv::Point2f> tri_pts_prev, tri_pts_curr;
    std::vector<int>         tri_match_indices;

    for (int i = 0; i < diag.good_matches; ++i) {
        // Use only the points that passed cheirality in recoverPose
        if (!pose_rec.inlier_mask.empty() && !pose_rec.inlier_mask[i]) continue;
        tri_pts_prev.push_back(match.pts_prev[i]);
        tri_pts_curr.push_back(match.pts_curr[i]);
        tri_match_indices.push_back(i);
    }

    if (!tri_pts_prev.empty()) {
        TriangulationResult tri = triangulate_points(
            tri_pts_prev, tri_pts_curr,
            prev_frame.pose, curr_frame.pose, K_);

        for (int i = 0; i < static_cast<int>(tri.points_3d.size()); ++i) {
            if (!tri.valid[i]) continue;

            int gm_idx = tri_match_indices[i];
            const cv::DMatch& dm = match.good_matches[gm_idx];

            MapPoint mp;
            mp.position = tri.points_3d[i];
            mp.add_observation(prev_frame.id, dm.queryIdx);
            mp.add_observation(curr_frame.id, dm.trainIdx);
            map.add_map_point(std::move(mp));
        }
        diag.triangulated_points = tri.num_valid;
    }

    // ------------------------------------------------------------------
    // 7. PnP relocalization from existing map (refine / validate pose)
    // ------------------------------------------------------------------
    try_pnp_relocalizer(prev_frame, curr_frame, match, map, diag);

    // ------------------------------------------------------------------
    // 8. Finalise acceptance bookkeeping
    // ------------------------------------------------------------------
    diag.accepted = true;
    if (diag.reason == "unknown") diag.reason = "accepted";

    consecutive_rejects_ = 0;
    last_accepted_frame_ = curr_frame.id;
    recent_accepted_steps_.push_back(diag);
    if (static_cast<int>(recent_accepted_steps_.size()) > kRecentWindowSize)
        recent_accepted_steps_.pop_front();

    return diag;
}

// ---------------------------------------------------------------------------
// PnP relocalization helper
// ---------------------------------------------------------------------------

int Frontend::try_pnp_relocalizer(
    const Frame&     prev_frame,
    Frame&           curr_frame,
    const MatchInfo& match,
    const SlamMap&   map,
    StepDiagnostics& diag)
{
    // Build lookup: prev keypoint index → curr keypoint index
    std::unordered_map<int, int> prev_kp_to_curr_kp;
    prev_kp_to_curr_kp.reserve(match.good_matches.size());
    for (const auto& dm : match.good_matches)
        prev_kp_to_curr_kp[dm.queryIdx] = dm.trainIdx;

    // Find map points observed in prev_frame whose match landed in curr_frame
    std::vector<cv::Point3f> pts3d;
    std::vector<cv::Point2f> pts2d;

    for (const MapPoint& mp : map.map_points()) {
        if (!mp.valid) continue;
        for (const Observation& obs : mp.observations) {
            if (obs.frame_id != prev_frame.id) continue;
            auto it = prev_kp_to_curr_kp.find(obs.keypoint_idx);
            if (it == prev_kp_to_curr_kp.end()) continue;

            const Eigen::Vector3d& p = mp.position;
            pts3d.push_back(cv::Point3f(
                static_cast<float>(p.x()),
                static_cast<float>(p.y()),
                static_cast<float>(p.z())));
            pts2d.push_back(curr_frame.keypoints[it->second].pt);
            break;  // each map point only once
        }
    }

    diag.pnp_correspondences = static_cast<int>(pts3d.size());

    if (diag.pnp_correspondences < 10) return 0;

    PnPResult pnp = solve_pnp(pts3d, pts2d, K_);
    if (!pnp.success) return 0;

    diag.pnp_inliers  = pnp.num_inliers;
    diag.reproj_error = pnp.mean_reproj_error;

    // Use PnP pose only if it has significantly more inliers than the
    // epipolar pose gave us and the reprojection error is low.
    // Otherwise keep the epipolar pose (scale is normalised anyway).
    const bool pnp_better =
        pnp.num_inliers >= 15 &&
        pnp.mean_reproj_error >= 0.0 &&
        pnp.mean_reproj_error < 2.5;

    if (pnp_better) {
        curr_frame.pose  = Pose::from_pnp_cv(prev_frame.pose, pnp.rvec, pnp.tvec);
        diag.pose_source = "pnp";
    }

    return pnp.num_inliers;
}
