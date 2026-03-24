#include "frontend.h"
#include "viewer.h"
#include "epipolar_geometry.h"
#include "triangulation.h"
#include "pnp_relocalizer.h"
#include "optimization.h"
#include "loop_closure.h"

#include <opencv2/calib3d.hpp>
#include <iostream>
#include <unordered_map>
#include <algorithm>

// Rotation angle in degrees via Rodrigues. Used for diagnostic logging.
static double rotation_angle_deg(const Eigen::Matrix3d& R)
{
    cv::Mat R_cv(3, 3, CV_64F);
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            R_cv.at<double>(r, c) = R(r, c);
    cv::Mat rvec;
    cv::Rodrigues(R_cv, rvec);
    return cv::norm(rvec) * (180.0 / CV_PI);
}

Frontend::Frontend(cv::Mat K, int orb_features, Viewer* viewer)
    : K_(std::move(K)), viewer_(viewer)
{
    orb_ = cv::ORB::create(
        orb_features,
        1.2f, 8, 31, 0, 2,
        cv::ORB::HARRIS_SCORE, 31, 20);
}

bool Frontend::extract_features(Frame& frame) const {
    return ::extract_features(frame, const_cast<cv::Ptr<cv::ORB>&>(orb_));
}

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
    diag.num_keyframes = map.keyframe_count();

    auto reject = [&](const std::string& reason) -> StepDiagnostics& {
        diag.accepted = false;
        diag.reason   = reason;
        curr_frame.pose = prev_frame.pose;
        ++consecutive_rejects_;
        return diag;
    };

    MatchInfo match = match_features(prev_frame, curr_frame);
    diag.raw_matches  = static_cast<int>(match.raw_matches.size());
    diag.good_matches = static_cast<int>(match.good_matches.size());

    if (viewer_) {
        viewer_->show_matches(prev_frame, curr_frame,
                              match.raw_matches, match.good_matches);
    }

    if (diag.good_matches < 12) {
        return reject("reject_low_good_match_count");
    }

    EpipolarResult epi = estimate_epipolar(match.pts_prev, match.pts_curr, K_);
    diag.epipolar_inliers = epi.num_inliers;
    diag.epi_error        = epi.mean_epi_error;
    if (diag.good_matches > 0)
        diag.inlier_ratio = static_cast<double>(epi.num_inliers) /
                            static_cast<double>(diag.good_matches);

    if (!epi.success) {
        return reject("reject_epipolar_estimation_failed");
    }

    PoseRecoveryResult pose_rec = recover_pose_from_essential(
        epi, match.pts_prev, match.pts_curr, K_);

    if (!pose_rec.success) {
        return reject("reject_pose_recovery_failed");
    }

    diag.step_r_deg  = rotation_angle_deg(pose_rec.R);
    diag.step_t_norm = pose_rec.t.norm();  // always ~1.0 for monocular

    AcceptedMotionStats recent_stats = compute_recent_motion_stats(recent_accepted_steps_);
    if (!evaluate_motion_acceptance(diag, recent_stats)) {
        curr_frame.pose = prev_frame.pose;
        ++consecutive_rejects_;
        return diag;
    }

    curr_frame.pose  = Pose::from_relative(prev_frame.pose, pose_rec.R, pose_rec.t);
    diag.pose_source = "epipolar";

    std::vector<cv::Point2f> tri_pts_prev, tri_pts_curr;
    std::vector<int>         tri_match_indices;

    for (int i = 0; i < diag.good_matches; ++i) {
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

            if (dm.queryIdx < prev_frame.descriptors.rows) {
                mp.descriptor = prev_frame.descriptors.row(dm.queryIdx).clone();
            }

            map.add_map_point(std::move(mp));
        }
        diag.triangulated_points = tri.num_valid;
    }

    ++accepted_since_kf_;
    ++accepted_total_;

    const Frame* last_kf = (last_kf_frame_idx_ >= 0)
                           ? &map.frame_at(last_kf_frame_idx_)
                           : nullptr;

    int curr_frame_idx = -1;
    for (int i = map.frame_count() - 1; i >= 0; --i) {
        if (map.frame_at(i).id == curr_frame.id) { curr_frame_idx = i; break; }
    }

    if (last_kf == nullptr || should_be_keyframe(*last_kf, curr_frame, diag)) {
        if (curr_frame_idx >= 0) {
            register_keyframe(curr_frame, curr_frame_idx, map, diag);
            diag.is_keyframe = true;
        }
    }

    if (accepted_total_ % kPnpPeriod == 0 && map.map_point_count() >= 20) {
        try_periodic_pnp(curr_frame, map, diag);
    }

    if (viewer_) {
        std::vector<cv::DMatch> inlier_dmatches;
        for (int i = 0; i < diag.good_matches; ++i) {
            if (!pose_rec.inlier_mask.empty() && !pose_rec.inlier_mask[i]) continue;
            inlier_dmatches.push_back(match.good_matches[i]);
        }
        viewer_->show_matches(prev_frame, curr_frame,
                              match.raw_matches, inlier_dmatches);
    }

    diag.accepted = true;
    if (diag.reason == "unknown") diag.reason = "accepted";

    consecutive_rejects_ = 0;
    last_accepted_frame_ = curr_frame.id;
    recent_accepted_steps_.push_back(diag);
    if (static_cast<int>(recent_accepted_steps_.size()) > kRecentWindowSize)
        recent_accepted_steps_.pop_front();

    return diag;
}

bool Frontend::should_be_keyframe(
    const Frame& last_kf, const Frame& curr_frame,
    const StepDiagnostics& diag) const
{
    if (accepted_since_kf_ >= kMaxAcceptedPerKf) return true;
    if (diag.step_r_deg >= kKfMinRotDeg) return true;
    if (diag.triangulated_points > 80) return true;

    // Cumulative rotation from last keyframe
    const Eigen::Matrix3d dR =
        curr_frame.pose.rotation_matrix *
        last_kf.pose.rotation_matrix.transpose();
    if (rotation_angle_deg(dR) >= 5.0) return true;

    return false;
}

void Frontend::register_keyframe(
    Frame& curr_frame, int frame_idx,
    SlamMap& map, StepDiagnostics& diag)
{
    map.register_keyframe(frame_idx);
    curr_frame.is_keyframe = true;

    int kf_db_idx = kf_db_.add_keyframe(
        frame_idx, curr_frame.id, curr_frame.descriptors);
    curr_frame.keyframe_db_idx = kf_db_idx;

    last_kf_frame_idx_ = frame_idx;
    accepted_since_kf_ = 0;
    diag.num_keyframes = map.keyframe_count();

    if (kf_db_.count() >= kLoopMinKeyframes) {
        LoopClosureResult lc = try_loop_closure(
            map, kf_db_,
            frame_idx, kf_db_idx,
            K_);

        diag.loop_candidate_found    = lc.candidate_found;
        diag.loop_verified           = lc.verified;
        diag.loop_correction_applied = lc.correction_applied;
        diag.loop_matched_frame_id   = lc.matched_frame_id;
        diag.loop_corrected_frames           = lc.corrected_frames;
        diag.loop_map_points_updated         = lc.map_points_updated;
        diag.loop_avg_map_point_displacement = lc.avg_map_point_displacement;
        diag.loop_max_map_point_displacement = lc.max_map_point_displacement;

        if (lc.correction_applied) {
            map.cull_map_points(2);
        }
    }
}

void Frontend::try_periodic_pnp(
    Frame& curr_frame, SlamMap& map, StepDiagnostics& diag)
{
    std::vector<int>           mp_indices;
    cv::Mat                    mp_descriptors;

    for (int i = 0; i < map.map_point_count(); ++i) {
        const MapPoint& mp = map.map_point_at(i);
        if (!mp.valid || mp.descriptor.empty()) continue;
        mp_indices.push_back(i);
        mp_descriptors.push_back(mp.descriptor);
    }

    if (mp_indices.empty() || curr_frame.descriptors.empty()) return;

    cv::BFMatcher bf(cv::NORM_HAMMING, false);
    std::vector<std::vector<cv::DMatch>> matches;
    try {
        bf.knnMatch(mp_descriptors, curr_frame.descriptors, matches, 2);
    } catch (...) {
        return;
    }

    std::vector<cv::Point3f> pts3d;
    std::vector<cv::Point2f> pts2d;

    for (const auto& pair : matches) {
        if (pair.size() < 2) continue;
        if (pair[0].distance >= 0.75f * pair[1].distance) continue;

        int mp_idx = mp_indices[pair[0].queryIdx];
        int kp_idx = pair[0].trainIdx;

        const Eigen::Vector3d& p = map.map_point_at(mp_idx).position;
        if (!p.allFinite()) continue;

        pts3d.push_back({static_cast<float>(p.x()),
                         static_cast<float>(p.y()),
                         static_cast<float>(p.z())});
        pts2d.push_back(curr_frame.keypoints[kp_idx].pt);
    }

    diag.pnp_correspondences = static_cast<int>(pts3d.size());
    diag.pnp_point_frame     = "world";
    diag.pnp_periodic_ran    = true;

    if (diag.pnp_correspondences < 12) return;

    // Object points are in world frame
    PnPResult pnp = solve_pnp(pts3d, pts2d, K_,
                               150, 3.0f, 0.995, 10);
    if (!pnp.success) return;

    diag.pnp_inliers  = pnp.num_inliers;
    diag.reproj_error = pnp.mean_reproj_error;

    if (pnp.num_inliers < 15 || pnp.mean_reproj_error > 3.0) return;

    Pose pnp_pose = Pose::from_world_to_camera_cv(pnp.rvec, pnp.tvec);

    std::vector<cv::Point3f> inlier_pts3d;
    std::vector<cv::Point2f> inlier_pts2d;
    inlier_pts3d.reserve(pnp.num_inliers);
    inlier_pts2d.reserve(pnp.num_inliers);
    for (int idx : pnp.inlier_indices) {
        inlier_pts3d.push_back(pts3d[idx]);
        inlier_pts2d.push_back(pts2d[idx]);
    }

    OptimResult optim = optimize_pose_lm(pnp_pose, inlier_pts3d, inlier_pts2d, K_);

    diag.optimizer_ran       = true;
    diag.reproj_before_optim = optim.reproj_before;
    diag.reproj_after_optim  = optim.reproj_after;

    const bool pose_ok = optim.reproj_after >= 0.0 &&
                         optim.reproj_after < 3.5;

    if (pose_ok) {
        curr_frame.pose  = optim.optimized_pose;
        diag.pose_source = "pnp";
    }
}
