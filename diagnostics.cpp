#include "diagnostics.h"

#include <algorithm>
#include <iostream>
#include <iomanip>

static double median_of_vector(std::vector<double> v, double fallback) {
    if (v.empty()) return fallback;
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    return (n % 2 == 1) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

AcceptedMotionStats compute_recent_motion_stats(
    const std::deque<StepDiagnostics>& recent)
{
    AcceptedMotionStats stats;
    std::vector<double> ts, rs;
    for (const auto& s : recent) {
        if (s.accepted && s.step_t_norm > 0.0 && std::isfinite(s.step_t_norm))
            ts.push_back(s.step_t_norm);
        if (s.accepted && s.step_r_deg > 0.0 && std::isfinite(s.step_r_deg))
            rs.push_back(s.step_r_deg);
    }
    stats.median_step_t     = std::max(median_of_vector(ts, 0.02), 0.003);
    stats.median_step_r_deg = std::max(median_of_vector(rs, 1.0),  0.15);
    return stats;
}

bool evaluate_motion_acceptance(
    StepDiagnostics& diag,
    const AcceptedMotionStats& recent_stats)
{
    const bool recovery = diag.recovery_mode;

    const int min_inliers = recovery ? 10 : 15;
    if (diag.epipolar_inliers < min_inliers) {
        diag.accepted = false;
        diag.reason   = "reject_low_inlier_count";
        return false;
    }

    const double min_ratio = recovery ? 0.25 : 0.35;
    if (diag.inlier_ratio >= 0.0 && diag.inlier_ratio < min_ratio) {
        diag.accepted = false;
        diag.reason   = "reject_low_inlier_ratio";
        return false;
    }

    const double max_epi = recovery ? 3.0 : 2.5;
    if (diag.epi_error > max_epi) {
        diag.accepted = false;
        diag.reason   = "reject_high_epi_error";
        return false;
    }

    if (diag.step_r_deg > 120.0) {
        diag.accepted = false;
        diag.reason   = "reject_near_flip_rotation";
        return false;
    }

    if (!recovery) {
        if (diag.step_r_deg > std::max(12.0, 8.0 * recent_stats.median_step_r_deg + 2.0)) {
            diag.accepted = false;
            diag.reason   = "reject_large_rotation_step";
            return false;
        }
        if (diag.consecutive_reject_count >= 3) {
            if (diag.step_r_deg > std::max(4.0, 4.0 * recent_stats.median_step_r_deg + 1.0)) {
                diag.accepted = false;
                diag.reason   = "reject_cooldown_rotation";
                return false;
            }
        }
    } else {
        if (diag.step_r_deg > std::max(3.0, 3.0 * recent_stats.median_step_r_deg + 0.8)) {
            diag.accepted = false;
            diag.reason   = "reject_recovery_rotation";
            return false;
        }
    }

    diag.accepted = true;
    diag.reason   = "accepted";
    return true;
}

void write_csv_header(std::ofstream& f) {
    f << "prev_frame,curr_frame,accepted,reason,pose_source,recovery_mode,"
      << "raw_matches,good_matches,epipolar_inliers,inlier_ratio,epi_error,"
      << "triangulated_points,"
      << "pnp_correspondences,pnp_inliers,reproj_error,pnp_periodic_ran,pnp_point_frame,"
      << "optimizer_ran,reproj_before_optim,reproj_after_optim,"
      << "loop_candidate_found,loop_verified,loop_correction_applied,loop_matched_frame_id,"
      << "loop_corrected_frames,loop_map_points_updated,"
      << "loop_avg_map_point_displacement,loop_max_map_point_displacement,"
      << "is_keyframe,num_keyframes,"
      << "step_t_norm,step_r_deg,"
      << "consecutive_reject_count,last_accepted_frame,frames_since_last_accept"
      << "\n";
}

void write_csv_row(std::ofstream& f, const StepDiagnostics& d) {
    f << d.prev_frame_id            << ","
      << d.curr_frame_id            << ","
      << (d.accepted ? 1 : 0)       << ","
      << d.reason                   << ","
      << d.pose_source              << ","
      << (d.recovery_mode ? 1 : 0)  << ","
      << d.raw_matches              << ","
      << d.good_matches             << ","
      << d.epipolar_inliers         << ","
      << d.inlier_ratio             << ","
      << d.epi_error                << ","
      << d.triangulated_points      << ","
      << d.pnp_correspondences      << ","
      << d.pnp_inliers              << ","
      << d.reproj_error             << ","
      << (d.pnp_periodic_ran ? 1 : 0) << ","
      << d.pnp_point_frame          << ","
      << (d.optimizer_ran ? 1 : 0)  << ","
      << d.reproj_before_optim      << ","
      << d.reproj_after_optim       << ","
      << (d.loop_candidate_found ? 1 : 0) << ","
      << (d.loop_verified ? 1 : 0)        << ","
      << (d.loop_correction_applied ? 1 : 0) << ","
      << d.loop_matched_frame_id    << ","
      << d.loop_corrected_frames    << ","
      << d.loop_map_points_updated  << ","
      << d.loop_avg_map_point_displacement << ","
      << d.loop_max_map_point_displacement << ","
      << (d.is_keyframe ? 1 : 0)    << ","
      << d.num_keyframes            << ","
      << d.step_t_norm              << ","
      << d.step_r_deg               << ","
      << d.consecutive_reject_count << ","
      << d.last_accepted_frame      << ","
      << d.frames_since_last_accept
      << "\n";
}

void write_summary_file(
    const std::string& path,
    const std::vector<StepDiagnostics>& all_steps,
    int total_frames,
    int total_map_points)
{
    std::ofstream f(path);
    if (!f.is_open()) {
        std::cerr << "failed to open summary file: " << path << "\n";
        return;
    }

    int accepted = 0, rejected = 0, recovery_accepted = 0, max_consec = 0;
    int optim_ran = 0, pnp_ran = 0, loop_verified = 0, loop_corrected = 0;
    int keyframes = 0;
    int total_loop_frames_corrected = 0;
    int total_loop_map_pts_updated  = 0;
    double max_loop_avg_disp = 0.0;
    double max_loop_max_disp = 0.0;

    for (const auto& s : all_steps) {
        if (s.accepted) {
            ++accepted;
            if (s.recovery_mode) ++recovery_accepted;
        } else {
            ++rejected;
        }
        max_consec = std::max(max_consec, s.consecutive_reject_count);
        if (s.optimizer_ran)            ++optim_ran;
        if (s.pnp_periodic_ran)         ++pnp_ran;
        if (s.loop_verified)            ++loop_verified;
        if (s.loop_correction_applied) {
            ++loop_corrected;
            total_loop_frames_corrected += s.loop_corrected_frames;
            total_loop_map_pts_updated  += s.loop_map_points_updated;
            if (s.loop_avg_map_point_displacement > 0.0)
                max_loop_avg_disp = std::max(max_loop_avg_disp,
                                             s.loop_avg_map_point_displacement);
            max_loop_max_disp = std::max(max_loop_max_disp,
                                         s.loop_max_map_point_displacement);
        }
        if (s.is_keyframe)              ++keyframes;
    }

    f << std::fixed << std::setprecision(4);
    f << "frames_loaded: "              << total_frames      << "\n";
    f << "rows: "                       << all_steps.size()  << "\n";
    f << "accepted: "                   << accepted          << "\n";
    f << "rejected: "                   << rejected          << "\n";
    f << "recovery_accepted: "          << recovery_accepted << "\n";
    f << "max_consecutive_rejects: "    << max_consec        << "\n";
    f << "map_points: "                 << total_map_points  << "\n";
    f << "keyframes: "                  << keyframes         << "\n";
    f << "optimizer_ran_count: "        << optim_ran         << "\n";
    f << "pnp_periodic_ran_count: "     << pnp_ran           << "\n";
    f << "loop_closures_verified: "     << loop_verified     << "\n";
    f << "loop_closures_applied: "      << loop_corrected    << "\n";
    f << "loop_total_frames_corrected: " << total_loop_frames_corrected << "\n";
    f << "loop_total_map_pts_updated: "  << total_loop_map_pts_updated  << "\n";
    f << "loop_max_avg_displacement: "   << max_loop_avg_disp           << "\n";
    f << "loop_max_max_displacement: "   << max_loop_max_disp           << "\n";
}

void write_jump_candidates_file(
    const std::string& path,
    const std::vector<StepDiagnostics>& all_steps)
{
    std::ofstream f(path);
    if (!f.is_open()) {
        std::cerr << "failed to open jump candidates file: " << path << "\n";
        return;
    }

    f << std::fixed << std::setprecision(4);
    for (const auto& s : all_steps) {
        bool suspicious =
            (!s.accepted) ||
            (s.step_r_deg > 10.0) ||
            (s.epi_error  > 2.0) ||
            (s.frames_since_last_accept >= 8) ||
            (s.loop_correction_applied);
        if (!suspicious) continue;

        f << "frame " << s.prev_frame_id << " -> " << s.curr_frame_id
          << " | accepted="     << (s.accepted ? 1 : 0)
          << " | reason="       << s.reason
          << " | pose_source="  << s.pose_source
          << " | recovery="     << (s.recovery_mode ? 1 : 0)
          << " | good="         << s.good_matches
          << " | epi_inliers="  << s.epipolar_inliers
          << " | inlier_ratio=" << s.inlier_ratio
          << " | epi_error="    << s.epi_error
          << " | reproj="       << s.reproj_error
          << " | step_r_deg="   << s.step_r_deg
          << " | step_t="       << s.step_t_norm
          << " | consec_rej="   << s.consecutive_reject_count
          << " | triangulated=" << s.triangulated_points
          << " | pnp_inliers="  << s.pnp_inliers
          << " | optim="        << (s.optimizer_ran ? 1 : 0)
          << " | reproj_before=" << s.reproj_before_optim
          << " | reproj_after="  << s.reproj_after_optim
          << " | loop_corrected=" << (s.loop_correction_applied ? 1 : 0)
          << "\n";
    }
}
