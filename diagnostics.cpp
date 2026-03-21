#include "diagnostics.h"

#include <algorithm>
#include <iostream>
#include <iomanip>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static double median_of_vector(std::vector<double> v, double fallback) {
    if (v.empty()) return fallback;
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    return (n % 2 == 1) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

// ---------------------------------------------------------------------------
// Motion statistics
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Acceptance evaluation
// ---------------------------------------------------------------------------

bool evaluate_motion_acceptance(
    StepDiagnostics& diag,
    const AcceptedMotionStats& recent_stats)
{
    const bool recovery = diag.recovery_mode;

    // Epipolar inlier count
    const int min_inliers = recovery ? 10 : 15;
    if (diag.epipolar_inliers < min_inliers) {
        diag.accepted = false;
        diag.reason   = "reject_low_inlier_count";
        return false;
    }

    // Inlier ratio
    const double min_ratio = recovery ? 0.25 : 0.35;
    if (diag.inlier_ratio >= 0.0 && diag.inlier_ratio < min_ratio) {
        diag.accepted = false;
        diag.reason   = "reject_low_inlier_ratio";
        return false;
    }

    // Epipolar error
    const double max_epi = recovery ? 3.0 : 2.5;
    if (diag.epi_error > max_epi) {
        diag.accepted = false;
        diag.reason   = "reject_high_epi_error";
        return false;
    }

    // Hard rotation limit
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
        // Cooldown (3+ consecutive rejects): even stricter
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

// ---------------------------------------------------------------------------
// CSV output
// ---------------------------------------------------------------------------

void write_csv_header(std::ofstream& f) {
    f << "prev_frame,curr_frame,accepted,reason,pose_source,recovery_mode,"
      << "raw_matches,good_matches,epipolar_inliers,inlier_ratio,epi_error,"
      << "triangulated_points,pnp_correspondences,pnp_inliers,reproj_error,"
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
    for (const auto& s : all_steps) {
        if (s.accepted) {
            ++accepted;
            if (s.recovery_mode) ++recovery_accepted;
        } else {
            ++rejected;
        }
        max_consec = std::max(max_consec, s.consecutive_reject_count);
    }

    f << std::fixed << std::setprecision(4);
    f << "frames_loaded: "          << total_frames      << "\n";
    f << "rows: "                   << all_steps.size()  << "\n";
    f << "accepted: "               << accepted          << "\n";
    f << "rejected: "               << rejected          << "\n";
    f << "recovery_accepted: "      << recovery_accepted << "\n";
    f << "max_consecutive_rejects: "<< max_consec        << "\n";
    f << "map_points: "             << total_map_points  << "\n";
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
            (s.frames_since_last_accept >= 8);
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
          << "\n";
    }
}
