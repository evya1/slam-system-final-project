#pragma once

#include <string>
#include <vector>
#include <deque>
#include <fstream>

// Per-step diagnostic data collected during the monocular tracking loop.
struct StepDiagnostics {
    int prev_frame_id = -1;
    int curr_frame_id = -1;

    // Feature matching
    int raw_matches  = 0;
    int good_matches = 0;

    // Epipolar geometry (primary backbone)
    int    epipolar_inliers = 0;   // inliers to E-matrix RANSAC
    double inlier_ratio     = -1.0; // epipolar_inliers / good_matches
    double epi_error        = -1.0; // mean epipolar error (pixels)

    // Triangulation
    int triangulated_points = 0;

    // PnP relocalization (optional, from existing map)
    int    pnp_correspondences = 0;  // 3D-2D pairs available
    int    pnp_inliers         = 0;
    double reproj_error        = -1.0;

    // Step motion (rotation from recoverPose, translation norm always 1 in mono)
    double step_t_norm = -1.0;
    double step_r_deg  = -1.0;

    // Tracking state
    int  consecutive_reject_count  = 0;
    int  last_accepted_frame       = -1;
    int  frames_since_last_accept  = -1;

    bool        recovery_mode = false;
    bool        accepted      = false;
    std::string pose_source   = "none";   // "epipolar", "pnp", "none"
    std::string reason        = "unknown";
};

// Rolling statistics over recently accepted steps (used for motion outlier detection).
struct AcceptedMotionStats {
    double median_step_t     = 0.02;
    double median_step_r_deg = 1.0;
};

// Compute median-based motion statistics from the recent accepted window.
AcceptedMotionStats compute_recent_motion_stats(
    const std::deque<StepDiagnostics>& recent_accepted_steps);

// Evaluate whether the step should be accepted based on motion bounds.
// Populates the acceptance-related fields in diag and returns accepted flag.
bool evaluate_motion_acceptance(
    StepDiagnostics& diag,
    const AcceptedMotionStats& recent_stats);

// --- CSV / log output ---

void write_csv_header(std::ofstream& f);
void write_csv_row(std::ofstream& f, const StepDiagnostics& d);

void write_summary_file(
    const std::string& path,
    const std::vector<StepDiagnostics>& all_steps,
    int total_frames,
    int total_map_points);

void write_jump_candidates_file(
    const std::string& path,
    const std::vector<StepDiagnostics>& all_steps);
