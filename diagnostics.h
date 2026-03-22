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
    int    epipolar_inliers = 0;
    double inlier_ratio     = -1.0;
    double epi_error        = -1.0;  // mean epipolar error (pixels)

    // Triangulation
    int triangulated_points = 0;

    // Periodic PnP relocalization
    int    pnp_correspondences = 0;
    int    pnp_inliers         = 0;
    double reproj_error        = -1.0;
    bool   pnp_periodic_ran    = false;
    // "world" = object points in world frame (correct, absolute pose result)
    // "none"  = PnP not run
    std::string pnp_point_frame = "none";

    // Pose-only LM optimization
    bool   optimizer_ran       = false;
    double reproj_before_optim = -1.0;
    double reproj_after_optim  = -1.0;

    // Loop closure
    bool loop_candidate_found    = false;
    bool loop_verified           = false;
    bool loop_correction_applied = false;
    int  loop_matched_frame_id   = -1;
    int  loop_corrected_frames           = 0;
    int  loop_map_points_updated         = 0;
    double loop_avg_map_point_displacement = -1.0;
    double loop_max_map_point_displacement = -1.0;

    // Keyframe
    bool is_keyframe  = false;
    int  num_keyframes = 0;

    // Step motion
    double step_t_norm = -1.0;
    double step_r_deg  = -1.0;

    // Tracking state
    int  consecutive_reject_count  = 0;
    int  last_accepted_frame       = -1;
    int  frames_since_last_accept  = -1;

    bool        recovery_mode = false;
    bool        accepted      = false;
    std::string pose_source   = "none";  // "epipolar", "pnp", "none"
    std::string reason        = "unknown";
};

// Rolling statistics over recently accepted steps.
struct AcceptedMotionStats {
    double median_step_t     = 0.02;
    double median_step_r_deg = 1.0;
};

AcceptedMotionStats compute_recent_motion_stats(
    const std::deque<StepDiagnostics>& recent_accepted_steps);

bool evaluate_motion_acceptance(
    StepDiagnostics& diag,
    const AcceptedMotionStats& recent_stats);


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
