#pragma once

#include "frame.h"
#include "slam_map.h"
#include "diagnostics.h"
#include "feature_matcher.h"
#include "keyframe_db.h"

#include <opencv2/features2d.hpp>
#include <deque>

// Forward declaration — include viewer.h only in frontend.cpp
class Viewer;

// Monocular visual-odometry frontend.
//
// Per-frame pipeline: feature matching -> essential matrix -> pose recovery ->
// motion acceptance -> triangulation -> keyframe selection -> periodic PnP ->
// loop closure attempt -> diagnostics.
class Frontend {
public:
    // K            — 3×3 camera intrinsic matrix.
    // viewer       — optional pointer for live match visualisation (may be nullptr).
    explicit Frontend(cv::Mat K, int orb_features = 3000, Viewer* viewer = nullptr);

    // Extract ORB features into a frame (populates keypoints + descriptors).
    bool extract_features(Frame& frame) const;

    // Process a consecutive frame pair.
    // Precondition: prev_frame.processed == true.
    // On acceptance: updates curr_frame.pose and inserts new map points.
    // On rejection:  sets curr_frame.pose = prev_frame.pose (pose hold).
    StepDiagnostics process(Frame& prev_frame, Frame& curr_frame, SlamMap& map);

private:
    // Attempt periodic PnP relocalization against the global map.
    // Uses world-frame map points → calls Pose::from_world_to_camera_cv.
    // Runs the LM optimizer on the PnP result.
    // On success, may update curr_frame.pose.
    void try_periodic_pnp(Frame& curr_frame, SlamMap& map, StepDiagnostics& diag);

    // Decide whether curr_frame should be a keyframe.
    bool should_be_keyframe(const Frame& prev_keyframe, const Frame& curr_frame,
                             const StepDiagnostics& diag) const;

    // Register curr_frame as a keyframe and (optionally) attempt loop closure.
    void register_keyframe(Frame& curr_frame, int frame_idx,
                            SlamMap& map, StepDiagnostics& diag);

    cv::Ptr<cv::ORB> orb_;
    cv::Mat          K_;
    Viewer*          viewer_ = nullptr;

    KeyframeDB kf_db_;
    int  last_kf_frame_idx_  = -1;  // SlamMap index of the most recent keyframe
    int  accepted_since_kf_  = 0;   // accepted frames since last keyframe

    // Tracking state
    int  accepted_total_       = 0;
    int  consecutive_rejects_  = 0;
    int  last_accepted_frame_  = -1;
    std::deque<StepDiagnostics> recent_accepted_steps_;

    // How often to run periodic PnP (in accepted frames between runs)
    static constexpr int kPnpPeriod         = 10;
    // Minimum keyframes before looking for loop candidates
    static constexpr int kLoopMinKeyframes  = 5;
    // Max accepted frames between keyframes (forces keyframe creation)
    static constexpr int kMaxAcceptedPerKf  = 5;
    // Minimum rotation change for a new keyframe (degrees)
    static constexpr double kKfMinRotDeg    = 3.0;
    // Recovery mode threshold
    static constexpr int kRecentWindowSize  = 25;
    static constexpr int kRecoveryThreshold = 8;
};
