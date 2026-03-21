#pragma once

#include "frame.h"
#include "slam_map.h"
#include "diagnostics.h"
#include "feature_matcher.h"

#include <opencv2/features2d.hpp>
#include <deque>

// Monocular visual-odometry frontend.
//
// Processing pipeline per frame pair:
//   1. Feature matching (ORB + BF + Lowe)
//   2. Essential-matrix estimation → epipolar inliers
//   3. Relative pose recovery from E
//   4. Motion-bounds acceptance check
//   5. Triangulation of epipolar inliers → new MapPoints with observations
//   6. PnP relocalization using existing map-point observations (optional)
//   7. Diagnostics collection
class Frontend {
public:
    // K — 3×3 camera intrinsic matrix, kept for the lifetime of the frontend.
    explicit Frontend(cv::Mat K, int orb_features = 3000);

    // Extract ORB features into a frame (populates keypoints + descriptors).
    bool extract_features(Frame& frame) const;

    // Process a consecutive frame pair.
    // Precondition: prev_frame.processed == true.
    // On acceptance: updates curr_frame.pose and inserts new map points.
    // On rejection:  sets curr_frame.pose = prev_frame.pose (pose hold).
    StepDiagnostics process(Frame& prev_frame, Frame& curr_frame, SlamMap& map);

private:
    // Attempt PnP relocalization from existing map points observed in prev_frame.
    // Returns number of PnP inliers (0 if not run or failed).
    // On success, updates curr_frame.pose.
    int try_pnp_relocalizer(
        const Frame&    prev_frame,
        Frame&          curr_frame,
        const MatchInfo& match,
        const SlamMap&  map,
        StepDiagnostics& diag);

    cv::Ptr<cv::ORB> orb_;
    cv::Mat          K_;

    // Tracking state
    int  consecutive_rejects_  = 0;
    int  last_accepted_frame_  = -1;
    std::deque<StepDiagnostics> recent_accepted_steps_;
    static constexpr int kRecentWindowSize = 25;
    static constexpr int kRecoveryThreshold = 8;
};
