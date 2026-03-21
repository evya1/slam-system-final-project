#pragma once

#include "slam_map.h"
#include "keyframe_db.h"

#include <opencv2/core.hpp>

// Result returned by try_loop_closure().
struct LoopClosureResult {
    bool candidate_found     = false;  // descriptor-similarity candidate found
    bool verified            = false;  // geometrically verified via E-matrix
    bool correction_applied  = false;  // pose-graph correction applied to map

    int  matched_frame_id    = -1;     // frame_id of the matched old keyframe
    int  matched_frame_idx   = -1;     // frame index in SlamMap of old keyframe
    int  match_score         = 0;      // descriptor match count (appearance)
    int  verification_inliers = 0;     // E-matrix inlier count
};

// Attempt to detect and close a loop for the keyframe that was just registered.
//
// current_frame_idx — index in SlamMap::frames_ of the current keyframe.
// kf_db             — the keyframe database (current keyframe must already be added).
// current_kf_db_idx — db index of the current keyframe inside kf_db.
//
// On a verified loop:
//   - All frame poses between the matched old frame and the current frame
//     are linearly corrected (positional + rotational interpolation).
//   - Affected map-point positions are also updated.
//
// K is used for geometric verification (findEssentialMat).
LoopClosureResult try_loop_closure(
    SlamMap&           map,
    const KeyframeDB&  kf_db,
    int                current_frame_idx,
    int                current_kf_db_idx,
    const cv::Mat&     K,
    int                min_temporal_gap   = 30,
    int                min_appearance_score = 40,
    int                min_verify_inliers  = 20);
