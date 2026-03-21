#pragma once

#include <opencv2/core.hpp>
#include <vector>

// One entry in the keyframe database.
struct KeyframeEntry {
    int     frame_idx  = -1;   // index into SlamMap::frames_
    int     frame_id   = -1;   // frame.id (same as frame_idx here)
    cv::Mat descriptors;       // ORB descriptors (Nx32 CV_8U)
};

// Stores representative keyframes and supports loop-candidate retrieval by
// descriptor-match-count similarity (BoW-like score without a vocabulary).
class KeyframeDB {
public:
    // Add a new keyframe.
    // Returns the DB index of the new entry.
    int add_keyframe(int frame_idx, int frame_id, const cv::Mat& descriptors);

    int count() const { return static_cast<int>(keyframes_.size()); }

    const KeyframeEntry& get(int db_idx) const { return keyframes_[db_idx]; }

    // Find the best loop-closure candidate for the given query descriptors.
    //
    // Skips entries whose frame_id is within min_temporal_gap of current_frame_id
    // (to avoid matching temporally adjacent frames).
    //
    // Returns the DB index of the best matching entry, or -1 if no entry
    // reaches min_score matches.
    int find_loop_candidate(
        const cv::Mat& query_descriptors,
        int current_frame_id,
        int min_temporal_gap = 30,
        int min_score        = 40) const;

private:
    std::vector<KeyframeEntry> keyframes_;
};
