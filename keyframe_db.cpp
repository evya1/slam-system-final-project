#include "keyframe_db.h"

#include <opencv2/features2d.hpp>

int KeyframeDB::add_keyframe(int frame_idx, int frame_id, const cv::Mat& descriptors) {
    KeyframeEntry entry;
    entry.frame_idx  = frame_idx;
    entry.frame_id   = frame_id;
    entry.descriptors = descriptors.clone();
    keyframes_.push_back(std::move(entry));
    return static_cast<int>(keyframes_.size()) - 1;
}

int KeyframeDB::find_loop_candidate(
    const cv::Mat& query_descriptors,
    int current_frame_id,
    int min_temporal_gap,
    int min_score) const
{
    if (keyframes_.empty() || query_descriptors.empty()) return -1;

    cv::BFMatcher bf(cv::NORM_HAMMING, /*crossCheck=*/false);

    int best_idx   = -1;
    int best_score = min_score - 1;

    for (int i = 0; i < static_cast<int>(keyframes_.size()); ++i) {
        const KeyframeEntry& kf = keyframes_[i];

        // Skip temporally close frames
        if (std::abs(current_frame_id - kf.frame_id) < min_temporal_gap) continue;

        if (kf.descriptors.empty()) continue;

        // kNN match (k=2) for Lowe ratio test
        std::vector<std::vector<cv::DMatch>> matches;
        try {
            bf.knnMatch(query_descriptors, kf.descriptors, matches, 2);
        } catch (...) {
            continue;
        }

        int score = 0;
        for (const auto& pair : matches) {
            if (pair.size() < 2) continue;
            if (pair[0].distance < 0.75f * pair[1].distance)
                ++score;
        }

        if (score > best_score) {
            best_score = score;
            best_idx   = i;
        }
    }

    return best_idx;
}
