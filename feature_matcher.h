#pragma once

#include "frame.h"

#include <opencv2/features2d.hpp>
#include <vector>

// Result of matching two frames.
struct MatchInfo {
    std::vector<cv::DMatch> raw_matches;   // all 1-NN matches
    std::vector<cv::DMatch> good_matches;  // after Lowe ratio + distance filter

    // Parallel arrays aligned with good_matches
    std::vector<cv::Point2f> pts_prev;
    std::vector<cv::Point2f> pts_curr;
};

// Extract ORB features into frame (populates keypoints + descriptors).
bool extract_features(Frame& frame, cv::Ptr<cv::ORB>& orb);

// Match features between two already-extracted frames using BF+Lowe+distance filter.
MatchInfo match_features(const Frame& prev, const Frame& curr);
