#include "feature_matcher.h"

#include <algorithm>
#include <limits>

bool extract_features(Frame& frame, cv::Ptr<cv::ORB>& orb) {
    if (frame.image_gray.empty()) return false;
    orb->detectAndCompute(frame.image_gray, cv::noArray(),
                          frame.keypoints, frame.descriptors);
    frame.processed = true;
    return !frame.keypoints.empty() && !frame.descriptors.empty();
}

MatchInfo match_features(const Frame& prev, const Frame& curr) {
    MatchInfo info;
    if (prev.descriptors.empty() || curr.descriptors.empty()) return info;

    cv::BFMatcher matcher(cv::NORM_HAMMING);
    std::vector<std::vector<cv::DMatch>> knn;
    matcher.knnMatch(prev.descriptors, curr.descriptors, knn, 2);

    for (const auto& pair : knn) {
        if (pair.size() < 2) continue;
        info.raw_matches.push_back(pair[0]);
        // Lowe ratio test
        if (pair[0].distance < 0.72f * pair[1].distance)
            info.good_matches.push_back(pair[0]);
    }

    if (info.good_matches.empty()) return info;

    // Additional distance filter: keep only matches within 2.2× the minimum
    double min_dist = std::numeric_limits<double>::max();
    for (const auto& m : info.good_matches)
        min_dist = std::min(min_dist, static_cast<double>(m.distance));

    std::vector<cv::DMatch> filtered;
    filtered.reserve(info.good_matches.size());
    for (const auto& m : info.good_matches)
        if (m.distance <= std::max(2.2 * min_dist, 28.0))
            filtered.push_back(m);
    info.good_matches = std::move(filtered);

    // Populate aligned 2D point arrays
    info.pts_prev.reserve(info.good_matches.size());
    info.pts_curr.reserve(info.good_matches.size());
    for (const auto& m : info.good_matches) {
        info.pts_prev.push_back(prev.keypoints[m.queryIdx].pt);
        info.pts_curr.push_back(curr.keypoints[m.trainIdx].pt);
    }

    return info;
}
