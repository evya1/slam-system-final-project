#pragma once

#include "pose.h"

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

#include <string>
#include <vector>

// Monocular RGB frame.
struct Frame {
    int    id        = -1;
    double timestamp = 0.0;
    std::string rgb_path;

    cv::Mat image_gray;  // for feature detection
    cv::Mat image_bgr;   // for visualization

    std::vector<cv::KeyPoint> keypoints;
    cv::Mat                   descriptors;

    Pose pose;
    bool processed   = false;

    // Keyframe flag.  Set by the frontend when this frame is selected as a
    // keyframe (sufficient baseline / motion since the previous keyframe).
    bool is_keyframe    = false;

    // Index of this frame's entry in KeyframeDB (-1 if not a keyframe).
    int  keyframe_db_idx = -1;
};
