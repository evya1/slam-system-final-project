#pragma once

#include "pose.h"

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

#include <string>
#include <vector>

// Monocular RGB frame.
// Depth-related fields have been removed; this struct represents a single
// camera image with its extracted features and estimated pose.
struct Frame {
    int    id        = -1;
    double timestamp = 0.0;
    std::string rgb_path;

    cv::Mat image_gray;  // for feature detection
    cv::Mat image_bgr;   // for visualization

    std::vector<cv::KeyPoint> keypoints;
    cv::Mat                   descriptors;

    Pose pose;
    bool processed = false;
};
