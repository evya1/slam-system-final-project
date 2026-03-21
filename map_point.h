#pragma once

#include <Eigen/Dense>
#include <opencv2/core.hpp>
#include <vector>

// A single observation of a map point in a frame.
struct Observation {
    int frame_id     = -1;  // which frame saw this point
    int keypoint_idx = -1;  // index into frame.keypoints
};

// A 3-D landmark in the world map created by triangulation.
// Stores back-references to every frame that observes it so that
// future PnP relocalization and bundle adjustment can look up
// 3D→2D correspondences without a brute-force search.
struct MapPoint {
    int              id       = -1;
    Eigen::Vector3d  position = Eigen::Vector3d::Zero();
    bool             valid    = true;

    // Representative ORB descriptor (one row, CV_8U).
    // Set from the first observing frame; used for global PnP matching.
    cv::Mat descriptor;

    std::vector<Observation> observations;

    MapPoint() = default;
    MapPoint(int id_, const Eigen::Vector3d& pos) : id(id_), position(pos) {}

    int observation_count() const {
        return static_cast<int>(observations.size());
    }

    void add_observation(int frame_id, int keypoint_idx) {
        observations.push_back({frame_id, keypoint_idx});
    }
};
