#pragma once

#include "frame.h"
#include "map_point.h"

#include <vector>
#include <unordered_map>

// Owns all frames and map points.
// Provides stable integer IDs and helpers for observation-based lookups.
class SlamMap {
public:
    // Add a frame (assigns id automatically if frame.id == -1).
    int add_frame(Frame f);

    // Add a map point (assigns id automatically if mp.id == -1).
    int add_map_point(MapPoint mp);

    // Access by sequential index (not by id).
    Frame&       frame_at(int idx)       { return frames_[idx]; }
    const Frame& frame_at(int idx) const { return frames_[idx]; }

    MapPoint&       map_point_at(int idx)       { return map_points_[idx]; }
    const MapPoint& map_point_at(int idx) const { return map_points_[idx]; }

    const std::vector<Frame>&    frames()     const { return frames_; }
    const std::vector<MapPoint>& map_points() const { return map_points_; }

    std::vector<Frame>&    frames()     { return frames_; }
    std::vector<MapPoint>& map_points() { return map_points_; }

    int frame_count()     const { return static_cast<int>(frames_.size());     }
    int map_point_count() const { return static_cast<int>(map_points_.size()); }

    // ---- Keyframe tracking --------------------------------------------------

    // Mark frame at frames_[frame_idx] as a keyframe and record the index.
    void register_keyframe(int frame_idx);

    // Indices into frames_[] of all keyframes, in order of registration.
    const std::vector<int>& keyframe_indices() const { return keyframe_indices_; }
    int keyframe_count() const { return static_cast<int>(keyframe_indices_.size()); }

    // ---- Map-point helpers --------------------------------------------------

    // Build lookup: keypoint_idx in frame_id → map_point index in map_points_.
    std::unordered_map<int, int> build_kp_to_mp_index(int frame_id) const;

    // Invalidate map points with fewer than min_observations observations,
    // non-finite positions, or zero valid-depth.
    void cull_map_points(int min_observations = 2);

private:
    std::vector<Frame>    frames_;
    std::vector<MapPoint> map_points_;
    std::vector<int>      keyframe_indices_;

    int next_frame_id_ = 0;
    int next_mp_id_    = 0;
};
