#include "slam_map.h"

int SlamMap::add_frame(Frame f) {
    if (f.id < 0) f.id = next_frame_id_;
    next_frame_id_ = std::max(next_frame_id_, f.id + 1);
    frames_.push_back(std::move(f));
    return frames_.back().id;
}

int SlamMap::add_map_point(MapPoint mp) {
    if (mp.id < 0) mp.id = next_mp_id_;
    next_mp_id_ = std::max(next_mp_id_, mp.id + 1);
    map_points_.push_back(std::move(mp));
    return map_points_.back().id;
}

void SlamMap::register_keyframe(int frame_idx) {
    if (frame_idx < 0 || frame_idx >= static_cast<int>(frames_.size())) return;
    frames_[frame_idx].is_keyframe    = true;
    frames_[frame_idx].keyframe_db_idx = static_cast<int>(keyframe_indices_.size());
    keyframe_indices_.push_back(frame_idx);
}

std::unordered_map<int, int> SlamMap::build_kp_to_mp_index(int frame_id) const {
    std::unordered_map<int, int> kp_to_mp;
    for (int mp_idx = 0; mp_idx < static_cast<int>(map_points_.size()); ++mp_idx) {
        const MapPoint& mp = map_points_[mp_idx];
        if (!mp.valid) continue;
        for (const Observation& obs : mp.observations) {
            if (obs.frame_id == frame_id) {
                kp_to_mp[obs.keypoint_idx] = mp_idx;
            }
        }
    }
    return kp_to_mp;
}

void SlamMap::cull_map_points(int min_observations) {
    for (MapPoint& mp : map_points_) {
        if (!mp.valid) continue;

        // Require minimum observation count
        if (mp.observation_count() < min_observations) {
            mp.valid = false;
            continue;
        }

        // Require finite position
        if (!mp.position.allFinite()) {
            mp.valid = false;
            continue;
        }
    }
}
