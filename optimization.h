#pragma once

#include "slam_map.h"

// ---------------------------------------------------------------------------
// Optimization / bundle adjustment scaffold
//
// This module is intentionally left as a scaffold for HW3 extension.
// Hooks are defined here so the rest of the pipeline can call them cleanly;
// the implementations below are no-ops that can be replaced without changing
// any call sites.
// ---------------------------------------------------------------------------

// Local bundle adjustment: optimise poses and map-point positions for the
// last N keyframes and all map points observed by them.
// Currently a no-op placeholder — implement with g2o / Ceres / manual GN.
void local_bundle_adjustment(SlamMap& map, int window_size = 10);

// Full bundle adjustment over the entire map.
// Currently a no-op placeholder.
void full_bundle_adjustment(SlamMap& map);
