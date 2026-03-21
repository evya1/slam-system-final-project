#include "optimization.h"

// Scaffold — implementations intentionally empty for now.
// Replace with g2o / Ceres bundle adjustment when ready.

void local_bundle_adjustment(SlamMap& /*map*/, int /*window_size*/) {
    // TODO: implement local BA over the last window_size keyframes
}

void full_bundle_adjustment(SlamMap& /*map*/) {
    // TODO: implement full BA over all frames and map points
}
