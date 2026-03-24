#pragma once

#include "slam_map.h"
#include "frame.h"

#include <pangolin/pangolin.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <string>
#include <vector>

// Wraps the Pangolin 3D viewer and OpenCV 2D windows.
class Viewer {
public:
    explicit Viewer(int width = 1280, int height = 720);

    // Draw the current map and trajectory in the Pangolin window.
    void render(const SlamMap& map);

    // Return true if the user has closed the Pangolin window.
    bool should_quit() const { return pangolin::ShouldQuit(); }

    // Non-blocking key poll (OpenCV side).
    int wait_key(int delay_ms) const { return cv::waitKey(delay_ms); }

    // Show keypoints in its own window.
    void show_keypoints(const Frame& frame, bool accepted) const;

    // Show raw and filtered match windows.
    // Pass empty vectors if one type is not available yet.
    void show_matches(
        const Frame&                   prev,
        const Frame&                   curr,
        const std::vector<cv::DMatch>& raw_matches,
        const std::vector<cv::DMatch>& filtered_matches) const;

    static cv::Mat draw_keypoints_image(
        const Frame&       frame,
        const std::string& overlay_text,
        bool               accepted);

    static cv::Mat draw_matches_image(
        const Frame&                   prev,
        const Frame&                   curr,
        const std::vector<cv::DMatch>& matches,
        const std::string&             overlay_text);

    // Save a top-down (X-Z plane) projection of the map as a PNG.
    static void save_top_down_map_image(
        const SlamMap& map,
        const std::string& output_path,
        int image_width  = 1400,
        int image_height = 1400,
        int margin       = 40);

private:
    static void draw_trajectory_and_map_gl(const SlamMap& map);

    int width_, height_;
    pangolin::OpenGlRenderState camera_render_state_;
    pangolin::View*             display_ = nullptr;
};
