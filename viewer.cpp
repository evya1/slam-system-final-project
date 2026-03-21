#include "viewer.h"

#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <iostream>
#include <limits>
#include <algorithm>

// ---------------------------------------------------------------------------
// Construction / Pangolin setup
// ---------------------------------------------------------------------------

Viewer::Viewer(int width, int height)
    : width_(width), height_(height)
{
    pangolin::CreateWindowAndBind("trajectory_and_map", width_, height_);
    glEnable(GL_DEPTH_TEST);
    glClearColor(0.1f, 0.1f, 0.1f, 1.0f);

    camera_render_state_ = pangolin::OpenGlRenderState(
        pangolin::ProjectionMatrix(width_, height_, 500, 500, 640, 360, 0.1, 1000.0),
        pangolin::ModelViewLookAt(-2.0, -2.0, -2.0, 0.0, 0.0, 0.0, pangolin::AxisY));

    display_ = &pangolin::CreateDisplay()
        .SetBounds(0.0, 1.0, 0.0, 1.0,
                   -static_cast<float>(width_) / static_cast<float>(height_))
        .SetHandler(new pangolin::Handler3D(camera_render_state_));

    cv::namedWindow("SLAM: Keypoints", cv::WINDOW_NORMAL);
    cv::resizeWindow("SLAM: Keypoints", 960, 540);
    cv::moveWindow("SLAM: Keypoints", 50, 500);
    cv::waitKey(1);
}

// ---------------------------------------------------------------------------
// Render (called once per frame)
// ---------------------------------------------------------------------------

void Viewer::render(const SlamMap& map) {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    display_->Activate(camera_render_state_);
    draw_trajectory_and_map_gl(map);
    pangolin::FinishFrame();
}

void Viewer::show_keypoints(const Frame& frame, bool accepted) const {
    const std::string text =
        "frame: " + std::to_string(frame.id) +
        "  keypoints: " + std::to_string(frame.keypoints.size());
    cv::Mat img = draw_keypoints_image(frame, text, accepted);
    cv::imshow("SLAM: Keypoints", img);
}

// ---------------------------------------------------------------------------
// Static OpenGL drawing
// ---------------------------------------------------------------------------

void Viewer::draw_trajectory_and_map_gl(const SlamMap& map) {
    // Map points
    glPointSize(2.0f);
    glBegin(GL_POINTS);
    glColor3f(1.0f, 1.0f, 1.0f);
    for (const MapPoint& mp : map.map_points()) {
        if (!mp.valid) continue;
        glVertex3d(mp.position.x(), mp.position.y(), mp.position.z());
    }
    glEnd();

    // Trajectory line
    glLineWidth(2.0f);
    glBegin(GL_LINE_STRIP);
    glColor3f(1.0f, 0.0f, 0.0f);
    for (const Frame& f : map.frames()) {
        const auto& t = f.pose.translation_vector;
        glVertex3d(t.x(), t.y(), t.z());
    }
    glEnd();

    // Per-frame axes
    for (const Frame& f : map.frames()) {
        glPushMatrix();
        Eigen::Matrix4d T = f.pose.matrix();
        glMultMatrixd(T.data());
        pangolin::glDrawAxis(0.1);
        glPopMatrix();
    }
}

// ---------------------------------------------------------------------------
// Static image helpers
// ---------------------------------------------------------------------------

cv::Mat Viewer::draw_keypoints_image(
    const Frame& frame,
    const std::string& overlay_text,
    bool accepted)
{
    cv::Mat out;
    cv::Scalar colour = accepted ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
    cv::drawKeypoints(frame.image_bgr, frame.keypoints, out, colour);
    cv::putText(out, overlay_text,
                cv::Point(20, 35), cv::FONT_HERSHEY_SIMPLEX, 0.8,
                cv::Scalar(255, 255, 255), 2);
    return out;
}

cv::Mat Viewer::draw_matches_image(
    const Frame& prev,
    const Frame& curr,
    const std::vector<cv::DMatch>& matches,
    const std::string& overlay_text)
{
    cv::Mat out;
    cv::drawMatches(prev.image_bgr, prev.keypoints,
                    curr.image_bgr, curr.keypoints,
                    matches, out);
    cv::putText(out, overlay_text,
                cv::Point(20, 35), cv::FONT_HERSHEY_SIMPLEX, 0.8,
                cv::Scalar(0, 255, 0), 2);
    return out;
}

void Viewer::save_top_down_map_image(
    const SlamMap& map,
    const std::string& output_path,
    int image_width,
    int image_height,
    int margin)
{
    if (map.frames().empty()) {
        std::cerr << "cannot save top-down map: no frames\n";
        return;
    }

    // Collect all XZ points
    std::vector<Eigen::Vector2d> all_xz;
    all_xz.reserve(map.map_point_count() + map.frame_count());

    for (const MapPoint& mp : map.map_points()) {
        if (!mp.valid || !mp.position.allFinite()) continue;
        all_xz.emplace_back(mp.position.x(), mp.position.z());
    }
    for (const Frame& f : map.frames()) {
        if (!f.pose.translation_vector.allFinite()) continue;
        all_xz.emplace_back(f.pose.translation_vector.x(),
                             f.pose.translation_vector.z());
    }

    if (all_xz.empty()) {
        std::cerr << "cannot save top-down map: no valid XZ points\n";
        return;
    }

    double min_x =  std::numeric_limits<double>::max();
    double max_x = -std::numeric_limits<double>::max();
    double min_z =  std::numeric_limits<double>::max();
    double max_z = -std::numeric_limits<double>::max();

    for (const auto& p : all_xz) {
        min_x = std::min(min_x, p.x()); max_x = std::max(max_x, p.x());
        min_z = std::min(min_z, p.y()); max_z = std::max(max_z, p.y());
    }

    double range_x = std::max(max_x - min_x, 1e-6);
    double range_z = std::max(max_z - min_z, 1e-6);
    double scale   = std::min(
        static_cast<double>(image_width  - 2 * margin) / range_x,
        static_cast<double>(image_height - 2 * margin) / range_z);

    auto to_pixel = [&](double wx, double wz) -> cv::Point {
        return {
            static_cast<int>(margin + (wx - min_x) * scale),
            static_cast<int>(image_height - margin - (wz - min_z) * scale)
        };
    };

    cv::Mat img(image_height, image_width, CV_8UC3, cv::Scalar(0, 0, 0));

    // Map points
    for (const MapPoint& mp : map.map_points()) {
        if (!mp.valid || !mp.position.allFinite()) continue;
        cv::Point px = to_pixel(mp.position.x(), mp.position.z());
        if (px.x >= 0 && px.x < image_width && px.y >= 0 && px.y < image_height)
            img.at<cv::Vec3b>(px.y, px.x) = cv::Vec3b(255, 255, 255);
    }

    // Trajectory
    const auto& frames = map.frames();
    for (size_t i = 1; i < frames.size(); ++i) {
        const auto& a = frames[i - 1].pose.translation_vector;
        const auto& b = frames[i].pose.translation_vector;
        if (!a.allFinite() || !b.allFinite()) continue;
        cv::line(img, to_pixel(a.x(), a.z()), to_pixel(b.x(), b.z()),
                 cv::Scalar(0, 0, 255), 2);
    }
    for (const Frame& f : frames) {
        if (!f.pose.translation_vector.allFinite()) continue;
        cv::circle(img, to_pixel(f.pose.translation_vector.x(),
                                  f.pose.translation_vector.z()),
                   2, cv::Scalar(255, 0, 0), -1);
    }
    if (!frames.empty() && frames.back().pose.translation_vector.allFinite()) {
        const auto& t = frames.back().pose.translation_vector;
        cv::circle(img, to_pixel(t.x(), t.z()), 6, cv::Scalar(0, 255, 0), -1);
    }

    cv::putText(img, "Top-down map view (X-Z plane)",
                cv::Point(20, 35), cv::FONT_HERSHEY_SIMPLEX, 0.8,
                cv::Scalar(0, 255, 255), 2);

    if (cv::imwrite(output_path, img))
        std::cout << "saved top-down map image to: " << output_path << "\n";
    else
        std::cerr << "failed to save top-down map image to: " << output_path << "\n";
}
