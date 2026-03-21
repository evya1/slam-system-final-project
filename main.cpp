#include "frontend.h"
#include "viewer.h"
#include "diagnostics.h"
#include "slam_map.h"
#include "frame.h"

#include <opencv2/imgcodecs.hpp>

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <iomanip>
#include <filesystem>

namespace fs = std::filesystem;

// ---------------------------------------------------------------------------
// Dataset helpers
// ---------------------------------------------------------------------------

static bool path_exists(const std::string& p) {
    return std::ifstream(p).good();
}

static std::string find_dataset_path(int argc, char** argv) {
    if (argc >= 2) return argv[1];
    for (const std::string& c : {
            std::string("./rgbd_dataset_freiburg2_pioneer_slam3"),
            std::string("../rgbd_dataset_freiburg2_pioneer_slam3"),
            std::string("../../rgbd_dataset_freiburg2_pioneer_slam3"),
            std::string("rgbd_dataset_freiburg2_pioneer_slam3")}) {
        if (fs::exists(c) && fs::is_directory(c)) return c;
    }
    return "";
}

// Load a TUM-style timestamp file: "timestamp  relative/path"
static std::vector<std::pair<double, std::string>>
load_timestamp_file(const std::string& file_path) {
    std::vector<std::pair<double, std::string>> entries;
    std::ifstream file(file_path);
    if (!file.is_open()) {
        std::cerr << "failed to open: " << file_path << "\n";
        return entries;
    }
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line);
        double      ts;
        std::string rel;
        ss >> ts >> rel;
        if (!ss.fail()) entries.push_back({ts, rel});
    }
    return entries;
}

// ---------------------------------------------------------------------------
// Key event helper (shared between frames in the loop)
// ---------------------------------------------------------------------------

// Polls for key input until the user either steps forward or quits.
// Returns false if the user requested quit.
static bool handle_key_events(bool& paused, int& frame_delay_ms) {
    while (true) {
        int key = cv::waitKey(paused ? 0 : frame_delay_ms);
        if (key == 'q' || key == 'Q' || key == 27) return false;
        if (key == ' ') {
            paused = !paused;
            std::cout << (paused ? "paused" : "resumed") << "\n";
            continue;
        }
        if (key == '+' || key == '=')
            frame_delay_ms = std::max(1, frame_delay_ms - 10);
        if (key == '-' || key == '_')
            frame_delay_ms = std::min(500, frame_delay_ms + 10);
        if (!paused) return true;
    }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
    std::cout << "========================================\n"
              << "  Monocular RGB SLAM (HW3)\n"
              << "========================================\n";
    std::cout << "cwd: " << fs::current_path().string() << "\n";

    // ------------------------------------------------------------------
    // Dataset loading
    // ------------------------------------------------------------------
    std::string dataset_path = find_dataset_path(argc, argv);
    if (dataset_path.empty()) {
        std::cerr << "dataset directory not found.\n"
                  << "Usage: ./slam_system <path_to_dataset>\n";
        return 1;
    }

    std::string rgb_txt = dataset_path + "/rgb.txt";
    if (!path_exists(rgb_txt)) {
        std::cerr << "could not find rgb.txt at: " << rgb_txt << "\n";
        return 1;
    }

    auto rgb_entries = load_timestamp_file(rgb_txt);
    if (rgb_entries.empty()) {
        std::cerr << "no rgb entries found\n";
        return 1;
    }

    // ------------------------------------------------------------------
    // Camera intrinsics (Freiburg2)
    // ------------------------------------------------------------------
    const double fx = 520.9, fy = 521.0, cx = 325.1, cy = 249.7;
    cv::Mat K = (cv::Mat_<double>(3, 3) <<
        fx, 0.0, cx,
        0.0, fy, cy,
        0.0, 0.0, 1.0);

    // ------------------------------------------------------------------
    // Load frames (monocular — RGB only)
    // ------------------------------------------------------------------
    SlamMap map;

    for (size_t i = 0; i < rgb_entries.size(); ++i) {
        std::string rgb_path = dataset_path + "/" + rgb_entries[i].second;
        cv::Mat bgr  = cv::imread(rgb_path, cv::IMREAD_COLOR);
        cv::Mat gray = cv::imread(rgb_path, cv::IMREAD_GRAYSCALE);
        if (bgr.empty() || gray.empty()) {
            std::cerr << "failed to load rgb frame: " << rgb_path << "\n";
            continue;
        }

        Frame f;
        f.id        = map.frame_count();
        f.timestamp = rgb_entries[i].first;
        f.rgb_path  = rgb_path;
        f.image_bgr  = bgr;
        f.image_gray = gray;
        map.add_frame(std::move(f));
    }

    if (map.frame_count() < 2) {
        std::cerr << "not enough frames loaded (" << map.frame_count() << ")\n";
        return 1;
    }

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "loaded " << map.frame_count() << " RGB frames\n";
    std::cout << "camera matrix:\n" << K << "\n";

    // ------------------------------------------------------------------
    // Viewer + frontend
    // ------------------------------------------------------------------
    Viewer   viewer;
    Frontend frontend(K);

    // Extract features for frame 0
    if (!frontend.extract_features(map.frame_at(0))) {
        std::cerr << "feature extraction failed on first frame\n";
        return 1;
    }

    // ------------------------------------------------------------------
    // Diagnostics output files
    // ------------------------------------------------------------------
    const std::string csv_path    = "slam_diagnostics.csv";
    const std::string jump_path   = "slam_jump_candidates.txt";
    const std::string summary_path= "slam_summary.txt";

    std::ofstream csv_file(csv_path);
    if (!csv_file.is_open()) {
        std::cerr << "failed to open: " << csv_path << "\n";
        return 1;
    }
    write_csv_header(csv_file);

    std::vector<StepDiagnostics> all_diags;
    bool paused        = false;
    int  frame_delay   = 30;

    std::cout << "processing " << map.frame_count() << " frames...\n";
    std::cout << "controls: q/ESC quit | SPACE pause | +/- speed\n";

    // ------------------------------------------------------------------
    // Main tracking loop
    // ------------------------------------------------------------------
    for (int idx = 1; idx < map.frame_count(); ++idx) {
        if (viewer.should_quit()) break;

        Frame& prev = map.frame_at(idx - 1);
        Frame& curr = map.frame_at(idx);

        if (!curr.processed)
            frontend.extract_features(curr);

        viewer.show_keypoints(curr, /*accepted=*/false);

        StepDiagnostics diag = frontend.process(prev, curr, map);
        all_diags.push_back(diag);
        write_csv_row(csv_file, diag);

        viewer.render(map);

        std::cout
            << "frame " << prev.id << " -> " << curr.id
            << " | accepted: "      << (diag.accepted ? 1 : 0)
            << " | reason: "        << diag.reason
            << " | pose_source: "   << diag.pose_source
            << " | recovery: "      << (diag.recovery_mode ? 1 : 0)
            << " | raw: "           << diag.raw_matches
            << " | good: "          << diag.good_matches
            << " | epi_inliers: "   << diag.epipolar_inliers
            << " | inlier_ratio: "  << diag.inlier_ratio
            << " | epi_err: "       << diag.epi_error
            << " | triangulated: "  << diag.triangulated_points
            << " | pnp_inliers: "   << diag.pnp_inliers
            << " | reproj: "        << diag.reproj_error
            << " | step_r_deg: "    << diag.step_r_deg
            << " | consec_rej: "    << diag.consecutive_reject_count
            << " | map_pts: "       << map.map_point_count()
            << "\n";

        if (!handle_key_events(paused, frame_delay)) {
            std::cout << "quit requested by user\n";
            break;
        }
    }

    // ------------------------------------------------------------------
    // Final output
    // ------------------------------------------------------------------
    csv_file.close();

    write_jump_candidates_file(jump_path, all_diags);
    write_summary_file(summary_path, all_diags,
                       map.frame_count(), map.map_point_count());

    std::cout << "finished\n"
              << "wrote: " << csv_path     << "\n"
              << "wrote: " << jump_path    << "\n"
              << "wrote: " << summary_path << "\n";

    Viewer::save_top_down_map_image(map, "top_down_map.png");

    // Keep viewer alive until user closes it
    std::cout << "press any key in the OpenCV window to exit\n";
    while (!viewer.should_quit()) {
        viewer.render(map);
        if (cv::waitKey(30) >= 0) break;
    }

    cv::destroyAllWindows();
    return 0;
}
