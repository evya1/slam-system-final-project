#include "frontend.h"
#include "viewer.h"
#include "diagnostics.h"
#include "slam_map.h"
#include "frame.h"
#include "geometry_checks.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <iomanip>
#include <filesystem>
#include <algorithm>

namespace fs = std::filesystem;

// Dataset helpers

// Load a TUM-style timestamp file: "timestamp  relative/path"
static std::vector<std::pair<double, std::string>>
load_timestamp_file(const std::string& file_path) {
    std::vector<std::pair<double, std::string>> entries;
    std::ifstream file(file_path);
    if (!file.is_open()) return entries;
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

// Supported image extensions
static bool is_image_extension(const std::string& ext) {
    std::string e = ext;
    std::transform(e.begin(), e.end(), e.begin(), ::tolower);
    return e == ".png" || e == ".jpg" || e == ".jpeg" || e == ".bmp";
}

// Collect sorted image paths from a directory.
// If rgb.txt exists, use it (TUM style).  Otherwise scan for image files.
static std::vector<std::pair<double, std::string>>
collect_image_paths(const std::string& dataset_path) {
    const std::string rgb_txt = dataset_path + "/rgb.txt";

    // Prefer rgb.txt if it exists and is non-empty
    if (fs::exists(rgb_txt)) {
        auto entries = load_timestamp_file(rgb_txt);
        if (!entries.empty()) {
            std::cout << "loaded image list from rgb.txt  (" << entries.size() << " frames)\n";
            return entries;
        }
    }

    // Fallback: scan for image files directly.
    // Also check common sub-directories: rgb/, images/, frames/
    std::vector<std::string> search_dirs = {
        dataset_path,
        dataset_path + "/rgb",
        dataset_path + "/images",
        dataset_path + "/frames",
    };

    std::vector<std::string> found_paths;
    for (const auto& dir : search_dirs) {
        if (!fs::is_directory(dir)) continue;
        for (const auto& entry : fs::directory_iterator(dir)) {
            if (!entry.is_regular_file()) continue;
            if (!is_image_extension(entry.path().extension().string())) continue;
            found_paths.push_back(entry.path().string());
        }
        if (!found_paths.empty()) break;
    }

    if (found_paths.empty()) return {};

    // Sort lexicographically — works for zero-padded numeric filenames
    std::sort(found_paths.begin(), found_paths.end());

    std::vector<std::pair<double, std::string>> entries;
    entries.reserve(found_paths.size());
    for (size_t i = 0; i < found_paths.size(); ++i)
        entries.push_back({static_cast<double>(i), found_paths[i]});

    std::cout << "scanned directory for images  (" << entries.size() << " frames)\n";
    return entries;
}

// Camera intrinsics
//
// If no calibration file is available we build K from the first image's size.
//
// Heuristic (common in monocular VO literature):
//   fx = fy = max(W, H)       — corresponds to ~45–53° diagonal FOV
//   cx = W / 2.0
//   cy = H / 2.0
//
// This is document-quality for unknown-calibration sequences; pass explicit
// values via --fx/--fy/... flags when metric accuracy is desired.
static cv::Mat build_K_from_image(const cv::Mat& img) {
    const double W = img.cols;
    const double H = img.rows;
    const double f = std::max(W, H);   // focal length heuristic
    const double cx = W / 2.0;
    const double cy = H / 2.0;
    cv::Mat K = (cv::Mat_<double>(3, 3) <<
        f,   0.0, cx,
        0.0,  f,  cy,
        0.0, 0.0, 1.0);
    std::cout << "intrinsics (from image size " << (int)W << "×" << (int)H << "):\n"
              << "  fx = fy = " << f << "  cx = " << cx << "  cy = " << cy << "\n";
    return K;
}

// Optionally parse simple --fx / --fy / --cx / --cy command-line overrides.
static cv::Mat parse_or_build_K(int argc, char** argv, const cv::Mat& img) {
    double fx = -1, fy = -1, cx = -1, cy = -1;
    for (int i = 1; i < argc - 1; ++i) {
        std::string flag = argv[i];
        if (flag == "--fx") fx = std::stod(argv[i + 1]);
        if (flag == "--fy") fy = std::stod(argv[i + 1]);
        if (flag == "--cx") cx = std::stod(argv[i + 1]);
        if (flag == "--cy") cy = std::stod(argv[i + 1]);
    }
    if (fx > 0 && fy > 0 && cx > 0 && cy > 0) {
        std::cout << "intrinsics (from command line): fx=" << fx
                  << " fy=" << fy << " cx=" << cx << " cy=" << cy << "\n";
        return (cv::Mat_<double>(3, 3) <<
                fx,  0.0, cx,
                0.0, fy,  cy,
                0.0, 0.0, 1.0);
    }
    return build_K_from_image(img);
}

// Key event helper

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

// main

int main(int argc, char** argv) {
    std::cout << "========================================\n"
              << "  Monocular RGB SLAM (HW3)  v0.3.0\n"
              << "========================================\n";
    std::cout << "cwd: " << fs::current_path().string() << "\n";

    // Dataset path
    std::string dataset_path;
    // First non-flag argument is the dataset path
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a.size() > 2 && a.substr(0, 2) == "--") continue;
        if (i > 1 && std::string(argv[i - 1]).size() > 2 &&
            std::string(argv[i - 1]).substr(0, 2) == "--") continue;
        dataset_path = a;
        break;
    }

    if (dataset_path.empty()) {
        // Search common relative locations
        for (const std::string& c : {
                std::string("./dataset"),
                std::string("./Dataset_VO"),
                std::string("../Dataset_VO"),
                std::string("./images"),
                std::string("./rgbd_dataset_freiburg2_pioneer_slam3"),
                std::string("./") }) {
            if (fs::exists(c) && fs::is_directory(c)) { dataset_path = c; break; }
        }
    }

    if (dataset_path.empty()) {
        std::cerr << "ERROR: dataset directory not found.\n"
                  << "Usage: ./slam_system <path_to_dataset_dir> [--fx F --fy F --cx C --cy C]\n"
                  << "       Images can be in the directory itself or in rgb/ images/ frames/\n"
                  << "       A TUM-style rgb.txt is used if present, otherwise images are\n"
                  << "       loaded sorted by filename.\n";
        return 1;
    }

    std::cout << "dataset: " << dataset_path << "\n";

    // Collect image list
    auto image_list = collect_image_paths(dataset_path);
    if (image_list.empty()) {
        std::cerr << "ERROR: no images found in " << dataset_path << "\n";
        return 1;
    }

    // Build a helper to resolve relative paths from the rgb.txt
    auto resolve_path = [&](const std::string& p) -> std::string {
        if (fs::path(p).is_absolute()) return p;
        // If path does not start with the dataset_path, prepend it
        const std::string full = dataset_path + "/" + p;
        if (fs::exists(full)) return full;
        return p;  // already absolute (directory-scan case)
    };

    // Load first image to determine size and build K
    cv::Mat first_img = cv::imread(resolve_path(image_list[0].second), cv::IMREAD_COLOR);
    if (first_img.empty()) {
        std::cerr << "ERROR: cannot read first image: "
                  << resolve_path(image_list[0].second) << "\n";
        return 1;
    }
    cv::Mat K = parse_or_build_K(argc, argv, first_img);

    // Load frames into SlamMap
    SlamMap map;
    int skipped = 0;

    for (size_t i = 0; i < image_list.size(); ++i) {
        std::string path = resolve_path(image_list[i].second);

        cv::Mat bgr  = (i == 0) ? first_img : cv::imread(path, cv::IMREAD_COLOR);
        if (bgr.empty()) { ++skipped; continue; }

        cv::Mat gray;
        cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);

        Frame f;
        f.id         = map.frame_count();
        f.timestamp  = image_list[i].first;
        f.rgb_path   = path;
        f.image_bgr  = bgr;
        f.image_gray = gray;
        map.add_frame(std::move(f));
    }

    if (skipped > 0)
        std::cerr << "warning: " << skipped << " images failed to load\n";

    if (map.frame_count() < 2) {
        std::cerr << "ERROR: not enough frames loaded (" << map.frame_count() << ")\n";
        return 1;
    }

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "loaded " << map.frame_count() << " frames\n";
    std::cout << "camera matrix:\n" << K << "\n";

    // Viewer + frontend
    Viewer   viewer;
    Frontend frontend(K, 3000, &viewer);

    // Extract features for frame 0
    if (!frontend.extract_features(map.frame_at(0))) {
        std::cerr << "feature extraction failed on first frame\n";
        return 1;
    }
    // Frame 0 is the reference — identity pose (already default)
    map.frame_at(0).pose = Pose::identity();

    // Geometry self-checks (debug builds only)
#ifndef NDEBUG
    run_geometry_self_checks();
#endif

    // Diagnostics output files
    const std::string csv_path     = "slam_diagnostics.csv";
    const std::string jump_path    = "slam_jump_candidates.txt";
    const std::string summary_path = "slam_summary.txt";

    std::ofstream csv_file(csv_path);
    if (!csv_file.is_open()) {
        std::cerr << "failed to open: " << csv_path << "\n";
        return 1;
    }
    write_csv_header(csv_file);

    std::vector<StepDiagnostics> all_diags;
    bool paused      = false;
    int  frame_delay = 30;

    std::cout << "processing " << map.frame_count() << " frames...\n";
    std::cout << "controls: q/ESC quit | SPACE pause | +/- speed\n";

    // Main tracking loop
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
            << "frame " << prev.id << "->" << curr.id
            << " | " << (diag.accepted ? "OK" : "REJ")
            << " | " << diag.reason
            << " | src=" << diag.pose_source
            << " | raw=" << diag.raw_matches
            << " | good=" << diag.good_matches
            << " | epi=" << diag.epipolar_inliers
            << " | err=" << diag.epi_error
            << " | tri=" << diag.triangulated_points
            << " | pnp=" << diag.pnp_inliers
            << " | optim=" << (diag.optimizer_ran ? "Y" : "N")
            << " | reproj=" << diag.reproj_after_optim
            << " | kf=" << (diag.is_keyframe ? "Y" : "N")
            << " | kfs=" << diag.num_keyframes
            << " | lc=" << (diag.loop_correction_applied ? "Y" : "N")
            << " | map=" << map.map_point_count()
            << "\n";

        if (!handle_key_events(paused, frame_delay)) {
            std::cout << "quit by user\n";
            break;
        }
    }

    // Final output
    csv_file.close();

    write_jump_candidates_file(jump_path, all_diags);
    write_summary_file(summary_path, all_diags,
                       map.frame_count(), map.map_point_count());

    std::cout << "finished\n"
              << "wrote: " << csv_path     << "\n"
              << "wrote: " << jump_path    << "\n"
              << "wrote: " << summary_path << "\n";

    Viewer::save_top_down_map_image(map, "top_down_map.png");

    std::cout << "press any key in an OpenCV window to exit\n";
    while (!viewer.should_quit()) {
        viewer.render(map);
        if (cv::waitKey(30) >= 0) break;
    }

    cv::destroyAllWindows();
    return 0;
}
