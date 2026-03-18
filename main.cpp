#include <pangolin/pangolin.h>

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>

#include <Eigen/Dense>

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <limits>
#include <deque>

using namespace std;

struct Pose
{
    Eigen::Matrix3d rotation_matrix;
    Eigen::Vector3d translation_vector;

    Pose()
    {
        rotation_matrix = Eigen::Matrix3d::Identity();
        translation_vector = Eigen::Vector3d::Zero();
    }

    Eigen::Matrix4d matrix() const
    {
        Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
        transform.block<3,3>(0,0) = rotation_matrix;
        transform.block<3,1>(0,3) = translation_vector;
        return transform;
    }
};

struct Frame
{
    int id;
    double rgb_timestamp;
    double depth_timestamp;
    string rgb_path;
    string depth_path;

    cv::Mat image_gray;
    cv::Mat image_bgr;
    cv::Mat depth_image;

    vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;

    Pose pose;
    bool processed;

    Frame()
    {
        id = -1;
        rgb_timestamp = 0.0;
        depth_timestamp = 0.0;
        processed = false;
    }
};

struct MapPoint
{
    int id;
    Eigen::Vector3d point;
    bool valid;

    MapPoint()
    {
        id = -1;
        point = Eigen::Vector3d::Zero();
        valid = true;
    }
};

struct MapData
{
    vector<Frame> frames;
    vector<MapPoint> points;
};

struct MatchInfo
{
    vector<cv::DMatch> raw_matches;
    vector<cv::DMatch> good_matches;
    vector<cv::DMatch> pnp_inlier_matches;

    vector<cv::Point2f> points_prev_2d;
    vector<cv::Point2f> points_curr_2d;

    vector<cv::Point3f> points_prev_3d_for_pnp;
    vector<cv::Point2f> points_curr_2d_for_pnp;

    vector<int> good_match_indices_used_for_pnp;
    vector<int> pnp_inlier_indices;

    double reproj_before = -1.0;
    double reproj_after = -1.0;
    double essential_epi_before = -1.0;
    double essential_epi_after = -1.0;
};

struct StepDiagnostics
{
    int prev_frame_id = -1;
    int curr_frame_id = -1;

    int raw_matches = 0;
    int good_matches = 0;
    int depth_valid_3d2d = 0;
    int pnp_inliers = 0;

    double inlier_ratio = -1.0;
    double depth_valid_ratio = -1.0;
    double reproj_before = -1.0;
    double reproj_after = -1.0;
    double epi_before = -1.0;
    double epi_after = -1.0;
    double step_t_norm = -1.0;
    double step_r_deg = -1.0;

    int consecutive_reject_count = 0;
    int last_accepted_frame = -1;
    int frames_since_last_accept = -1;
    int map_points_added_this_step = 0;

    bool recovery_mode = false;
    bool accepted = false;
    string accepted_pose_source = "none";
    string reason = "unknown";
};

struct AcceptedMotionStats
{
    double median_step_t = 0.02;
    double median_step_r_deg = 1.0;
};

static bool path_exists(const string& path)
{
    ifstream file(path);
    return file.good();
}

static vector<pair<double, string>> load_timestamp_file(const string& file_path)
{
    vector<pair<double, string>> entries;

    ifstream file(file_path);
    if (!file.is_open())
    {
        cerr << "failed to open file: " << file_path << endl;
        return entries;
    }

    string line;
    while (getline(file, line))
    {
        if (line.empty() || line[0] == '#')
        {
            continue;
        }

        stringstream stream(line);
        double timestamp;
        string relative_path;
        stream >> timestamp >> relative_path;

        if (!stream.fail())
        {
            entries.push_back({timestamp, relative_path});
        }
    }

    return entries;
}

static int find_closest_depth_index(
    double rgb_timestamp,
    const vector<pair<double, string>>& depth_entries,
    double max_time_diff_sec)
{
    if (depth_entries.empty())
    {
        return -1;
    }

    int best_index = -1;
    double best_diff = numeric_limits<double>::max();

    for (int i = 0; i < static_cast<int>(depth_entries.size()); ++i)
    {
        double diff = fabs(depth_entries[i].first - rgb_timestamp);
        if (diff < best_diff)
        {
            best_diff = diff;
            best_index = i;
        }
    }

    if (best_diff > max_time_diff_sec)
    {
        return -1;
    }

    return best_index;
}

static bool extract_features(Frame& frame, cv::Ptr<cv::ORB>& orb)
{
    if (frame.image_gray.empty())
    {
        return false;
    }

    orb->detectAndCompute(frame.image_gray, cv::noArray(), frame.keypoints, frame.descriptors);
    frame.processed = true;

    return !frame.keypoints.empty() && !frame.descriptors.empty();
}

static MatchInfo match_features(const Frame& frame_prev, const Frame& frame_curr)
{
    MatchInfo match_info;

    if (frame_prev.descriptors.empty() || frame_curr.descriptors.empty())
    {
        return match_info;
    }

    cv::BFMatcher matcher(cv::NORM_HAMMING);
    vector<vector<cv::DMatch>> knn_matches;
    matcher.knnMatch(frame_prev.descriptors, frame_curr.descriptors, knn_matches, 2);

    for (const auto& pair_matches : knn_matches)
    {
        if (pair_matches.size() < 2)
        {
            continue;
        }

        const cv::DMatch& best_match = pair_matches[0];
        const cv::DMatch& second_match = pair_matches[1];

        match_info.raw_matches.push_back(best_match);

        if (best_match.distance < 0.72f * second_match.distance)
        {
            match_info.good_matches.push_back(best_match);
        }
    }

    if (match_info.good_matches.empty())
    {
        return match_info;
    }

    double min_distance = numeric_limits<double>::max();
    for (const auto& match : match_info.good_matches)
    {
        min_distance = min(min_distance, static_cast<double>(match.distance));
    }

    vector<cv::DMatch> filtered_matches;
    filtered_matches.reserve(match_info.good_matches.size());

    for (const auto& match : match_info.good_matches)
    {
        if (match.distance <= max(2.2 * min_distance, 28.0))
        {
            filtered_matches.push_back(match);
        }
    }

    match_info.good_matches = filtered_matches;

    for (const auto& match : match_info.good_matches)
    {
        match_info.points_prev_2d.push_back(frame_prev.keypoints[match.queryIdx].pt);
        match_info.points_curr_2d.push_back(frame_curr.keypoints[match.trainIdx].pt);
    }

    return match_info;
}

static bool get_depth_in_meters_single(
    const cv::Mat& depth_image,
    int u,
    int v,
    double depth_scale,
    double& depth_meters)
{
    if (depth_image.empty())
    {
        return false;
    }

    if (u < 0 || v < 0 || u >= depth_image.cols || v >= depth_image.rows)
    {
        return false;
    }

    if (depth_image.type() == CV_16UC1)
    {
        uint16_t raw_depth = depth_image.at<uint16_t>(v, u);
        if (raw_depth == 0)
        {
            return false;
        }

        depth_meters = static_cast<double>(raw_depth) / depth_scale;
        return std::isfinite(depth_meters) && depth_meters > 0.0;
    }

    if (depth_image.type() == CV_32FC1)
    {
        float raw_depth = depth_image.at<float>(v, u);
        if (!std::isfinite(raw_depth) || raw_depth <= 0.0f)
        {
            return false;
        }

        depth_meters = static_cast<double>(raw_depth);
        return true;
    }

    return false;
}


static bool get_depth_in_meters_robust(
    const cv::Mat& depth_image,
    int center_u,
    int center_v,
    double depth_scale,
    double& depth_meters,
    double& depth_spread_meters,
    int radius_pixels = 2,
    int min_valid_samples = 5)
{
    vector<double> valid_depths;
    valid_depths.reserve((2 * radius_pixels + 1) * (2 * radius_pixels + 1));

    for (int dv = -radius_pixels; dv <= radius_pixels; ++dv)
    {
        for (int du = -radius_pixels; du <= radius_pixels; ++du)
        {
            int u = center_u + du;
            int v = center_v + dv;

            double d = 0.0;
            if (!get_depth_in_meters_single(depth_image, u, v, depth_scale, d))
            {
                continue;
            }

            if (d < 0.25 || d > 4.0)
            {
                continue;
            }

            valid_depths.push_back(d);
        }
    }

    if (static_cast<int>(valid_depths.size()) < min_valid_samples)
    {
        return false;
    }

    sort(valid_depths.begin(), valid_depths.end());
    size_t n = valid_depths.size();

    if (n % 2 == 1)
    {
        depth_meters = valid_depths[n / 2];
    }
    else
    {
        depth_meters = 0.5 * (valid_depths[n / 2 - 1] + valid_depths[n / 2]);
    }

    depth_spread_meters = valid_depths.back() - valid_depths.front();

    if (!std::isfinite(depth_meters) || depth_meters <= 0.0)
    {
        return false;
    }

    return true;
}


static cv::Point3f pixel_to_camera_3d(
    double u,
    double v,
    double depth_meters,
    const cv::Mat& camera_matrix)
{
    double fx = camera_matrix.at<double>(0,0);
    double fy = camera_matrix.at<double>(1,1);
    double cx = camera_matrix.at<double>(0,2);
    double cy = camera_matrix.at<double>(1,2);

    double x = (u - cx) * depth_meters / fx;
    double y = (v - cy) * depth_meters / fy;
    double z = depth_meters;

    return cv::Point3f(static_cast<float>(x), static_cast<float>(y), static_cast<float>(z));
}

static void enforce_spatially_balanced_pnp_correspondences(
    const Frame& frame_prev,
    const Frame& frame_curr,
    MatchInfo& match_info,
    int grid_cols = 4,
    int grid_rows = 3,
    int max_points_per_cell = 12)
{
    if (match_info.points_prev_3d_for_pnp.empty() || match_info.points_curr_2d_for_pnp.empty())
    {
        return;
    }

    const int image_width = frame_curr.image_gray.cols;
    const int image_height = frame_curr.image_gray.rows;

    if (image_width <= 0 || image_height <= 0)
    {
        return;
    }

    struct Candidate
    {
        cv::Point3f point3d;
        cv::Point2f point2d;
        int good_match_index;
        int cell_index;
    };

    vector<vector<Candidate>> cell_candidates(grid_cols * grid_rows);

    for (int i = 0; i < static_cast<int>(match_info.points_prev_3d_for_pnp.size()); ++i)
    {
        const cv::Point2f& curr_pt = match_info.points_curr_2d_for_pnp[i];

        int cell_x = static_cast<int>((curr_pt.x / static_cast<double>(image_width)) * grid_cols);
        int cell_y = static_cast<int>((curr_pt.y / static_cast<double>(image_height)) * grid_rows);

        cell_x = std::max(0, std::min(grid_cols - 1, cell_x));
        cell_y = std::max(0, std::min(grid_rows - 1, cell_y));

        int cell_index = cell_y * grid_cols + cell_x;

        Candidate c;
        c.point3d = match_info.points_prev_3d_for_pnp[i];
        c.point2d = match_info.points_curr_2d_for_pnp[i];
        c.good_match_index = match_info.good_match_indices_used_for_pnp[i];
        c.cell_index = cell_index;

        cell_candidates[cell_index].push_back(c);
    }

    vector<cv::Point3f> balanced_3d;
    vector<cv::Point2f> balanced_2d;
    vector<int> balanced_good_match_indices;

    for (int cell_index = 0; cell_index < static_cast<int>(cell_candidates.size()); ++cell_index)
    {
        int kept = 0;
        for (const Candidate& c : cell_candidates[cell_index])
        {
            if (kept >= max_points_per_cell)
            {
                break;
            }

            balanced_3d.push_back(c.point3d);
            balanced_2d.push_back(c.point2d);
            balanced_good_match_indices.push_back(c.good_match_index);
            kept++;
        }
    }

    match_info.points_prev_3d_for_pnp = balanced_3d;
    match_info.points_curr_2d_for_pnp = balanced_2d;
    match_info.good_match_indices_used_for_pnp = balanced_good_match_indices;
}

static void build_pnp_correspondences_from_prev_depth(
    const Frame& frame_prev,
    const Frame& frame_curr,
    const cv::Mat& camera_matrix,
    double depth_scale,
    MatchInfo& match_info)
{
    for (int i = 0; i < static_cast<int>(match_info.good_matches.size()); ++i)
    {
        const cv::DMatch& match = match_info.good_matches[i];

        cv::Point2f prev_pt = frame_prev.keypoints[match.queryIdx].pt;
        cv::Point2f curr_pt = frame_curr.keypoints[match.trainIdx].pt;

        int u = static_cast<int>(std::round(prev_pt.x));
        int v = static_cast<int>(std::round(prev_pt.y));

        double depth_meters = 0.0;
        double depth_spread_meters = 0.0;

        bool ok = get_depth_in_meters_robust(
            frame_prev.depth_image,
            u,
            v,
            depth_scale,
            depth_meters,
            depth_spread_meters,
            2,
            5
        );

        if (!ok)
        {
            continue;
        }

        double allowed_spread = std::max(0.08, 0.08 * depth_meters);
        if (depth_spread_meters > allowed_spread)
        {
            continue;
        }

        cv::Point3f p3d = pixel_to_camera_3d(prev_pt.x, prev_pt.y, depth_meters, camera_matrix);

        match_info.points_prev_3d_for_pnp.push_back(p3d);
        match_info.points_curr_2d_for_pnp.push_back(curr_pt);
        match_info.good_match_indices_used_for_pnp.push_back(i);
    }

    enforce_spatially_balanced_pnp_correspondences(
        frame_prev,
        frame_curr,
        match_info,
        4,
        3,
        12
    );
}

static double compute_mean_reprojection_error(
    const vector<cv::Point3f>& object_points,
    const vector<cv::Point2f>& image_points,
    const cv::Mat& rvec,
    const cv::Mat& tvec,
    const cv::Mat& camera_matrix,
    const cv::Mat& dist_coeffs,
    const vector<int>* selected_indices = nullptr)
{
    if (object_points.empty() || image_points.empty())
    {
        return -1.0;
    }

    vector<cv::Point2f> projected_points;
    cv::projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs, projected_points);

    double total_error = 0.0;
    int count = 0;

    if (selected_indices == nullptr)
    {
        for (size_t i = 0; i < projected_points.size(); ++i)
        {
            double dx = projected_points[i].x - image_points[i].x;
            double dy = projected_points[i].y - image_points[i].y;
            total_error += std::sqrt(dx * dx + dy * dy);
            count++;
        }
    }
    else
    {
        for (int idx : *selected_indices)
        {
            if (idx < 0 || idx >= static_cast<int>(projected_points.size()))
            {
                continue;
            }

            double dx = projected_points[idx].x - image_points[idx].x;
            double dy = projected_points[idx].y - image_points[idx].y;
            total_error += std::sqrt(dx * dx + dy * dy);
            count++;
        }
    }

    if (count == 0)
    {
        return -1.0;
    }

    return total_error / static_cast<double>(count);
}

static double compute_mean_epipolar_error(
    const vector<cv::Point2f>& points_prev,
    const vector<cv::Point2f>& points_curr,
    const cv::Mat& fundamental_matrix)
{
    if (points_prev.empty() || points_curr.empty() || fundamental_matrix.empty())
    {
        return -1.0;
    }

    cv::Mat F;
    fundamental_matrix.convertTo(F, CV_64F);

    double total_error = 0.0;
    int count = 0;

    for (size_t i = 0; i < points_prev.size(); ++i)
    {
        cv::Mat x1 = (cv::Mat_<double>(3,1) << points_prev[i].x, points_prev[i].y, 1.0);
        cv::Mat line2 = F * x1;

        double a = line2.at<double>(0,0);
        double b = line2.at<double>(1,0);
        double c = line2.at<double>(2,0);

        double numerator = fabs(a * points_curr[i].x + b * points_curr[i].y + c);
        double denominator = sqrt(a * a + b * b);

        if (denominator > 1e-12)
        {
            total_error += numerator / denominator;
            count++;
        }
    }

    if (count == 0)
    {
        return -1.0;
    }

    return total_error / static_cast<double>(count);
}

static bool estimate_pose_pnp(
    MatchInfo& match_info,
    const cv::Mat& camera_matrix,
    cv::Mat& rvec,
    cv::Mat& tvec)
{
    if (match_info.points_prev_3d_for_pnp.size() < 10 || match_info.points_curr_2d_for_pnp.size() < 10)
    {
        return false;
    }

    cv::Mat inlier_indices_mat;
    cv::Mat dist_coeffs = cv::Mat::zeros(4, 1, CV_64F);

    bool ok = cv::solvePnPRansac(
        match_info.points_prev_3d_for_pnp,
        match_info.points_curr_2d_for_pnp,
        camera_matrix,
        dist_coeffs,
        rvec,
        tvec,
        false,
        150,
        3.0,
        0.995,
        inlier_indices_mat,
        cv::SOLVEPNP_ITERATIVE
    );

    if (!ok || inlier_indices_mat.rows < 10)
    {
        return false;
    }

    match_info.pnp_inlier_indices.clear();
    match_info.pnp_inlier_matches.clear();

    for (int i = 0; i < inlier_indices_mat.rows; ++i)
    {
        match_info.pnp_inlier_indices.push_back(inlier_indices_mat.at<int>(i,0));
    }

    for (int idx_in_pnp_set : match_info.pnp_inlier_indices)
    {
        if (idx_in_pnp_set < 0 || idx_in_pnp_set >= static_cast<int>(match_info.good_match_indices_used_for_pnp.size()))
        {
            continue;
        }

        int idx_in_good_matches = match_info.good_match_indices_used_for_pnp[idx_in_pnp_set];
        if (idx_in_good_matches >= 0 && idx_in_good_matches < static_cast<int>(match_info.good_matches.size()))
        {
            match_info.pnp_inlier_matches.push_back(match_info.good_matches[idx_in_good_matches]);
        }
    }

    match_info.reproj_before = compute_mean_reprojection_error(
        match_info.points_prev_3d_for_pnp,
        match_info.points_curr_2d_for_pnp,
        cv::Mat::zeros(3,1,CV_64F),
        cv::Mat::zeros(3,1,CV_64F),
        camera_matrix,
        dist_coeffs
    );

    match_info.reproj_after = compute_mean_reprojection_error(
        match_info.points_prev_3d_for_pnp,
        match_info.points_curr_2d_for_pnp,
        rvec,
        tvec,
        camera_matrix,
        dist_coeffs,
        &match_info.pnp_inlier_indices
    );

    return true;
}

static void update_frame_pose_from_pnp(
    Frame& frame_curr,
    const Frame& frame_prev,
    const cv::Mat& rvec,
    const cv::Mat& tvec)
{
    cv::Mat rotation_matrix_cv;
    cv::Rodrigues(rvec, rotation_matrix_cv);

    cv::Mat rotation_64f;
    cv::Mat translation_64f;

    rotation_matrix_cv.convertTo(rotation_64f, CV_64F);
    tvec.convertTo(translation_64f, CV_64F);

    Eigen::Matrix3d R_cp;
    Eigen::Vector3d t_cp;

    for (int r = 0; r < 3; ++r)
    {
        for (int c = 0; c < 3; ++c)
        {
            R_cp(r,c) = rotation_64f.at<double>(r,c);
        }
        t_cp(r) = translation_64f.at<double>(r,0);
    }

    Eigen::Matrix3d R_pc = R_cp.transpose();
    Eigen::Vector3d t_pc = -R_cp.transpose() * t_cp;

    frame_curr.pose.rotation_matrix = frame_prev.pose.rotation_matrix * R_pc;
    frame_curr.pose.translation_vector = frame_prev.pose.translation_vector + frame_prev.pose.rotation_matrix * t_pc;
}

static int add_current_frame_depth_points_to_map(
    const Frame& frame_curr,
    const cv::Mat& camera_matrix,
    double depth_scale,
    int step_pixels,
    int& next_map_point_id,
    MapData& map_data)
{
    if (frame_curr.depth_image.empty())
    {
        return 0;
    }

    int added_count = 0;

    for (int v = 0; v < frame_curr.depth_image.rows; v += step_pixels)
    {
        for (int u = 0; u < frame_curr.depth_image.cols; u += step_pixels)
        {
            double depth_meters = 0.0;
            bool ok = get_depth_in_meters_single(frame_curr.depth_image, u, v, depth_scale, depth_meters);
            if (!ok)
            {
                continue;
            }

            if (depth_meters < 0.25 || depth_meters > 4.0)
            {
                continue;
            }

            cv::Point3f pc = pixel_to_camera_3d(static_cast<double>(u), static_cast<double>(v), depth_meters, camera_matrix);

            Eigen::Vector3d p_camera(pc.x, pc.y, pc.z);
            Eigen::Vector3d p_world =
                frame_curr.pose.rotation_matrix * p_camera +
                frame_curr.pose.translation_vector;

            if (!p_world.allFinite())
            {
                continue;
            }

            MapPoint map_point;
            map_point.id = next_map_point_id++;
            map_point.point = p_world;
            map_data.points.push_back(map_point);
            added_count++;
        }
    }

    return added_count;
}

static cv::Mat draw_matches_image(
    const Frame& frame_prev,
    const Frame& frame_curr,
    const vector<cv::DMatch>& matches,
    const string& overlay_text)
{
    cv::Mat output;
    cv::drawMatches(
        frame_prev.image_bgr,
        frame_prev.keypoints,
        frame_curr.image_bgr,
        frame_curr.keypoints,
        matches,
        output
    );

    cv::putText(
        output,
        overlay_text,
        cv::Point(20, 35),
        cv::FONT_HERSHEY_SIMPLEX,
        0.8,
        cv::Scalar(0, 255, 0),
        2
    );

    return output;
}

static cv::Mat draw_keypoints_image(
    const Frame& frame_curr,
    const string& overlay_text,
    bool accepted)
{
    cv::Mat output;
    cv::Scalar color = accepted ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
    cv::drawKeypoints(frame_curr.image_bgr, frame_curr.keypoints, output, color);

    cv::putText(
        output,
        overlay_text,
        cv::Point(20, 35),
        cv::FONT_HERSHEY_SIMPLEX,
        0.8,
        cv::Scalar(255, 255, 255),
        2
    );

    return output;
}

static void draw_trajectory_and_map(const MapData& map_data)
{
    glPointSize(2.0f);
    glBegin(GL_POINTS);
    glColor3f(1.0f, 1.0f, 1.0f);
    for (const auto& map_point : map_data.points)
    {
        if (!map_point.valid)
        {
            continue;
        }
        glVertex3d(map_point.point.x(), map_point.point.y(), map_point.point.z());
    }
    glEnd();

    glLineWidth(2.0f);
    glBegin(GL_LINE_STRIP);
    glColor3f(1.0f, 0.0f, 0.0f);
    for (const auto& frame : map_data.frames)
    {
        glVertex3d(
            frame.pose.translation_vector.x(),
            frame.pose.translation_vector.y(),
            frame.pose.translation_vector.z()
        );
    }
    glEnd();

    for (const auto& frame : map_data.frames)
    {
        glPushMatrix();
        Eigen::Matrix4d transform = frame.pose.matrix();
        glMultMatrixd(transform.data());
        pangolin::glDrawAxis(0.1);
        glPopMatrix();
    }
}

static double compute_rotation_angle_deg_from_rvec(const cv::Mat& rvec)
{
    if (rvec.empty())
    {
        return -1.0;
    }

    cv::Mat rvec64;
    rvec.convertTo(rvec64, CV_64F);
    double angle_rad = cv::norm(rvec64);
    return angle_rad * 180.0 / CV_PI;
}

static double median_of_vector(vector<double> values, double fallback_value)
{
    if (values.empty())
    {
        return fallback_value;
    }

    sort(values.begin(), values.end());
    size_t n = values.size();
    if (n % 2 == 1)
    {
        return values[n / 2];
    }
    return 0.5 * (values[n / 2 - 1] + values[n / 2]);
}

static AcceptedMotionStats compute_recent_motion_stats(const deque<StepDiagnostics>& recent_accepted_steps)
{
    AcceptedMotionStats stats;

    vector<double> step_t_values;
    vector<double> step_r_values;

    for (const auto& step : recent_accepted_steps)
    {
        if (step.accepted && step.step_t_norm > 0.0 && std::isfinite(step.step_t_norm))
        {
            step_t_values.push_back(step.step_t_norm);
        }

        if (step.accepted && step.step_r_deg > 0.0 && std::isfinite(step.step_r_deg))
        {
            step_r_values.push_back(step.step_r_deg);
        }
    }

    stats.median_step_t = median_of_vector(step_t_values, 0.02);
    stats.median_step_r_deg = median_of_vector(step_r_values, 1.0);

    stats.median_step_t = max(stats.median_step_t, 0.003);
    stats.median_step_r_deg = max(stats.median_step_r_deg, 0.15);

    return stats;
}

static void write_csv_header(ofstream& csv_file)
{
    csv_file
        << "prev_frame,curr_frame,accepted,reason,accepted_pose_source,recovery_mode,"
        << "raw_matches,good_matches,depth_valid_3d2d,pnp_inliers,inlier_ratio,depth_valid_ratio,"
        << "reproj_before,reproj_after,epi_before,epi_after,step_t_norm,step_r_deg,"
        << "consecutive_reject_count,last_accepted_frame,frames_since_last_accept,map_points_added_this_step"
        << "\n";
}

static void write_csv_row(ofstream& csv_file, const StepDiagnostics& diag)
{
    csv_file
        << diag.prev_frame_id << ","
        << diag.curr_frame_id << ","
        << (diag.accepted ? 1 : 0) << ","
        << diag.reason << ","
        << diag.accepted_pose_source << ","
        << (diag.recovery_mode ? 1 : 0) << ","
        << diag.raw_matches << ","
        << diag.good_matches << ","
        << diag.depth_valid_3d2d << ","
        << diag.pnp_inliers << ","
        << diag.inlier_ratio << ","
        << diag.depth_valid_ratio << ","
        << diag.reproj_before << ","
        << diag.reproj_after << ","
        << diag.epi_before << ","
        << diag.epi_after << ","
        << diag.step_t_norm << ","
        << diag.step_r_deg << ","
        << diag.consecutive_reject_count << ","
        << diag.last_accepted_frame << ","
        << diag.frames_since_last_accept << ","
        << diag.map_points_added_this_step
        << "\n";
}

static void write_summary_file(
    const string& summary_path,
    const vector<StepDiagnostics>& all_steps,
    int total_frames_loaded,
    int total_map_points)
{
    ofstream summary_file(summary_path);
    if (!summary_file.is_open())
    {
        cerr << "failed to open summary file for writing: " << summary_path << endl;
        return;
    }

    int accepted_count = 0;
    int rejected_count = 0;
    int recovery_accepted_count = 0;
    int max_consecutive_rejects = 0;

    for (const auto& step : all_steps)
    {
        if (step.accepted)
        {
            accepted_count++;
            if (step.recovery_mode)
            {
                recovery_accepted_count++;
            }
        }
        else
        {
            rejected_count++;
        }

        max_consecutive_rejects = max(max_consecutive_rejects, step.consecutive_reject_count);
    }

    summary_file << fixed << setprecision(4);
    summary_file << "frames_loaded: " << total_frames_loaded << "\n";
    summary_file << "rows: " << all_steps.size() << "\n";
    summary_file << "accepted: " << accepted_count << "\n";
    summary_file << "rejected: " << rejected_count << "\n";
    summary_file << "recovery_accepted: " << recovery_accepted_count << "\n";
    summary_file << "max_consecutive_rejects: " << max_consecutive_rejects << "\n";
    summary_file << "map_points: " << total_map_points << "\n";
}

static void write_jump_candidates_file(
    const string& jump_candidates_path,
    const vector<StepDiagnostics>& all_steps)
{
    ofstream jump_file(jump_candidates_path);
    if (!jump_file.is_open())
    {
        cerr << "failed to open jump candidates file for writing: " << jump_candidates_path << endl;
        return;
    }

    jump_file << fixed << setprecision(4);

    for (const auto& step : all_steps)
    {
        bool suspicious =
            (!step.accepted) ||
            (step.step_t_norm > 0.15) ||
            (step.step_r_deg > 10.0) ||
            (step.reproj_after > 2.5) ||
            (step.epi_after > 2.0) ||
            (step.frames_since_last_accept >= 8);

        if (!suspicious)
        {
            continue;
        }

        jump_file
            << "frame " << step.prev_frame_id << " -> " << step.curr_frame_id
            << " | accepted=" << (step.accepted ? 1 : 0)
            << " | reason=" << step.reason
            << " | pose_source=" << step.accepted_pose_source
            << " | recovery_mode=" << (step.recovery_mode ? 1 : 0)
            << " | good=" << step.good_matches
            << " | depth_valid=" << step.depth_valid_3d2d
            << " | depth_valid_ratio=" << step.depth_valid_ratio
            << " | inliers=" << step.pnp_inliers
            << " | inlier_ratio=" << step.inlier_ratio
            << " | reproj_after=" << step.reproj_after
            << " | epi_after=" << step.epi_after
            << " | step_t=" << step.step_t_norm
            << " | step_r_deg=" << step.step_r_deg
            << " | consecutive_rejects=" << step.consecutive_reject_count
            << " | last_accepted_frame=" << step.last_accepted_frame
            << " | frames_since_last_accept=" << step.frames_since_last_accept
            << " | map_points_added=" << step.map_points_added_this_step
            << "\n";
    }
}

static StepDiagnostics evaluate_step_diagnostics(
    int prev_frame_id,
    int curr_frame_id,
    const MatchInfo& match_info,
    const cv::Mat& rvec,
    const cv::Mat& tvec,
    const AcceptedMotionStats& recent_stats,
    int consecutive_rejects,
    int last_accepted_frame)
{
    StepDiagnostics diag;
    diag.prev_frame_id = prev_frame_id;
    diag.curr_frame_id = curr_frame_id;
    diag.raw_matches = static_cast<int>(match_info.raw_matches.size());
    diag.good_matches = static_cast<int>(match_info.good_matches.size());
    diag.depth_valid_3d2d = static_cast<int>(match_info.points_prev_3d_for_pnp.size());
    diag.pnp_inliers = static_cast<int>(match_info.pnp_inlier_matches.size());
    diag.reproj_before = match_info.reproj_before;
    diag.reproj_after = match_info.reproj_after;
    diag.epi_before = match_info.essential_epi_before;
    diag.epi_after = match_info.essential_epi_after;

    diag.consecutive_reject_count = consecutive_rejects;
    diag.last_accepted_frame = last_accepted_frame;
    diag.frames_since_last_accept = (last_accepted_frame >= 0) ? (curr_frame_id - last_accepted_frame) : -1;
    diag.recovery_mode = (consecutive_rejects >= 8);

    if (diag.depth_valid_3d2d > 0)
    {
        diag.inlier_ratio = static_cast<double>(diag.pnp_inliers) / static_cast<double>(diag.depth_valid_3d2d);
    }

    if (diag.good_matches > 0)
    {
        diag.depth_valid_ratio = static_cast<double>(diag.depth_valid_3d2d) / static_cast<double>(diag.good_matches);
    }

    diag.step_t_norm = cv::norm(tvec);
    diag.step_r_deg = compute_rotation_angle_deg_from_rvec(rvec);

    const int normal_min_inliers = 16;
    const int recovery_default_min_inliers = 13;

    const double normal_min_inlier_ratio = 0.72;
    const double recovery_min_inlier_ratio = 0.60;

    const double normal_max_reproj_after = 2.5;
    const double recovery_max_reproj_after = 1.8;

    const double max_epi_after = 2.5;

    bool very_stable_recovery =
        diag.recovery_mode &&
        diag.reproj_after >= 0.0 &&
        diag.reproj_after <= 1.2 &&
        diag.step_t_norm <= max(0.020, 2.0 * recent_stats.median_step_t) &&
        diag.step_r_deg <= max(2.0, 2.0 * recent_stats.median_step_r_deg + 0.5) &&
        diag.inlier_ratio >= 0.70;

    int recovery_min_inliers = very_stable_recovery ? 13 : recovery_default_min_inliers;

    if (diag.pnp_inliers < (diag.recovery_mode ? recovery_min_inliers : normal_min_inliers))
    {
        diag.accepted = false;
        diag.reason = "reject_low_inlier_count";
        return diag;
    }

    if (diag.inlier_ratio >= 0.0 &&
        diag.inlier_ratio < (diag.recovery_mode ? recovery_min_inlier_ratio : normal_min_inlier_ratio))
    {
        diag.accepted = false;
        diag.reason = "reject_low_inlier_ratio";
        return diag;
    }

    if (diag.reproj_after < 0.0 || !std::isfinite(diag.reproj_after))
    {
        diag.accepted = false;
        diag.reason = "reject_invalid_reproj_after";
        return diag;
    }

    if (diag.reproj_after > (diag.recovery_mode ? recovery_max_reproj_after : normal_max_reproj_after))
    {
        diag.accepted = false;
        diag.reason = "reject_high_reproj_after";
        return diag;
    }

    if (diag.epi_after > max_epi_after)
    {
        diag.accepted = false;
        diag.reason = "reject_high_epi_after";
        return diag;
    }

    if (diag.step_r_deg > 120.0)
    {
        diag.accepted = false;
        diag.reason = "reject_near_flip_rotation";
        return diag;
    }

    if (!diag.recovery_mode)
    {
        if (diag.step_r_deg > max(12.0, 8.0 * recent_stats.median_step_r_deg + 2.0))
        {
            diag.accepted = false;
            diag.reason = "reject_large_rotation_step";
            return diag;
        }

        if (diag.step_t_norm > max(0.20, 10.0 * recent_stats.median_step_t))
        {
            diag.accepted = false;
            diag.reason = "reject_large_translation_step";
            return diag;
        }
    }
    else
    {
        if (diag.step_r_deg > max(3.0, 3.0 * recent_stats.median_step_r_deg + 0.8))
        {
            diag.accepted = false;
            diag.reason = "reject_recovery_rotation";
            return diag;
        }

        if (diag.step_t_norm > max(0.035, 3.0 * recent_stats.median_step_t))
        {
            diag.accepted = false;
            diag.reason = "reject_recovery_translation";
            return diag;
        }
    }

    if (consecutive_rejects >= 3 && !diag.recovery_mode)
    {
        if (diag.step_r_deg > max(4.0, 4.0 * recent_stats.median_step_r_deg + 1.0))
        {
            diag.accepted = false;
            diag.reason = "reject_cooldown_rotation";
            return diag;
        }

        if (diag.step_t_norm > max(0.06, 4.5 * recent_stats.median_step_t))
        {
            diag.accepted = false;
            diag.reason = "reject_cooldown_translation";
            return diag;
        }
    }

    diag.accepted = true;
    diag.accepted_pose_source = diag.recovery_mode ? "pnp_recovery" : "pnp";
    diag.reason = "accepted";
    return diag;
}

int main(int argc, char** argv)
{
    string dataset_path;
    if (argc >= 2)
    {
        dataset_path = argv[1];
    }
    else
    {
        dataset_path = "./rgbd_dataset_freiburg2_pioneer_slam3";
    }

    string rgb_txt_path = dataset_path + "/rgb.txt";
    string depth_txt_path = dataset_path + "/depth.txt";

    if (!path_exists(rgb_txt_path))
    {
        cerr << "could not find rgb.txt at: " << rgb_txt_path << endl;
        return 1;
    }

    if (!path_exists(depth_txt_path))
    {
        cerr << "could not find depth.txt at: " << depth_txt_path << endl;
        return 1;
    }

    vector<pair<double, string>> rgb_entries = load_timestamp_file(rgb_txt_path);
    vector<pair<double, string>> depth_entries = load_timestamp_file(depth_txt_path);

    if (rgb_entries.empty() || depth_entries.empty())
    {
        cerr << "rgb or depth entries missing" << endl;
        return 1;
    }

    const double max_rgb_depth_time_diff_sec = 0.03;

    MapData map_data;

    for (size_t i = 0; i < rgb_entries.size(); ++i)
    {
        int depth_index = find_closest_depth_index(
            rgb_entries[i].first,
            depth_entries,
            max_rgb_depth_time_diff_sec
        );

        if (depth_index < 0)
        {
            continue;
        }

        Frame frame;
        frame.id = static_cast<int>(map_data.frames.size());
        frame.rgb_timestamp = rgb_entries[i].first;
        frame.depth_timestamp = depth_entries[depth_index].first;
        frame.rgb_path = dataset_path + "/" + rgb_entries[i].second;
        frame.depth_path = dataset_path + "/" + depth_entries[depth_index].second;

        frame.image_bgr = cv::imread(frame.rgb_path, cv::IMREAD_COLOR);
        frame.image_gray = cv::imread(frame.rgb_path, cv::IMREAD_GRAYSCALE);
        frame.depth_image = cv::imread(frame.depth_path, cv::IMREAD_UNCHANGED);

        if (frame.image_gray.empty() || frame.image_bgr.empty() || frame.depth_image.empty())
        {
            cerr << "failed loading rgb/depth for frame " << frame.id << endl;
            continue;
        }

        map_data.frames.push_back(frame);
    }

    if (map_data.frames.size() < 2)
    {
        cerr << "not enough valid associated rgb-depth frames" << endl;
        return 1;
    }

    int image_width = map_data.frames[0].image_gray.cols;
    int image_height = map_data.frames[0].image_gray.rows;

    double fx = 520.9;
    double fy = 521.0;
    double cx = 325.1;
    double cy = 249.7;

    cv::Mat camera_matrix = (cv::Mat_<double>(3,3) <<
        fx, 0.0, cx,
        0.0, fy, cy,
        0.0, 0.0, 1.0
    );

    double depth_scale = 5000.0;

    cout << fixed << setprecision(4);
    cout << "dataset loaded: " << map_data.frames.size() << " associated rgb-depth frames" << endl;
    cout << "camera matrix:" << endl << camera_matrix << endl;
    cout << "depth scale: " << depth_scale << endl;
    cout << "image size: " << image_width << "x" << image_height << endl;

    cv::Ptr<cv::ORB> orb = cv::ORB::create(
        3000,
        1.2f,
        8,
        31,
        0,
        2,
        cv::ORB::HARRIS_SCORE,
        31,
        20
    );

    pangolin::CreateWindowAndBind("trajectory_and_map", 1280, 720);
    glEnable(GL_DEPTH_TEST);

    pangolin::OpenGlRenderState camera_render_state(
        pangolin::ProjectionMatrix(1280, 720, 500, 500, 640, 360, 0.1, 1000.0),
        pangolin::ModelViewLookAt(-2.0, -2.0, -2.0, 0.0, 0.0, 0.0, pangolin::AxisY)
    );

    pangolin::View& display = pangolin::CreateDisplay()
        .SetBounds(0.0, 1.0, 0.0, 1.0, -1280.0f / 720.0f)
        .SetHandler(new pangolin::Handler3D(camera_render_state));

    cv::namedWindow("current_frame", cv::WINDOW_NORMAL);
    cv::namedWindow("matches_before_filtering", cv::WINDOW_NORMAL);
    cv::namedWindow("matches_after_pnp", cv::WINDOW_NORMAL);

    if (!extract_features(map_data.frames[0], orb))
    {
        cerr << "feature extraction failed on first frame" << endl;
        return 1;
    }

    int next_map_point_id = 0;
    int initial_points_added = add_current_frame_depth_points_to_map(
        map_data.frames[0],
        camera_matrix,
        depth_scale,
        14,
        next_map_point_id,
        map_data
    );

    vector<StepDiagnostics> all_step_diagnostics;
    deque<StepDiagnostics> recent_accepted_steps;
    int consecutive_rejects = 0;
    int last_accepted_frame = 0;
    const size_t recent_window_size = 25;

    string diagnostics_csv_path = "slam_diagnostics.csv";
    string jump_candidates_path = "slam_jump_candidates.txt";
    string summary_path = "slam_summary.txt";

    ofstream diagnostics_csv_file(diagnostics_csv_path);
    if (!diagnostics_csv_file.is_open())
    {
        cerr << "failed to open diagnostics csv for writing: " << diagnostics_csv_path << endl;
        return 1;
    }
    write_csv_header(diagnostics_csv_file);

    cout << "initial map points added: " << initial_points_added << endl;

    for (size_t frame_index = 1; frame_index < map_data.frames.size(); ++frame_index)
    {
        if (pangolin::ShouldQuit())
        {
            break;
        }

        Frame& frame_prev = map_data.frames[frame_index - 1];
        Frame& frame_curr = map_data.frames[frame_index];

        if (!frame_prev.processed)
        {
            if (!extract_features(frame_prev, orb))
            {
                cerr << "feature extraction failed for prev frame " << frame_prev.id << endl;
                continue;
            }
        }

        if (!extract_features(frame_curr, orb))
        {
            cerr << "feature extraction failed for curr frame " << frame_curr.id << endl;
            continue;
        }

        MatchInfo match_info = match_features(frame_prev, frame_curr);

        if (match_info.good_matches.size() < 12)
        {
            StepDiagnostics diag;
            diag.prev_frame_id = frame_prev.id;
            diag.curr_frame_id = frame_curr.id;
            diag.raw_matches = static_cast<int>(match_info.raw_matches.size());
            diag.good_matches = static_cast<int>(match_info.good_matches.size());
            diag.consecutive_reject_count = consecutive_rejects;
            diag.last_accepted_frame = last_accepted_frame;
            diag.frames_since_last_accept = (last_accepted_frame >= 0) ? (frame_curr.id - last_accepted_frame) : -1;
            diag.accepted = false;
            diag.reason = "reject_low_good_match_count";

            all_step_diagnostics.push_back(diag);
            write_csv_row(diagnostics_csv_file, diag);
            consecutive_rejects++;

            cerr << "not enough good matches for frame pair "
                 << frame_prev.id << " -> " << frame_curr.id << endl;
            continue;
        }

        cv::Mat F_before = cv::findFundamentalMat(
            match_info.points_prev_2d,
            match_info.points_curr_2d,
            cv::FM_RANSAC,
            1.0,
            0.99
        );
        match_info.essential_epi_before = compute_mean_epipolar_error(
            match_info.points_prev_2d,
            match_info.points_curr_2d,
            F_before
        );

        build_pnp_correspondences_from_prev_depth(
            frame_prev,
            frame_curr,
            camera_matrix,
            depth_scale,
            match_info
        );

        double depth_valid_ratio =
            (match_info.good_matches.empty())
            ? -1.0
            : static_cast<double>(match_info.points_prev_3d_for_pnp.size()) /
              static_cast<double>(match_info.good_matches.size());

        if (match_info.points_prev_3d_for_pnp.size() < 10)
        {
            StepDiagnostics diag;
            diag.prev_frame_id = frame_prev.id;
            diag.curr_frame_id = frame_curr.id;
            diag.raw_matches = static_cast<int>(match_info.raw_matches.size());
            diag.good_matches = static_cast<int>(match_info.good_matches.size());
            diag.depth_valid_3d2d = static_cast<int>(match_info.points_prev_3d_for_pnp.size());
            diag.depth_valid_ratio = depth_valid_ratio;
            diag.consecutive_reject_count = consecutive_rejects;
            diag.last_accepted_frame = last_accepted_frame;
            diag.frames_since_last_accept = (last_accepted_frame >= 0) ? (frame_curr.id - last_accepted_frame) : -1;
            diag.accepted = false;

            if (diag.consecutive_reject_count >= 20 &&
                diag.good_matches >= 80 &&
                diag.depth_valid_3d2d <= 2 &&
                diag.depth_valid_ratio >= 0.0 &&
                diag.depth_valid_ratio < 0.03)
            {
                diag.reason = "tracking_lost_depth_desert";
            }
            else if (diag.good_matches >= 80 &&
                     diag.depth_valid_ratio >= 0.0 &&
                     diag.depth_valid_ratio < 0.08)
            {
                diag.reason = "reject_depth_valid_ratio_too_low";
            }
            else
            {
                diag.reason = "reject_low_depth_valid_count";
            }

            all_step_diagnostics.push_back(diag);
            write_csv_row(diagnostics_csv_file, diag);
            consecutive_rejects++;

            cerr << "not enough valid 3D-2D correspondences for frame pair "
                 << frame_prev.id << " -> " << frame_curr.id
                 << " | depth-valid: " << match_info.points_prev_3d_for_pnp.size()
                 << " | depth_valid_ratio: " << depth_valid_ratio
                 << endl;
            continue;
        }

        cv::Mat rvec, tvec;
        bool pose_success = estimate_pose_pnp(match_info, camera_matrix, rvec, tvec);

        if (!pose_success)
        {
            StepDiagnostics diag;
            diag.prev_frame_id = frame_prev.id;
            diag.curr_frame_id = frame_curr.id;
            diag.raw_matches = static_cast<int>(match_info.raw_matches.size());
            diag.good_matches = static_cast<int>(match_info.good_matches.size());
            diag.depth_valid_3d2d = static_cast<int>(match_info.points_prev_3d_for_pnp.size());
            diag.depth_valid_ratio = depth_valid_ratio;
            diag.consecutive_reject_count = consecutive_rejects;
            diag.last_accepted_frame = last_accepted_frame;
            diag.frames_since_last_accept = (last_accepted_frame >= 0) ? (frame_curr.id - last_accepted_frame) : -1;
            diag.accepted = false;
            diag.reason = "reject_pnp_failed";

            all_step_diagnostics.push_back(diag);
            write_csv_row(diagnostics_csv_file, diag);
            consecutive_rejects++;

            cerr << "PnP pose estimation failed for frame pair "
                 << frame_prev.id << " -> " << frame_curr.id
                 << " | depth-valid: " << match_info.points_prev_3d_for_pnp.size()
                 << endl;
            continue;
        }

        vector<cv::Point2f> inlier_prev_2d_for_F;
        vector<cv::Point2f> inlier_curr_2d_for_F;
        for (int idx_in_pnp_set : match_info.pnp_inlier_indices)
        {
            if (idx_in_pnp_set < 0 || idx_in_pnp_set >= static_cast<int>(match_info.points_curr_2d_for_pnp.size()))
            {
                continue;
            }

            int idx_in_good = match_info.good_match_indices_used_for_pnp[idx_in_pnp_set];
            if (idx_in_good < 0 || idx_in_good >= static_cast<int>(match_info.points_prev_2d.size()))
            {
                continue;
            }

            inlier_prev_2d_for_F.push_back(match_info.points_prev_2d[idx_in_good]);
            inlier_curr_2d_for_F.push_back(match_info.points_curr_2d[idx_in_good]);
        }

        if (inlier_prev_2d_for_F.size() >= 8)
        {
            cv::Mat F_after = cv::findFundamentalMat(
                inlier_prev_2d_for_F,
                inlier_curr_2d_for_F,
                cv::FM_RANSAC,
                1.0,
                0.99
            );
            match_info.essential_epi_after = compute_mean_epipolar_error(
                inlier_prev_2d_for_F,
                inlier_curr_2d_for_F,
                F_after
            );
        }

        AcceptedMotionStats recent_stats = compute_recent_motion_stats(recent_accepted_steps);

        StepDiagnostics diag = evaluate_step_diagnostics(
            frame_prev.id,
            frame_curr.id,
            match_info,
            rvec,
            tvec,
            recent_stats,
            consecutive_rejects,
            last_accepted_frame
        );

        int map_points_added_this_step = 0;

        if (diag.accepted)
        {
            update_frame_pose_from_pnp(frame_curr, frame_prev, rvec, tvec);

            map_points_added_this_step = add_current_frame_depth_points_to_map(
                frame_curr,
                camera_matrix,
                depth_scale,
                14,
                next_map_point_id,
                map_data
            );

            diag.map_points_added_this_step = map_points_added_this_step;

            recent_accepted_steps.push_back(diag);
            if (recent_accepted_steps.size() > recent_window_size)
            {
                recent_accepted_steps.pop_front();
            }

            last_accepted_frame = frame_curr.id;
            consecutive_rejects = 0;
        }
        else
        {
            frame_curr.pose = frame_prev.pose;
            diag.map_points_added_this_step = 0;
            consecutive_rejects++;
        }

        all_step_diagnostics.push_back(diag);
        write_csv_row(diagnostics_csv_file, diag);

        string keypoints_text =
            "frame: " + to_string(frame_curr.id) +
            " keypoints: " + to_string(frame_curr.keypoints.size()) +
            " status: " + (diag.accepted ? "ACCEPTED" : "REJECTED");

        string before_text =
            "raw: " + to_string(match_info.raw_matches.size()) +
            " good: " + to_string(match_info.good_matches.size()) +
            " epi_before: " + to_string(match_info.essential_epi_before);

        string after_text =
            "3D-2D: " + to_string(match_info.points_prev_3d_for_pnp.size()) +
            " pnp_inliers: " + to_string(match_info.pnp_inlier_matches.size()) +
            " reproj_after: " + to_string(match_info.reproj_after) +
            " step_t: " + to_string(diag.step_t_norm);

        cv::Mat keypoints_image = draw_keypoints_image(frame_curr, keypoints_text, diag.accepted);
        cv::Mat matches_before_image = draw_matches_image(frame_prev, frame_curr, match_info.good_matches, before_text);
        cv::Mat matches_after_image = draw_matches_image(frame_prev, frame_curr, match_info.pnp_inlier_matches, after_text);

        if (!diag.accepted)
        {
            cv::putText(
                keypoints_image,
                "REASON: " + diag.reason,
                cv::Point(20, 70),
                cv::FONT_HERSHEY_SIMPLEX,
                0.7,
                cv::Scalar(0, 0, 255),
                2
            );
        }
        else if (diag.recovery_mode)
        {
            cv::putText(
                keypoints_image,
                "RECOVERY ACCEPT",
                cv::Point(20, 70),
                cv::FONT_HERSHEY_SIMPLEX,
                0.7,
                cv::Scalar(0, 255, 255),
                2
            );
        }

        cv::imshow("current_frame", keypoints_image);
        cv::imshow("matches_before_filtering", matches_before_image);
        cv::imshow("matches_after_pnp", matches_after_image);

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        display.Activate(camera_render_state);
        draw_trajectory_and_map(map_data);
        pangolin::FinishFrame();

        char key = static_cast<char>(cv::waitKey(1));
        if (key == 27 || key == 'q')
        {
            break;
        }

        cout
            << "frame " << frame_prev.id << " -> " << frame_curr.id
            << " | accepted: " << (diag.accepted ? 1 : 0)
            << " | reason: " << diag.reason
            << " | pose_source: " << diag.accepted_pose_source
            << " | recovery_mode: " << (diag.recovery_mode ? 1 : 0)
            << " | raw: " << diag.raw_matches
            << " | good: " << diag.good_matches
            << " | depth_valid_3d2d: " << diag.depth_valid_3d2d
            << " | depth_valid_ratio: " << diag.depth_valid_ratio
            << " | pnp_inliers: " << diag.pnp_inliers
            << " | inlier_ratio: " << diag.inlier_ratio
            << " | reproj_before: " << diag.reproj_before
            << " | reproj_after: " << diag.reproj_after
            << " | epi_before: " << diag.epi_before
            << " | epi_after: " << diag.epi_after
            << " | step_t_norm: " << diag.step_t_norm
            << " | step_r_deg: " << diag.step_r_deg
            << " | consecutive_rejects: " << diag.consecutive_reject_count
            << " | last_accepted_frame: " << diag.last_accepted_frame
            << " | frames_since_last_accept: " << diag.frames_since_last_accept
            << " | map_points_added: " << diag.map_points_added_this_step
            << " | map_points_total: " << map_data.points.size()
            << endl;
    }

    diagnostics_csv_file.close();

    write_jump_candidates_file(jump_candidates_path, all_step_diagnostics);
    write_summary_file(summary_path, all_step_diagnostics, static_cast<int>(map_data.frames.size()), static_cast<int>(map_data.points.size()));

    cout << "finished" << endl;
    cout << "wrote: " << diagnostics_csv_path << endl;
    cout << "wrote: " << jump_candidates_path << endl;
    cout << "wrote: " << summary_path << endl;

    return 0;
}