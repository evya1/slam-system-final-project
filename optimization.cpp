#include "optimization.h"

#include <Eigen/Dense>
#include <cmath>

// Helpers

// Skew-symmetric matrix [v]_×
static Eigen::Matrix3d skew(const Eigen::Vector3d& v) {
    Eigen::Matrix3d S;
    S <<  0.0,   -v.z(),  v.y(),
          v.z(),  0.0,   -v.x(),
         -v.y(),  v.x(),  0.0;
    return S;
}

// 2×6 Jacobian of the reprojection residual w.r.t. left perturbation δξ of T_cw.
//
// x_c — 3D point in camera frame (must have z > 0).
// J   = J_π * [ -[x_c]_×  |  I_3 ]
static Eigen::Matrix<double, 2, 6> reproj_jacobian(
    const Eigen::Vector3d& x_c,
    double fx, double fy)
{
    const double xc = x_c.x(), yc = x_c.y(), zc = x_c.z();
    const double inv_z  = 1.0 / zc;
    const double inv_z2 = inv_z * inv_z;

    // J_π  (2×3): Jacobian of pixel coords w.r.t. camera-frame point
    Eigen::Matrix<double, 2, 3> J_pi;
    J_pi << fx * inv_z,  0.0,         -fx * xc * inv_z2,
            0.0,          fy * inv_z,  -fy * yc * inv_z2;

    // ∂x_c / ∂[δφ, δρ] = [ -[x_c]_×  |  I_3 ]  (3×6)
    Eigen::Matrix<double, 3, 6> dx_dxi;
    dx_dxi.leftCols<3>()  = -skew(x_c);
    dx_dxi.rightCols<3>() =  Eigen::Matrix3d::Identity();

    return J_pi * dx_dxi;
}

// Sum of squared reprojection errors given T_cw = (R, t).
static double sum_squared_reproj(
    const Eigen::Matrix3d& R_cw,
    const Eigen::Vector3d& t_cw,
    const std::vector<Eigen::Vector3d>& pts_w,
    const std::vector<cv::Point2f>&     pts2d,
    double fx, double fy, double cx, double cy)
{
    double total = 0.0;
    const int n  = static_cast<int>(pts_w.size());
    for (int i = 0; i < n; ++i) {
        Eigen::Vector3d x_c = R_cw * pts_w[i] + t_cw;
        if (x_c.z() <= 1e-6) { total += 1e6; continue; }
        double u = fx * x_c.x() / x_c.z() + cx;
        double v = fy * x_c.y() / x_c.z() + cy;
        double du = u - pts2d[i].x;
        double dv = v - pts2d[i].y;
        total += du * du + dv * dv;
    }
    return total;
}

// Public API

double compute_reproj_error(
    const Pose&                     pose,
    const std::vector<cv::Point3f>& pts3d,
    const std::vector<cv::Point2f>& pts2d,
    const cv::Mat&                  K)
{
    const int n = static_cast<int>(pts3d.size());
    if (n == 0 || pts3d.size() != pts2d.size()) return -1.0;

    cv::Mat K64;
    K.convertTo(K64, CV_64F);
    const double fx = K64.at<double>(0, 0), fy = K64.at<double>(1, 1);
    const double cx = K64.at<double>(0, 2), cy = K64.at<double>(1, 2);

    Eigen::Matrix3d R_cw = pose.R_cw();
    Eigen::Vector3d t_cw = pose.t_cw();

    double total = 0.0;
    for (int i = 0; i < n; ++i) {
        Eigen::Vector3d x_w(pts3d[i].x, pts3d[i].y, pts3d[i].z);
        Eigen::Vector3d x_c = R_cw * x_w + t_cw;
        if (x_c.z() <= 1e-6) continue;
        double u = fx * x_c.x() / x_c.z() + cx;
        double v = fy * x_c.y() / x_c.z() + cy;
        double du = u - pts2d[i].x;
        double dv = v - pts2d[i].y;
        total += du * du + dv * dv;
    }
    return std::sqrt(total / n);
}

OptimResult optimize_pose_lm(
    const Pose&                     initial_pose,
    const std::vector<cv::Point3f>& pts3d,
    const std::vector<cv::Point2f>& pts2d,
    const cv::Mat&                  K,
    int    max_iter,
    double lambda_init,
    double converge_eps)
{
    OptimResult result;
    result.optimized_pose = initial_pose;

    const int n = static_cast<int>(pts3d.size());
    if (n < 4 || pts3d.size() != pts2d.size()) {
        return result;
    }

    // Extract intrinsics
    cv::Mat K64;
    K.convertTo(K64, CV_64F);
    const double fx = K64.at<double>(0, 0), fy = K64.at<double>(1, 1);
    const double cx = K64.at<double>(0, 2), cy = K64.at<double>(1, 2);

    // Convert pts3d to Eigen world-frame vectors (avoids repeated conversion)
    std::vector<Eigen::Vector3d> pts_w(n);
    for (int i = 0; i < n; ++i)
        pts_w[i] = {pts3d[i].x, pts3d[i].y, pts3d[i].z};

    // Work in T_cw space
    Eigen::Matrix3d R_cw = initial_pose.R_cw();
    Eigen::Vector3d t_cw = initial_pose.t_cw();

    result.reproj_before = std::sqrt(
        sum_squared_reproj(R_cw, t_cw, pts_w, pts2d, fx, fy, cx, cy) / n);

    double lambda    = lambda_init;
    double prev_cost = sum_squared_reproj(R_cw, t_cw, pts_w, pts2d, fx, fy, cx, cy);

    for (int iter = 0; iter < max_iter; ++iter) {
        // Build Hessian approximation H = Σ Jᵀ J  and gradient g = Σ Jᵀ r
        Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Zero();
        Eigen::Matrix<double, 6, 1> g = Eigen::Matrix<double, 6, 1>::Zero();

        for (int i = 0; i < n; ++i) {
            Eigen::Vector3d x_c = R_cw * pts_w[i] + t_cw;
            if (x_c.z() <= 1e-6) continue;

            const double u = fx * x_c.x() / x_c.z() + cx;
            const double v = fy * x_c.y() / x_c.z() + cy;

            // Residual: projected - observed (matches J and normal equations)
            const Eigen::Vector2d r(u - pts2d[i].x, v - pts2d[i].y);

            const auto J = reproj_jacobian(x_c, fx, fy);
            H += J.transpose() * J;
            g += J.transpose() * r;
        }

        // LM damping: scale diagonal by (1 + λ)
        Eigen::Matrix<double, 6, 6> H_lm = H;
        for (int k = 0; k < 6; ++k)
            H_lm(k, k) *= (1.0 + lambda);

        // Solve: (H + λ diag(H)) δξ = -g
        const Eigen::Matrix<double, 6, 1> delta = H_lm.ldlt().solve(-g);

        if (!delta.allFinite()) break;

        // Trial update ——————————————————————————————————————————————————————
        const Eigen::Vector3d delta_phi = delta.head<3>();
        const Eigen::Vector3d delta_rho = delta.tail<3>();

        // Rotation update via Rodrigues
        const double angle = delta_phi.norm();
        Eigen::Matrix3d delta_R;
        if (angle < 1e-9) {
            // First-order approximation for tiny angles
            delta_R = Eigen::Matrix3d::Identity() + skew(delta_phi);
        } else {
            delta_R = Eigen::AngleAxisd(angle, delta_phi / angle)
                          .toRotationMatrix();
        }

        Eigen::Matrix3d R_new = delta_R * R_cw;
        Eigen::Vector3d t_new = delta_R * t_cw + delta_rho;

        const double new_cost =
            sum_squared_reproj(R_new, t_new, pts_w, pts2d, fx, fy, cx, cy);

        if (new_cost < prev_cost) {
            // Accept step
            R_cw      = R_new;
            t_cw      = t_new;
            prev_cost = new_cost;
            lambda   *= 0.1;
            if (lambda < 1e-10) lambda = 1e-10;

            result.iterations = iter + 1;
            if (delta.norm() < converge_eps) {
                result.converged = true;
                break;
            }
        } else {
            // Reject step — increase damping
            lambda *= 10.0;
            if (lambda > 1e8) {
                result.iterations = iter + 1;
                break;
            }
        }
    }
    if (result.iterations == 0) result.iterations = max_iter;

    // Convert optimised T_cw back to stored T_wc
    const Eigen::Matrix3d R_wc = R_cw.transpose();
    const Eigen::Vector3d t_wc = -(R_wc * t_cw);
    result.optimized_pose = Pose(R_wc, t_wc);

    result.reproj_after = std::sqrt(
        sum_squared_reproj(R_cw, t_cw, pts_w, pts2d, fx, fy, cx, cy) / n);

    return result;
}
