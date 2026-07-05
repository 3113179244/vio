#ifndef PROJECTION_FACTOR_H
#define PROJECTION_FACTOR_H

#include <ceres/ceres.h>
#include <Eigen/Dense>

// 改用 Ceres 自动求导 Functor 结构
class ProjectionInverseDepthFactor
{
public:
    ProjectionInverseDepthFactor(const Eigen::Vector2d &_pts_i, const Eigen::Vector2d &_pts_j)
        : pts_i(_pts_i), pts_j(_pts_j) {}

    // 使用重载的 operator() 模板函数，Ceres 会利用仿函数进行自动求导，彻底根治 Uninitialized 错误
    template <typename T>
    bool operator()(const T *const logo_pose_i, const T *const logo_pose_j,
                    const T *const ex_pose, const T *const inverse_depth,
                    T *residuals) const
    {
        // 1. 解析参考帧 i 和当前帧 j 的 IMU 位姿 (包含位移 P 和四元数 Q)
        Eigen::Matrix<T, 3, 1> Pi(logo_pose_i[0], logo_pose_i[1], logo_pose_i[2]);
        Eigen::Quaternion<T> Qi(logo_pose_i[6], logo_pose_i[3], logo_pose_i[4], logo_pose_i[5]);

        Eigen::Matrix<T, 3, 1> Pj(logo_pose_j[0], logo_pose_j[1], logo_pose_j[2]);
        Eigen::Quaternion<T> Qj(logo_pose_j[6], logo_pose_j[3], logo_pose_j[4], logo_pose_j[5]);

        // 2. 解析相机与 IMU 之间的外参 (tic, qic)
        Eigen::Matrix<T, 3, 1> tic(ex_pose[0], ex_pose[1], ex_pose[2]);
        Eigen::Quaternion<T> qic(ex_pose[6], ex_pose[3], ex_pose[4], ex_pose[5]);

        // 3. 获取参考帧中该特征点的逆深度
        T inv_dep = inverse_depth[0];

        // 4. 【关键步骤】空间坐标系变换
        Eigen::Matrix<T, 3, 1> pts_camera_i;
        pts_camera_i << T(pts_i.x()), T(pts_i.y()), T(1.0);
        pts_camera_i = pts_camera_i / inv_dep; // 逆投影：归一化坐标 -> 参考帧相机3D点

        Eigen::Matrix<T, 3, 1> pts_imu_i = qic * pts_camera_i + tic;             // 相机 i -> IMU i
        Eigen::Matrix<T, 3, 1> pts_w = Qi * pts_imu_i + Pi;                      // IMU i  -> 世界坐标系
        Eigen::Matrix<T, 3, 1> pts_imu_j = Qj.inverse() * (pts_w - Pj);          // 世界   -> IMU j
        Eigen::Matrix<T, 3, 1> pts_camera_j = qic.inverse() * (pts_imu_j - tic); // IMU j -> 相机 j

        // 5. 投影到当前帧归一化平面并计算残差
        T dep_j = pts_camera_j.z();

        // 数值安全保护，防止深度为0导致除以0崩溃
        if (dep_j < T(1e-5))
        {
            dep_j = T(1e-5);
        }

        // 残差 = 预测的归一化平面坐标 - 实际观测到的归一化平面坐标
        residuals[0] = pts_camera_j.x() / dep_j - T(pts_j.x());
        residuals[1] = pts_camera_j.y() / dep_j - T(pts_j.y());

        return true;
    }

    // 静态工厂创建函数：方便外部快捷调用
    static ceres::CostFunction *Create(const Eigen::Vector2d &_pts_i, const Eigen::Vector2d &_pts_j)
    {
        // 自动求导参数分配：残差维度(2)，四个参数块的维度分别是位姿i(7)、位姿j(7)、外参(7)、逆深度(1)
        return (new ceres::AutoDiffCostFunction<ProjectionInverseDepthFactor, 2, 7, 7, 7, 1>(
            new ProjectionInverseDepthFactor(_pts_i, _pts_j)));
    }

    Eigen::Vector2d pts_i, pts_j;
};

#endif // PROJECTION_FACTOR_H