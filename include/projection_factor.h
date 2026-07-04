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
        // 1. 解析参考帧 i 的位姿
        Eigen::Matrix<T, 3, 1> Pi(logo_pose_i[0], logo_pose_i[1], logo_pose_i[2]);
        Eigen::Quaternion<T> Qi(logo_pose_i[6], logo_pose_i[3], logo_pose_i[4], logo_pose_i[5]);

        // 2. 解析当前帧 j 的位姿
        Eigen::Matrix<T, 3, 1> Pj(logo_pose_j[0], logo_pose_j[1], logo_pose_j[2]);
        Eigen::Quaternion<T> Qj(logo_pose_j[6], logo_pose_j[3], logo_pose_j[4], logo_pose_j[5]);

        // 3. 解析外参
        Eigen::Matrix<T, 3, 1> tic(ex_pose[0], ex_pose[1], ex_pose[2]);
        Eigen::Quaternion<T> qic(ex_pose[6], ex_pose[3], ex_pose[4], ex_pose[5]);

        T inv_dep = inverse_depth[0];

        // 4. 三维点从 Camera_i 逆投影变换至世界坐标系，再投影到 Camera_j
        Eigen::Matrix<T, 3, 1> pts_camera_i;
        pts_camera_i << T(pts_i.x()), T(pts_i.y()), T(1.0);
        pts_camera_i = pts_camera_i / inv_dep;

        Eigen::Matrix<T, 3, 1> pts_imu_i = qic * pts_camera_i + tic;
        Eigen::Matrix<T, 3, 1> pts_w = Qi * pts_imu_i + Pi;
        Eigen::Matrix<T, 3, 1> pts_imu_j = Qj.inverse() * (pts_w - Pj);
        Eigen::Matrix<T, 3, 1> pts_camera_j = qic.inverse() * (pts_imu_j - tic);

        // 5. 计算残差
        T dep_j = pts_camera_j.z();
        
        // 增加数值安全保护，防止位姿退化重叠时除以 0 导致数值发散
        if (dep_j < T(1e-5)) {
            dep_j = T(1e-5);
        }

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