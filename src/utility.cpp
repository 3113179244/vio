#include "utility.h"
#include <cmath>

namespace utility
{

    Eigen::Matrix3d skewSymmetric(const Eigen::Vector3d &vector_3d)
    {
        Eigen::Matrix3d skew_matrix;
        skew_matrix << 0.0, -vector_3d.z(), vector_3d.y(),
            vector_3d.z(), 0.0, -vector_3d.x(),
            -vector_3d.y(), vector_3d.x(), 0.0;
        return skew_matrix;
    }

    Eigen::Matrix3d rightJacobianSO3(const Eigen::Vector3d &phi_rot_vector)
    {
        double rotation_angle = phi_rot_vector.norm();

        // 如果旋转角极小，使用泰勒展开的一阶近似，避免分母除以 0 导致数值发散
        if (rotation_angle < 1e-6)
        {
            return Eigen::Matrix3d::Identity() - 0.5 * skewSymmetric(phi_rot_vector);
        }

        Eigen::Vector3d rotation_axis = phi_rot_vector.normalized();
        Eigen::Matrix3d skew_axis = skewSymmetric(rotation_axis);

        // 闭式解公式: Jr = I - ((1 - cos(theta)) / theta) * [n]^x + ((1 - sin(theta)/theta)) * [n]^x^2
        return Eigen::Matrix3d::Identity() - ((1.0 - std::cos(rotation_angle)) / rotation_angle) * skew_axis + (1.0 - std::sin(rotation_angle) / rotation_angle) * skew_axis * skew_axis;
    }

    Eigen::Matrix4d quaternionLeftProdMat(const Eigen::Quaterniond &q_left)
    {
        Eigen::Matrix4d left_matrix;
        left_matrix.block<1, 1>(0, 0) << q_left.w();
        left_matrix.block<1, 3>(0, 1) = -q_left.vec().transpose();
        left_matrix.block<3, 1>(1, 0) = q_left.vec();
        left_matrix.block<3, 3>(1, 1) = q_left.w() * Eigen::Matrix3d::Identity() + skewSymmetric(q_left.vec());
        return left_matrix;
    }

    Eigen::Matrix4d quaternionRightProdMat(const Eigen::Quaterniond &q_right)
    {
        Eigen::Matrix4d right_matrix;
        right_matrix.block<1, 1>(0, 0) << q_right.w();
        right_matrix.block<1, 3>(0, 1) = -q_right.vec().transpose();
        right_matrix.block<3, 1>(1, 0) = q_right.vec();
        right_matrix.block<3, 3>(1, 1) = q_right.w() * Eigen::Matrix3d::Identity() - skewSymmetric(q_right.vec());
        return right_matrix;
    }

    Eigen::Vector3d quaternionLogMap(const Eigen::Quaterniond &q_error)
    {
        // 在 VIO 优化和 Ceres 切空间姿态残差定义中，误差四元数的虚部通常近似代表 1/2 的李代数旋转误差
        // 这里乘 2 还原为对应的 3 维误差状态变量
        return 2.0 * q_error.vec();
    }

    Eigen::Quaterniond quaternionExpMap(const Eigen::Vector3d &phi_rot_vector)
    {
        double rotation_angle = phi_rot_vector.norm();

        // 针对零近邻的泰勒展开保护
        if (rotation_angle < 1e-6)
        {
            return Eigen::Quaterniond(1.0, 0.5 * phi_rot_vector.x(), 0.5 * phi_rot_vector.y(), 0.5 * phi_rot_vector.z());
        }

        Eigen::Vector3d rotation_axis = phi_rot_vector.normalized();
        double sin_half_angle = std::sin(rotation_angle * 0.5);
        double cos_half_angle = std::cos(rotation_angle * 0.5);

        return Eigen::Quaterniond(cos_half_angle,
                                  rotation_axis.x() * sin_half_angle,
                                  rotation_axis.y() * sin_half_angle,
                                  rotation_axis.z() * sin_half_angle);
    }

} // namespace utility