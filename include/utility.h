#ifndef UTILITY_H
#define UTILITY_H

#include <Eigen/Dense>

namespace utility
{

    /**
     * @brief 计算三维向量的反对称矩阵 (Skew-symmetric matrix / Hat operator)
     * 将向量变为什么形式的矩阵，以便将叉乘 (a x b) 改写为矩阵乘法 (A * b)
     * @param vector_3d 输入的 3 维向量 \mathbf{\omega} = [x, y, z]^T
     * @return 3x3 的反对称矩阵 [\mathbf{\omega}]_\times
     */
    Eigen::Matrix3d skewSymmetric(const Eigen::Vector3d &vector_3d);

    /**
     * @brief 计算三维旋转李代数 SO(3) 的右雅可比矩阵 (Right Jacobian of SO(3))
     * 常用于对旋转施加李代数扰动时，将残差在切空间进行一阶泰勒展开
     * @param phi_rot_vector 3 维旋转向量（李代数 so(3) 元素），方向为旋转轴，模长为旋转角（弧度）
     * @return 3x3 的右雅可比矩阵 J_r(\mathbf{\phi})
     */
    Eigen::Matrix3d rightJacobianSO3(const Eigen::Vector3d &phi_rot_vector);

    /**
     * @brief 计算四元数左乘变换矩阵 (Left quaternion multiplication matrix)
     * 用于将四元数乘法 q_a * q_b 转化为矩阵与向量的乘法：[q_a]_L * q_b_vector
     * @param q_left 被左乘的四元数
     * @return 4x4 的左乘矩阵
     */
    Eigen::Matrix4d quaternionLeftProdMat(const Eigen::Quaterniond &q_left);

    /**
     * @brief 计算四元数右乘变换矩阵 (Right quaternion multiplication matrix)
     * 用于将四元数乘法 q_a * q_b 转化为矩阵与向量的乘法：[q_b]_R * q_a_vector
     * @param q_right 被右乘的四元数
     * @return 4x4 的右乘矩阵
     */
    Eigen::Matrix4d quaternionRightProdMat(const Eigen::Quaterniond &q_right);

    /**
     * @brief 四元数到旋转向量的对数映射 (Logarithmic Map: SO(3) -> so(3))
     * 常用于在切空间（Tangent Space）计算两个四元数之间的姿态误差（误差状态变量定义）
     * @param q_error 误差四元数
     * @return 3 维李代数旋转误差向量（四元数虚部的 2 倍近似）
     */
    Eigen::Vector3d quaternionLogMap(const Eigen::Quaterniond &q_error);

    /**
     * @brief 旋转向量到四元数的指数映射 (Exponential Map: so(3) -> SO(3))
     * 用于根据角速度和时间步长 \mathbf{\omega} \cdot \Delta t 增量式递推更新姿态四元数
     * @param phi_rot_vector 3 维旋转增量向量（李代数 so(3)）
     * @return 更新所需的增量四元数 \Delta q
     */
    Eigen::Quaterniond quaternionExpMap(const Eigen::Vector3d &phi_rot_vector);

} // namespace utility

#endif // UTILITY_H