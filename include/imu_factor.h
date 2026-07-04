#ifndef IMU_FACTOR_H
#define IMU_FACTOR_H

#include <ceres/ceres.h>
#include <Eigen/Dense>

class IMUFactor
{
public:
    // 构造函数传入相邻两帧之间高频 IMU 累积计算出的中值相对测量量，以及总时间间隔 dt
    IMUFactor(const Eigen::Vector3d &_delta_p, const Eigen::Vector3d &_delta_v, const Eigen::Quaterniond &_delta_q, double _dt)
        : delta_p(_delta_p), delta_v(_delta_v), delta_q(_delta_q), dt(_dt) {}

    template <typename T>
    bool operator()(const T *const pose_i, const T *const speed_i,
                    const T *const pose_j, const T *const speed_j,
                    T *residuals) const
    {
        // 1. 解析 i 帧的位姿与速度
        Eigen::Matrix<T, 3, 1> Pi(pose_i[0], pose_i[1], pose_i[2]);
        Eigen::Quaternion<T> Qi(pose_i[6], pose_i[3], pose_i[4], pose_i[5]);
        Eigen::Matrix<T, 3, 1> Vi(speed_i[0], speed_i[1], speed_i[2]);

        // 2. 解析 j 帧的位姿与速度
        Eigen::Matrix<T, 3, 1> Pj(pose_j[0], pose_j[1], pose_j[2]);
        Eigen::Quaternion<T> Qj(pose_j[6], pose_j[3], pose_j[4], pose_j[5]);
        Eigen::Matrix<T, 3, 1> Vj(speed_j[0], speed_j[1], speed_j[2]);

        // 定义重力加速度常量 (9.80766)
        Eigen::Matrix<T, 3, 1> G(T(0.0), T(0.0), T(-9.80766));
        T dT = T(dt);

        // 3. 经典的 IMU 动力学残差公式 (将状态预测值与预积分测量值做差)
        // 位移残差 (3维)
        Eigen::Matrix<T, 3, 1> res_p = Qi.inverse() * (Pj - Pi - Vi * dT - T(0.5) * G * dT * dT) - delta_p.template cast<T>();
        // 速度残差 (3维)
        Eigen::Matrix<T, 3, 1> res_v = Qi.inverse() * (Vj - Vi - G * dT) - delta_v.template cast<T>();
        // 姿态残差 (3维，四元数虚部近似代表李代数旋转误差)
        Eigen::Quaternion<T> delta_q_T = delta_q.template cast<T>();
        Eigen::Quaternion<T> res_q = delta_q_T.inverse() * (Qi.inverse() * Qj);

        // 4. 将计算出来的 9 维残差填入输出数组
        residuals[0] = res_p.x(); residuals[1] = res_p.y(); residuals[2] = res_p.z();
        residuals[3] = res_v.x(); residuals[4] = res_v.y(); residuals[5] = res_v.z();
        residuals[6] = res_q.x() * T(2.0); residuals[7] = res_q.y() * T(2.0); residuals[8] = res_q.z() * T(2.0);

        return true;
    }

    // 工厂创建接口
    static ceres::CostFunction *Create(const Eigen::Vector3d &_delta_p, const Eigen::Vector3d &_delta_v, const Eigen::Quaterniond &_delta_q, double _dt)
    {
        // 自动求导配置：残差(9维), 参数块分别为: 位姿i(7), 速度i(3), 位姿j(7), 速度j(3)
        return (new ceres::AutoDiffCostFunction<IMUFactor, 9, 7, 3, 7, 3>(
            new IMUFactor(_delta_p, _delta_v, _delta_q, _dt)));
    }

    Eigen::Vector3d delta_p, delta_v;
    Eigen::Quaterniond delta_q;
    double dt;
};

#endif // IMU_FACTOR_H