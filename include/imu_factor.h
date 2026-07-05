#ifndef IMU_FACTOR_H
#define IMU_FACTOR_H

#include <ceres/ceres.h>
#include <Eigen/Dense>
#include <sophus/so3.hpp> // 引入 Sophus 支持 Ceres 自动求导的泛型头文件

class IMUFactor
{
public:
    IMUFactor(const Eigen::Vector3d &_delta_p, const Eigen::Vector3d &_delta_v, const Eigen::Quaterniond &_delta_q, double _dt)
        : delta_p(_delta_p), delta_v(_delta_v), delta_q(_delta_q), dt(_dt) {}

    template <typename T>
    bool operator()(const T *const pose_i, const T *const speed_i,
                    const T *const pose_j, const T *const speed_j,
                    T *residuals) const
    {
        // 1. 解析 i 帧和 j 帧的状态量
        Eigen::Matrix<T, 3, 1> Pi(pose_i[0], pose_i[1], pose_i[2]);
        Eigen::Quaternion<T> Qi(pose_i[6], pose_i[3], pose_i[4], pose_i[5]);
        Eigen::Matrix<T, 3, 1> Vi(speed_i[0], speed_i[1], speed_i[2]);

        Eigen::Matrix<T, 3, 1> Pj(pose_j[0], pose_j[1], pose_j[2]);
        Eigen::Quaternion<T> Qj(pose_j[6], pose_j[3], pose_j[4], pose_j[5]);
        Eigen::Matrix<T, 3, 1> Vj(speed_j[0], speed_j[1], speed_j[2]);

        Eigen::Matrix<T, 3, 1> G(T(0.0), T(0.0), T(-9.80766));
        T dT = T(dt);

        // 2. 计算位移与速度残差 (保持不变)
        Eigen::Matrix<T, 3, 1> res_p = Qi.inverse() * (Pj - Pi - Vi * dT - T(0.5) * G * dT * dT) - delta_p.template cast<T>();
        Eigen::Matrix<T, 3, 1> res_v = Qi.inverse() * (Vj - Vi - G * dT) - delta_v.template cast<T>();

        // 3. 【核心修改】利用 Sophus 计算优雅的姿态残差
        // 将状态量和测量量全部打包为支持 Ceres 自动求导的 Sophus::SO3<T> 旋转对象
        Sophus::SO3<T> SO3_Qi(Qi);
        Sophus::SO3<T> SO3_Qj(Qj);
        Sophus::SO3<T> SO3_delta_q(delta_q.template cast<T>());

        // 姿态残差公式：测量值的逆 * (i帧的逆 * j帧)
        // 完美的李群乘法，并通过 .log() 一键转换成 3 维的李代数旋转残差！
        Eigen::Matrix<T, 3, 1> res_q = (SO3_delta_q.inverse() * (SO3_Qi.inverse() * SO3_Qj)).log();

        // 4. 将 9 维残差填入输出数组
        residuals[0] = res_p.x();
        residuals[1] = res_p.y();
        residuals[2] = res_p.z();
        
        residuals[3] = res_v.x();
        residuals[4] = res_v.y();
        residuals[5] = res_v.z();
        
        // 姿态残差现在由 Sophus 严格计算，不再需要手动乘 2 逼近
        residuals[6] = res_q.x();
        residuals[7] = res_q.y();
        residuals[8] = res_q.z();

        return true;
    }

    static ceres::CostFunction *Create(const Eigen::Vector3d &_delta_p, const Eigen::Vector3d &_delta_v, const Eigen::Quaterniond &_delta_q, double _dt)
    {
        return (new ceres::AutoDiffCostFunction<IMUFactor, 9, 7, 3, 7, 3>(
            new IMUFactor(_delta_p, _delta_v, _delta_q, _dt)));
    }

    Eigen::Vector3d delta_p, delta_v;
    Eigen::Quaterniond delta_q;
    double dt;
};

#endif // IMU_FACTOR_H