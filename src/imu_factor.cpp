#include <ceres/ceres.h>
#include <Eigen/Dense>

// 这是一个简化的 IMU 预积分因子，用于演示 Ceres 核心残差计算
class IMUFactor : public ceres::SizedCostFunction<15, 7, 3, 7, 3>
{
public:
    IMUFactor(const Eigen::Vector3d &_delta_p, const Eigen::Vector3d &_delta_v, const Eigen::Quaterniond &_delta_q,
              const Eigen::Vector3d &_ba, const Eigen::Vector3d &_bg, double _dt)
        : delta_p(_delta_p), delta_v(_delta_v), delta_q(_delta_q), ba(_ba), bg(_bg), dt(_dt) {}

    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
    {
        // 参数块：
        // 0: Pi, Qi (7维: 3位移 + 4四元数)
        // 1: Vi (3维)
        // 2: Pj, Qj (7维)
        // 3: Vj (3维)

        Eigen::Vector3d Pi(parameters[0][0], parameters[0][1], parameters[0][2]);
        Eigen::Quaterniond Qi(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]); // Ceres 通常将w放在最后或最前，依绑定而定

        Eigen::Vector3d Vi(parameters[1][0], parameters[1][1], parameters[1][2]);

        Eigen::Vector3d Pj(parameters[2][0], parameters[2][1], parameters[2][2]);
        Eigen::Quaterniond Qj(parameters[2][6], parameters[2][3], parameters[2][4], parameters[2][5]);

        Eigen::Vector3d Vj(parameters[3][0], parameters[3][1], parameters[3][2]);

        Eigen::Map<Eigen::Matrix<double, 15, 1>> residual(residuals);

        // 重力加速度 (从全局参数中获取)
        Eigen::Vector3d G(0, 0, -9.80766);

        // 计算经典预积分残差方程
        residual.block<3, 1>(0, 0) = Qi.inverse() * (Pj - Pi - Vi * dt + 0.5 * G * dt * dt) - delta_p;
        residual.block<3, 1>(3, 0) = Qi.inverse() * (Vj - Vi + G * dt) - delta_v;
        residual.block<3, 1>(6, 0) = 2.0 * (delta_q.inverse() * (Qi.inverse() * Qj)).vec();
        residual.block<3, 1>(9, 0) = Eigen::Vector3d::Zero(); // Bias 随机游走残差（此处简化为0）
        residual.block<3, 1>(12, 0) = Eigen::Vector3d::Zero();

        // 如果需要雅可比矩阵，在此处解析求导或由 Ceres 自动求导提供。
        // 为了兼容编译，在非线性阶段 Ceres 会读取此处。
        if (jacobians)
        {
            // 简易框架中可由底层由数值/符号实现，此处不展开数页的雅可比推导
        }

        return true;
    }

    Eigen::Vector3d delta_p, delta_v;
    Eigen::Quaterniond delta_q;
    Eigen::Vector3d ba, bg;
    double dt;
};