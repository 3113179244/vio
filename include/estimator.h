#ifndef ESTIMATOR_H
#define ESTIMATOR_H

#include <iostream>
#include <vector>
#include <map>
#include <queue>
#include <mutex>
#include <iomanip>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include "data_manager.h"
#include "parameters.h"

// 特征点在滑窗内单帧的观测信息
struct FeaturePerFrame
{
    Eigen::Vector2d point; // 归一化相机坐标系下的 (u, v)
};

// 特征点在滑窗生命周期内的全局管理器
struct FeaturePerId
{
    int feature_id;
    int start_frame;                                // 第一次被滑窗内哪一帧看到
    std::vector<FeaturePerFrame> feature_per_frame; // 每一帧的对应观测记录
    double estimated_depth;                         // 逆深度估计值
};

class Estimator
{
public:
    Estimator();
    ~Estimator() = default;

    void processMeasurement(const MeasurementPackage &package);

    enum SolverFlag
    {
        INITIAL,
        NON_LINEAR
    };
    SolverFlag solver_flag;

    // 仿 VINS-Mono 类内静态常量，防止与全局命名冲突
    static constexpr int WINDOW_SIZE = 10;

private:
    void processIMU(double dt, const Eigen::Vector3d &linear_acceleration, const Eigen::Vector3d &angular_velocity);
    void processImage(const std::map<int, Eigen::Vector2d> &image_msg, double header_time);

    bool initialStructure();
    void optimization();
    void slideWindow();

    // ---- 滑动窗口内的状态量 ----
    double Headers[WINDOW_SIZE + 1];
    Eigen::Vector3d Ps[WINDOW_SIZE + 1];
    Eigen::Vector3d Vs[WINDOW_SIZE + 1];
    Eigen::Matrix3d Rs[WINDOW_SIZE + 1];
    Eigen::Vector3d Bas[WINDOW_SIZE + 1];
    Eigen::Vector3d Bgs[WINDOW_SIZE + 1];

    // ==== 核心新增：存放滑窗内相邻两帧之间的真实预积分测量值 ====
    double dt_buf[WINDOW_SIZE];                      // 相邻帧间的时间差
    Eigen::Vector3d delta_p_buf[WINDOW_SIZE];        // 相邻帧间的位移预积分量
    Eigen::Vector3d delta_v_buf[WINDOW_SIZE];        // 相邻帧间的速度预积分量
    Eigen::Quaterniond delta_q_buf[WINDOW_SIZE];     // 相邻帧间的姿态预积分量

    // ==== 后端地图特征点管理容器 ====
    std::map<int, FeaturePerId> feature_manager;

    int frame_count;
    double last_imu_time;
    bool first_imu;
};

#endif // ESTIMATOR_H