#ifndef PARAMETERS_H
#define PARAMETERS_H

#include <iostream>
#include <string>
#include <vector>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

// ==== 1. 全局参数声明 (通过 extern 供全工程使用) ====

// 常见参数 / 话题 / 路径
extern std::string IMU_TOPIC;
extern std::string IMAGE_TOPIC;
extern std::string OUTPUT_PATH;

// 相机外壳/名称信息
extern std::string MODEL_TYPE;
extern std::string CAMERA_NAME;
extern int IMAGE_WIDTH;
extern int IMAGE_HEIGHT;

// 相机内参及畸变 (KANNALA_BRANDT 模型)
extern double FX, FY, CX, CY;
extern double K2, K3, K4, K5;

// 相机-IMU 外参控制
extern int ESTIMATE_EXTRINSIC;
extern Eigen::Matrix3d RIC;
extern Eigen::Vector3d TIC;

// 特征提取与追踪参数
extern int MAX_CNT;
extern int MIN_DIST;
extern int WINDOW_SIZE;
extern double FREQ;
extern double F_THRESHOLD;
extern int SHOW_TRACK;
extern int EQUALIZE;

// 优化控制参数
extern double MAX_SOLVER_TIME;
extern int MAX_NUM_ITERATIONS;
extern double KEYFRAME_PARALLAX;

// IMU 噪声与重力参数
extern double ACC_N, GYR_N;
extern double ACC_W, GYR_W;
extern double G_NORM;

// 回环检测相关参数
extern int LOOP_CLOSURE;
extern int LOAD_PREVIOUS_POSE_GRAPH;
extern int FAST_RELOCALIZATION;
extern std::string POSE_GRAPH_SAVE_PATH;

// 在线时间同步参数
extern int ESTIMATE_TD;
extern double TD;

// 快门参数 (卷帘/全局)
extern int ROLLING_SHUTTER;
extern double ROLLING_SHUTTER_TR;

// 可视化与调试参数
extern int SAVE_IMAGE;
extern int VISUALIZE_IMU_FORWARD;
extern int VISUALIZE_CAMERA;

// ==== 2. 读取函数声明 =======
bool readParameters(const std::string& config_file_path);

#endif // PARAMETERS_H