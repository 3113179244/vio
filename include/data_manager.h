#ifndef DATA_MANAGER_H
#define DATA_MANAGER_H

#include <iostream>
#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <memory>
#include <map>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

// IMU 数据结构体
struct ImuMeasurement
{
    double timestamp;
    Eigen::Vector3d acc; // 加速度
    Eigen::Vector3d gyr; // 角速度
};

// 图像与视觉特征点数据结构体
struct ImageMeasurement
{
    double timestamp;
    cv::Mat image; // 原始图像
    std::map<int, Eigen::Vector2d> feature_points; // 新增：前端追踪出来的去畸变特征点集合！
};

// 传感器同步数据包（一帧图像、对应的特征点，以及它与上一帧之间的所有 IMU 数据）
struct MeasurementPackage
{
    ImageMeasurement image;
    std::vector<ImuMeasurement> imus;
};

class DataManager
{
public:
    DataManager() = default;
    ~DataManager() = default;

    // 禁止拷贝
    DataManager(const DataManager &) = delete;
    DataManager &operator=(const DataManager &) = delete;

    // 输入接口
    void inputImu(double timestamp, const Eigen::Vector3d &acc, const Eigen::Vector3d &gyr);
    // 修改接口：不仅输入时间戳和图像，同时输入追踪好的特征点
    void inputImage(double timestamp, const cv::Mat &image, const std::map<int, Eigen::Vector2d> &features);

    // 输出接口
    bool getMeasurements(MeasurementPackage &package);

    // 清空缓存
    void clear();

private:
    bool hasValidMeasurements();

    std::queue<ImuMeasurement> imu_buf_;
    std::queue<ImageMeasurement> image_buf_;

    std::mutex mutex_;
    std::condition_variable cond_;
};

#endif // DATA_MANAGER_H