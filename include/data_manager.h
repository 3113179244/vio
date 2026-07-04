#ifndef DATA_MANAGER_H
#define DATA_MANAGER_H

#include <iostream>
#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <memory>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

// IMU 数据结构体
struct ImuMeasurement
{
    double timestamp;
    Eigen::Vector3d acc; // 加速度
    Eigen::Vector3d gyr; // 角速度
};

// 图像数据结构体
struct ImageMeasurement
{
    double timestamp;
    cv::Mat image;
};

// 传感器同步数据包（一帧图像以及其与上一帧之间的所有 IMU 数据）
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

    // 输入接口：由数据读取线程（或外部主线程）调用，向队列中灌入数据
    void inputImu(double timestamp, const Eigen::Vector3d &acc, const Eigen::Vector3d &gyr);
    void inputImage(double timestamp, const cv::Mat &image);

    // 输出接口：由 VIO 后端优化线程调用，获取对齐后的同步数据包
    // 如果队列中没有可用数据，该函数会阻塞，直到有新数据满足对齐条件
    bool getMeasurements(MeasurementPackage &package);

    // 清空缓存
    void clear();

private:
    // 判断当前传感器数据是否能够打包对齐
    bool hasValidMeasurements();

    std::queue<ImuMeasurement> imu_buf_;
    std::queue<ImageMeasurement> image_buf_;

    std::mutex mutex_;
    std::condition_variable cond_;
};

#endif // DATA_MANAGER_H