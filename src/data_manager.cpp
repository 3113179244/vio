#include "data_manager.h"

void DataManager::inputImu(double timestamp, const Eigen::Vector3d &acc, const Eigen::Vector3d &gyr)
{
    std::unique_lock<std::mutex> lock(mutex_);
    ImuMeasurement imu{timestamp, acc, gyr};
    imu_buf_.push(imu);
    cond_.notify_one(); // 唤醒可能正在等待数据的后端线程
}

// 修复点：这里的实现必须与头文件严格一致，接收并打包 features 集合
void DataManager::inputImage(double timestamp, const cv::Mat &image, const std::map<int, Eigen::Vector2d> &features)
{
    std::unique_lock<std::mutex> lock(mutex_);
    ImageMeasurement img{timestamp, image.clone(), features}; // 使用 features 构造结构体
    image_buf_.push(img);
    cond_.notify_one();
}

bool DataManager::hasValidMeasurements()
{
    if (image_buf_.empty() || imu_buf_.empty())
    {
        return false;
    }

    // 核心对齐条件：IMU 最新的数据时间戳，必须大于（或等于）图像队列中第一帧的时间戳
    // 这样才能保证图像帧有足够的 IMU 数据来进行积分处理
    if (imu_buf_.back().timestamp < image_buf_.front().timestamp)
    {
        return false;
    }

    // 容错处理：如果 IMU 的时间戳远远落后于当前图像（可能是初始化或丢帧），需要剔除过时的图像
    if (image_buf_.front().timestamp < imu_buf_.front().timestamp)
    {
        std::cerr << "[DataManager] Image timestamp is older than the oldest IMU. Pop old image." << std::endl;
        image_buf_.pop();
        return false;
    }

    return true;
}

bool DataManager::getMeasurements(MeasurementPackage &package)
{
    std::unique_lock<std::mutex> lock(mutex_);

    // 阻塞等待，直到队列中有满足对齐条件的数据
    cond_.wait(lock, [this]()
               { return hasValidMeasurements(); });

    // 1. 提取当前最前面的图像帧（内部已包含特征点）
    package.image = image_buf_.front();
    image_buf_.pop();

    package.imus.clear();
    
    // 2. 提取当前图像帧之前的所有 IMU 数据
    while (!imu_buf_.empty() && imu_buf_.front().timestamp <= package.image.timestamp)
    {
        package.imus.push_back(imu_buf_.front());

        // 如果当前 IMU 时间戳恰好等于图像时间戳，说明对齐到了完美边界
        if (imu_buf_.front().timestamp == package.image.timestamp)
        {
            imu_buf_.pop();
            break;
        }

        // 如果当前 IMU 的下一条数据依然小于当前图像时间戳，说明当前这条是安全的旧数据，直接弹出
        if (imu_buf_.size() > 1)
        {
            imu_buf_.pop();
        }
        else
        {
            // 如果 buffer 里面只剩这一条数据了，弹出并结束
            imu_buf_.pop();
            break;
        }
    }

    return true;
}

void DataManager::clear()
{
    std::unique_lock<std::mutex> lock(mutex_);
    std::queue<ImuMeasurement>().swap(imu_buf_);
    std::queue<ImageMeasurement>().swap(image_buf_);
}