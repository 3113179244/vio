#include "data_manager.h"

void DataManager::inputImu(double timestamp, const Eigen::Vector3d &acc, const Eigen::Vector3d &gyr)
{
    std::unique_lock<std::mutex> lock(mutex_);
    ImuMeasurement imu{timestamp, acc, gyr};
    imu_buf_.push(imu);
    cond_.notify_one(); // 唤醒可能正在等待数据的后端线程
}

void DataManager::inputImage(double timestamp, const cv::Mat &image)
{
    std::unique_lock<std::mutex> lock(mutex_);
    ImageMeasurement img{timestamp, image.clone()}; // 使用 clone 防止外部内存释放
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

    // 1. 提取当前最前面的图像帧
    package.image = image_buf_.front();
    image_buf_.pop();

    package.imus.clear();
    // 2. 提取当前图像帧之前的所有 IMU 数据（包含与当前图像时间戳最接近的那一帧）
    // 注意：这里保留 <= 图像时间戳的 IMU 数据。
    // 为了使下一帧图像做预积分时连续，这一帧 IMU 的数据通常在处理完后不需要从 imu_buf 中彻底抹去，或者留作下一帧的起点。
    // 下面是一种常见的标准做法：把当前图像时间戳之前的 IMU 提取出来。
    while (!imu_buf_.empty() && imu_buf_.front().timestamp <= package.image.timestamp)
    {
        package.imus.push_back(imu_buf_.front());

        // 如果是最后一帧接近图像时间戳的 IMU，我们可以保留它在队列中，作为下一帧图像预积分的起点
        if (imu_buf_.front().timestamp == package.image.timestamp)
        {
            break;
        }

        // 如果当前 IMU 时间已经大于等于下一帧可能用到的范围（或紧邻当前图像），在严格对齐时可以保留最后一帧
        if (imu_buf_.size() > 1 && imu_buf_.queue::front().timestamp < package.image.timestamp)
        {
            // 如果下一个 IMU 已经超过了当前图像时间戳，说明当前 IMU 是图象前的最后一帧
            auto it_next = imu_buf_.front();
            // 简单起见，通常直接把小于当前图像时间戳的 IMU 全部 pop 掉
            imu_buf_.pop();
        }
        else
        {
            // 如果只剩一个或者正好等于，保留不 pop（或者直接 pop，取决于你的预积分插值策略）
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