#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <thread>
#include <chrono>
#include <iomanip>

#include "parameters.h"
#include "data_manager.h"
#include "feature_tracker.h"
#include "estimator.h" 
// 定义数据集根目录
const std::string DATASET_PATH = "/home/wzj/TUM/dataset-corridor1_512_16/mav0/";

// ==========================================
// 线程 ①：【前端】IMU 数据读取与灌入线程 (200Hz)
// ==========================================
void imuDataThread(DataManager *data_manager)
{
    std::string imu_csv_path = DATASET_PATH + "imu0/data.csv";
    std::ifstream infile(imu_csv_path);

    if (!infile.is_open())
    {
        std::cerr << "[Error] Cannot open IMU csv file: " << imu_csv_path << std::endl;
        return;
    }

    std::string line;
    std::getline(infile, line); // 跳过表头

    std::cout << "[Front-End IMU] Start loading IMU data..." << std::endl;

    while (std::getline(infile, line))
    {
        if (line.empty() || line[0] == '#')
            continue;

        std::stringstream ss(line);
        std::string str_time;
        double timestamp_ns;
        double gx, gy, gz, ax, ay, az;
        char comma;

        if (std::getline(ss, str_time, ','))
        {
            try
            {
                timestamp_ns = std::stod(str_time);
                double timestamp_s = timestamp_ns * 1e-9;

                ss >> gx >> comma >> gy >> comma >> gz >> comma >> ax >> comma >> ay >> comma >> az;

                Eigen::Vector3d acc(ax, ay, az);
                Eigen::Vector3d gyr(gx, gy, gz);

                data_manager->inputImu(timestamp_s, acc, gyr);
            }
            catch (const std::exception &e)
            {
                std::cerr << "[Warning] IMU data parsing error: " << e.what() << std::endl;
                continue;
            }

            // 模拟真实传感器频率（200Hz = 5ms 间隔）
            std::this_thread::sleep_for(std::chrono::microseconds(5000));
        }
    }
    std::cout << "[Front-End IMU] IMU data loading finished." << std::endl;
}

// ==========================================
// 线程 ②：【前端】图像读取与视觉前端特征追踪线程 (20Hz)
// ==========================================
void imageDataThread(DataManager *data_manager)
{
    std::string cam_csv_path = DATASET_PATH + "cam0/data.csv";
    std::ifstream infile(cam_csv_path);

    if (!infile.is_open())
    {
        std::cerr << "[Error] Cannot open Camera csv file: " << cam_csv_path << std::endl;
        return;
    }

    std::string line;
    std::getline(infile, line);

    FeatureTracker tracker;
    std::cout << "[Front-End Vision] Start loading and tracking Image data..." << std::endl;

    while (std::getline(infile, line))
    {
        if (line.empty() || line[0] == '#')
            continue;

        std::stringstream ss(line);
        std::string str_time, filename;

        if (std::getline(ss, str_time, ',') && std::getline(ss, filename))
        {
            if (!filename.empty() && filename.back() == '\r')
                filename.pop_back();

            double timestamp_s = std::stod(str_time) * 1e-9;
            std::string image_path = DATASET_PATH + "cam0/data/" + filename;

            cv::Mat img = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
            if (img.empty())
            {
                std::cerr << "[Warning] Failed to load image at: " << image_path << std::endl;
                continue;
            }

            // 1. 进行视觉特征提取与光流追踪
            std::map<int, Eigen::Vector2d> feature_frame = tracker.trackImage(timestamp_s, img);

            // 2. 如果配置要求显示追踪（show_track: 1）
            if (SHOW_TRACK == 1)
            {
                cv::Mat track_window = tracker.drawTrackImage();
                cv::imshow("VIO Front-End Feature Tracking", track_window);
                cv::waitKey(5);
            }

            // 3. 核心修改：将图像和追踪好的特征点一并灌入数据管理器
            data_manager->inputImage(timestamp_s, img, feature_frame);

            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
    }
    std::cout << "[Front-End Vision] Image data loading finished." << std::endl;
}
// ==========================================
// 线程 ③：【后端】滑动窗口与 Ceres 优化估计线程
// ==========================================
void vioProcessThread(DataManager *data_manager)
{
    std::cout << "[Back-End VIO] Backend optimization thread started." << std::endl;

    // 1. 实例化我们刚刚写好的后端主估计器
    Estimator estimator;

    while (true)
    {
        MeasurementPackage package;

        // 2. 核心对齐阻塞接口：醒来时，代表拿到了一帧干净、且严格与 IMU 同步的数据包
        if (data_manager->getMeasurements(package))
        {
            // 3. 直接将同步数据包丢进 estimator 状态机
            estimator.processMeasurement(package);
        }
    }
}


int main(int argc, char **argv)
{
    std::string config_file = "/home/wzj/vio/config/tum_config.yaml";
    if (!readParameters(config_file))
    {
        std::cerr << "[Fatal] System initialization failed due to configuration errors." << std::endl;
        return -1;
    }

    // 实例化共享的数据管理器
    DataManager data_manager;

    // 分别启动三个独立的并行线程：IMU驱动、图像追踪前端、VIO后端优化
    std::thread t_imu(imuDataThread, &data_manager);
    std::thread t_img(imageDataThread, &data_manager);
    std::thread t_vio(vioProcessThread, &data_manager);

    t_imu.join();
    t_img.join();
    t_vio.join();

    std::cout << "[System] All threads joined. System shutdown cleanly." << std::endl;
    return 0;
}