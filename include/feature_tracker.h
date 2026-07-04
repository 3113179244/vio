#ifndef FEATURE_TRACKER_H
#define FEATURE_TRACKER_H

#include <iostream>
#include <vector>
#include <map>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include "parameters.h"

class FeatureTracker
{
public:
    FeatureTracker();
    ~FeatureTracker() = default;

    // 核心追踪接口：输入当前帧图像和时间戳，返回当前帧追踪并去畸变后的特征点
    // 返回格式：map <特征点ID, 规范化相机坐标系下的 (u, v) 坐标>
    std::map<int, Eigen::Vector2d> trackImage(double timestamp, const cv::Mat &img);

    // 可视化：绘制追踪连线和特征点
    cv::Mat drawTrackImage();

private:
    // 为新提取的特征点分配全局唯一 ID
    bool updateID(unsigned int i);
    
    // 提取新的角点以补充稀疏区域
    void setMask();
    void addPoints();

    // 剔除追踪失败或越界的点
    bool inBorder(const cv::Point2f &pt);
    void reduceVector(std::vector<cv::Point2f> &v, std::vector<uchar> status);
    void reduceVector(std::vector<int> &v, std::vector<uchar> status);

    // 对特征点进行相机去畸变（根据 Kannala-Brandt 模型参数）
    Eigen::Vector2d liftProjective(const cv::Point2f &pt);

    // 图像数据
    cv::Mat prev_img_;
    cv::Mat cur_img_;
    cv::Mat forw_img_; // 最新进来的帧

    // 特征点数据
    std::vector<cv::Point2f> prev_pts_;
    std::vector<cv::Point2f> cur_pts_;
    std::vector<cv::Point2f> forw_pts_;
    std::vector<cv::Point2f> n_pts_; // 新提取的点

    // 特征点 ID 管理
    std::vector<int> ids_;
    std::vector<int> track_cnt_; // 每个点被追踪的帧数（用于后端判断是否稳定）
    
    static int n_id; // 全局自增 ID 计数器
    double cur_time_;
    double prev_time_;
};

#endif // FEATURE_TRACKER_H