#include "feature_tracker.h"

int FeatureTracker::n_id = 0;

FeatureTracker::FeatureTracker()
{
    // 构造函数：初始化参数可以通过 parameters.h 中的全局变量读取
}

bool FeatureTracker::inBorder(const cv::Point2f &pt)
{
    const int BORDER_SIZE = 20;
    int img_x = cvRound(pt.x);
    int img_y = cvRound(pt.y);
    return BORDER_SIZE <= img_x && img_x < IMAGE_WIDTH - BORDER_SIZE && 
           BORDER_SIZE <= img_y && img_y < IMAGE_HEIGHT - BORDER_SIZE;
}

void FeatureTracker::reduceVector(std::vector<cv::Point2f> &v, std::vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < (int)v.size(); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

void FeatureTracker::reduceVector(std::vector<int> &v, std::vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < (int)v.size(); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

// 针对 Kannala-Brandt 鱼眼/大畸变模型的去畸变与投影函数
// 输入图像像素坐标，输出归一化相机坐标系 (u, v) 满足 z=1
Eigen::Vector2d FeatureTracker::liftProjective(const cv::Point2f &pt)
{
    // 从参数服务器或全局变量获取内参 (来自参数读取模块加载的 mu, mv, u0, v0)
    // fx = FX, fy = FY, cx = CX, cy = CY
    double x = (pt.x - CX) / FX;
    double y = (pt.y - CY) / FY;
    
    double r = sqrt(x * x + y * y);
    if (r < 1e-8) {
        return Eigen::Vector2d(x, y);
    }

    // 通过多项式牛顿迭代或近似解反解出真实的 theta (即去畸变角度)
    // 这里的 yaml 提供了 k2, k3, k4, k5。
    // 简易近似：如果畸变较小可直接使用普通模型，这里采用针对 KB 模型的反畸变牛顿法迭代
    double theta = atan(r); 
    for (int i = 0; i < 5; i++) {
        double theta2 = theta * theta;
        double theta3 = theta2 * theta;
        double theta5 = theta3 * theta2;
        double theta7 = theta5 * theta2;
        double theta9 = theta7 * theta2;
        
        // f(theta) = theta + k2*theta^3 + k3*theta^5 + k4*theta^7 + k5*theta^9 - r
        double f = theta + K2 * theta3 + K3 * theta5 + K4 * theta7 + K5 * theta9 - r;
        // f'(theta)
        double df = 1.0 + 3.0 * K2 * theta2 + 5.0 * K3 * theta * theta3 + 7.0 * K4 * theta5 * theta + 9.0 * K5 * theta7 * theta;
        theta = theta - f / df;
    }

    double scale = tan(theta) / r;
    return Eigen::Vector2d(x * scale, y * scale);
}

void FeatureTracker::setMask()
{
    // 创建掩码，防止特征点密集地挤在同一个区域，保持特征点均匀分布
    cv::Mat mask = cv::Mat(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC1, cv::Scalar(255));
    
    // 根据追踪成功率对特征点排序，优先保留追踪时间长的稳定点
    std::vector<std::pair<int, std::pair<cv::Point2f, int>>> effect_pts;
    for (size_t i = 0; i < forw_pts_.size(); i++) {
        effect_pts.push_back(std::make_pair(track_cnt_[i], std::make_pair(forw_pts_[i], ids_[i])));
    }
    std::sort(effect_pts.begin(), effect_pts.end(), [](const auto &a, const auto &b) {
        return a.first > b.first;
    });

    forw_pts_.clear();
    ids_.clear();
    track_cnt_.clear();

    for (auto &it : effect_pts) {
        if (mask.at<uchar>(it.second.first) == 255) {
            forw_pts_.push_back(it.second.first);
            ids_.push_back(it.second.second);
            track_cnt_.push_back(it.first);
            // 在特征点周围画圈，禁止在 MIN_DIST 范围内重复提取新点 (min_dist: 25)
            cv::circle(mask, it.second.first, MIN_DIST, 0, -1);
        }
    }
}

void FeatureTracker::addPoints()
{
    // 均匀补充新特征点
    int n_max_cnt = MAX_CNT - forw_pts_.size(); // max_cnt: 150
    if (n_max_cnt > 0) {
        cv::Mat mask = cv::Mat(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC1, cv::Scalar(255));
        for (const auto &pt : forw_pts_) {
            cv::circle(mask, pt, MIN_DIST, 0, -1);
        }

        std::vector<cv::Point2f> new_pts;
        cv::goodFeaturesToTrack(forw_img_, new_pts, n_max_cnt, 0.01, MIN_DIST, mask);

        for (const auto &pt : new_pts) {
            forw_pts_.push_back(pt);
            ids_.push_back(n_id++);
            track_cnt_.push_back(1);
        }
    }
}

std::map<int, Eigen::Vector2d> FeatureTracker::trackImage(double timestamp, const cv::Mat &img)
{
    forw_img_ = img;
    forw_pts_.clear();

    if (cur_pts_.size() > 0) {
        // 1. KLT 光流追踪
        std::vector<uchar> status;
        std::vector<float> err;
        cv::calcOpticalFlowPyrLK(cur_img_, forw_img_, cur_pts_, forw_pts_, status, err, cv::Size(21, 21), 3);

        // 剔除越界点
        for (int i = 0; i < (int)forw_pts_.size(); i++) {
            if (status[i] && !inBorder(forw_pts_[i])) {
                status[i] = 0;
            }
        }

        reduceVector(cur_pts_, status);
        reduceVector(forw_pts_, status);
        reduceVector(ids_, status);
        reduceVector(track_cnt_, status);
    }

    // 更新追踪计数
    for (auto &n : track_cnt_) n++;

    // 2. 利用基础矩阵 RANSAC 剔除错配点 (F_threshold: 1.0)
    if (forw_pts_.size() >= 8) {
        std::vector<uchar> status;
        cv::findFundamentalMat(cur_pts_, forw_pts_, cv::FM_RANSAC, F_THRESHOLD, 0.99, status);
        reduceVector(cur_pts_, status);
        reduceVector(forw_pts_, status);
        reduceVector(ids_, status);
        reduceVector(track_cnt_, status);
    }

    // 3. 均匀化当前点并补充新点
    setMask();
    addPoints();

    // 4. 轮转图像缓存
    prev_img_ = cur_img_;
    cur_img_ = forw_img_;
    cur_pts_ = forw_pts_;

    // 5. 将当前帧的特征点进行去畸变投影，打包为规范化相机坐标传给后端
    std::map<int, Eigen::Vector2d> feature_points;
    for (size_t i = 0; i < forw_pts_.size(); i++) {
        Eigen::Vector2d un_pt = liftProjective(forw_pts_[i]);
        feature_points[ids_[i]] = un_pt;
    }

    return feature_points;
}

cv::Mat FeatureTracker::drawTrackImage()
{
    // 将特征点可视化在彩色图上
    cv::Mat out_img;
    cv::cvtColor(forw_img_, out_img, cv::COLOR_GRAY2BGR);
    for (size_t i = 0; i < forw_pts_.size(); i++) {
        cv::circle(out_img, forw_pts_[i], 3, cv::Scalar(0, 255, 0), -1);
        // 如果是被长久追踪的稳定点，用红色细线画出其与前一帧的运动轨迹
        if (track_cnt_[i] > 2 && i < cur_pts_.size()) {
            cv::line(out_img, cur_pts_[i], forw_pts_[i], cv::Scalar(0, 0, 255), 1);
        }
    }
    return out_img;
}