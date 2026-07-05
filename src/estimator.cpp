#include "estimator.h"
#include "projection_factor.h"
#include <ceres/ceres.h>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <map>
#include <vector>

// 辅助函数：将 Eigen 的 R, T 转换为 OpenCV 矩阵格式
void eigen2cv(const Eigen::Matrix3d &R, const Eigen::Vector3d &T, cv::Mat &cv_R, cv::Mat &cv_T)
{
    cv_R = (cv::Mat_<double>(3, 3) << R(0,0), R(0,1), R(0,2), R(1,0), R(1,1), R(1,2), R(2,0), R(2,1), R(2,2));
    cv_T = (cv::Mat_<double>(3, 1) << T.x(), T.y(), T.z());
}

void cv2eigen(const cv::Mat &cv_R, const cv::Mat &cv_T, Eigen::Matrix3d &R, Eigen::Vector3d &T)
{
    R << cv_R.at<double>(0,0), cv_R.at<double>(0,1), cv_R.at<double>(0,2),
         cv_R.at<double>(1,0), cv_R.at<double>(1,1), cv_R.at<double>(1,2),
         cv_R.at<double>(2,0), cv_R.at<double>(2,1), cv_R.at<double>(2,2);
    T << cv_T.at<double>(0,0), cv_T.at<double>(1,0), cv_T.at<double>(2,0);
}

// 纯视觉三角化基础函数（计算三维坐标）
Eigen::Vector3d triangulatePoint(const Eigen::Matrix3d &R0, const Eigen::Vector3d &T0,
                                 const Eigen::Matrix3d &R1, const Eigen::Vector3d &T1,
                                 const Eigen::Vector2d &pt0, const Eigen::Vector2d &pt1)
{
    Eigen::Matrix4d P0, P1;
    P0.block<3,3>(0,0) = R0.transpose(); P0.block<3,1>(0,3) = -R0.transpose() * T0;
    P1.block<3,3>(0,0) = R1.transpose(); P1.block<3,1>(0,3) = -R1.transpose() * T1;
    
    Eigen::Matrix4d A;
    A.row(0) = pt0.x() * P0.row(2) - P0.row(0);
    A.row(1) = pt0.y() * P0.row(2) - P0.row(1);
    A.row(2) = pt1.x() * P1.row(2) - P1.row(0);
    A.row(3) = pt1.y() * P1.row(2) - P1.row(1);

    Eigen::JacobiSVD<Eigen::Matrix4d> svd(A, Eigen::ComputeFullV);
    Eigen::Vector4d x_homo = svd.matrixV().col(3);
    return x_homo.head<3>() / x_homo(3);
}

// ==========================================
// 1. 构造函数
// ==========================================
Estimator::Estimator()
{
    std::cout << "[Estimator] 初始化纯视觉滑窗后端估计器 (无恒速模型)..." << std::endl;
    solver_flag = INITIAL;
    frame_count = 0;
    first_imu = true;
    last_imu_time = 0.0;

    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        Ps[i].setZero();
        Vs[i].setZero();
        Rs[i].setIdentity();
        Bas[i].setZero();
        Bgs[i].setZero();
    }
    feature_manager.clear();
}

// 补齐头文件声明的成员函数，保持结构完整
void Estimator::processIMU(double dt, const Eigen::Vector3d &linear_acceleration, const Eigen::Vector3d &angular_velocity)
{
    // 纯视觉模式下此函数留空
}

// 数据处理入口
void Estimator::processMeasurement(const MeasurementPackage &package)
{
    std::cout << "\n[Backend Estimator] ====== 开始处理同步数据包 ======" << std::endl;
    processImage(package.image.feature_points, package.image.timestamp);
}

// ==========================================
// 2. 视觉状态机更新
// ==========================================
void Estimator::processImage(const std::map<int, Eigen::Vector2d> &image_msg, double header_time)
{
    std::cout << "[Backend Estimator] 当前帧追踪到的特征点数量: " << image_msg.size() << std::endl;

    int current_frame_idx = (frame_count < WINDOW_SIZE) ? frame_count : WINDOW_SIZE;
    for (const auto &pts : image_msg)
    {
        int feature_id = pts.first;
        Eigen::Vector2d point_uv = pts.second;

        if (feature_manager.find(feature_id) == feature_manager.end())
        {
            FeaturePerId new_feature;
            new_feature.feature_id = feature_id;
            new_feature.start_frame = current_frame_idx;
            new_feature.estimated_depth = 1.0; // 默认初始逆深度
            new_feature.feature_per_frame.push_back({point_uv});
            feature_manager[feature_id] = new_feature;
        }
        else
        {
            feature_manager[feature_id].feature_per_frame.push_back({point_uv});
        }
    }

    if (frame_count < WINDOW_SIZE)
    {
        Headers[frame_count] = header_time;
        frame_count++;
        std::cout << "[Backend Estimator] 窗口未满... 当前进度: " << frame_count << "/" << WINDOW_SIZE << std::endl;
    }
    else
    {
        Headers[WINDOW_SIZE] = header_time;

        if (solver_flag == INITIAL)
        {
            std::cout << "[Backend Estimator] 窗口已满，开始 SfM 初始化..." << std::endl;
            if (initialStructure())
            {
                solver_flag = NON_LINEAR; 
                std::cout << "[Backend Estimator] SfM 初始化成功，进入非线性优化状态！" << std::endl;
                optimization(); 
            }
            slideWindow();
        }
        else
        {
            // 💡【已删除恒速模型代码】：最新进来的帧不再强行加 delta_P 和 delta_R 推力
            // 初始值直接继承上一帧优化出来的状态，完全交给 Ceres 视觉因子去拉动和解算
            Ps[WINDOW_SIZE] = Ps[WINDOW_SIZE - 1];
            Rs[WINDOW_SIZE] = Rs[WINDOW_SIZE - 1];

            // 💡【逆深度初值保护】：在优化前，对于新滑入窗口的特征点或者深度未初始化的特征点，
            // 利用当前窗口的相机位姿对其进行即时三角化，为其赋予合理的初值，代替盲目的默认值 1.0
            for (auto &it : feature_manager)
            {
                auto &feature = it.second;
                if (feature.estimated_depth == 1.0 && feature.feature_per_frame.size() >= 2)
                {
                    int i = feature.start_frame;
                    int j = i + feature.feature_per_frame.size() - 1;
                    if (j <= WINDOW_SIZE)
                    {
                        Eigen::Vector2d pt_i = feature.feature_per_frame[0].point;
                        Eigen::Vector2d pt_j = feature.feature_per_frame.back().point;
                        Eigen::Vector3d P_w = triangulatePoint(Rs[i], Ps[i], Rs[j], Ps[j], pt_i, pt_j);
                        double depth = (Rs[i].transpose() * (P_w - Ps[i])).z();
                        if (depth > 0.1)
                        {
                            feature.estimated_depth = 1.0 / depth;
                        }
                    }
                }
            }

            optimization();
            slideWindow();
        }
    }
}

// ==========================================
// 3. 纯视觉滑窗 SfM 初始化
// ==========================================
bool Estimator::initialStructure()
{
    int l = 0; 
    int r = WINDOW_SIZE;

    std::vector<cv::Point2f> pts_l, pts_r;
    std::vector<int> common_feature_ids;

    for (const auto &it : feature_manager)
    {
        const auto &feature = it.second;
        if (feature.start_frame == l && feature.feature_per_frame.size() == (size_t)(r - l + 1))
        {
            pts_l.push_back(cv::Point2f(feature.feature_per_frame[0].point.x(), feature.feature_per_frame[0].point.y()));
            pts_r.push_back(cv::Point2f(feature.feature_per_frame[r - l].point.x(), feature.feature_per_frame[r - l].point.y()));
            common_feature_ids.push_back(feature.feature_id);
        }
    }

    if (pts_l.size() < 30)
    {
        std::cerr << "[SfM 失败] 共视特征点太少(" << pts_l.size() << ")，等待下一帧..." << std::endl;
        return false;
    }

    cv::Mat mask;
    cv::Mat E = cv::findEssentialMat(pts_l, pts_r, 1.0, cv::Point2d(0, 0), cv::FM_RANSAC, 0.999, 0.003, mask);
    
    cv::Mat cv_R, cv_T;
    cv::recoverPose(E, pts_l, pts_r, cv_R, cv_T, 1.0, cv::Point2d(0, 0), mask);

    Rs[l].setIdentity();
    Ps[l].setZero();

    Eigen::Matrix3d R_r_to_l;
    Eigen::Vector3d T_r_to_l;
    cv2eigen(cv_R, cv_T, R_r_to_l, T_r_to_l);
    
    Rs[r] = Rs[l] * R_r_to_l.transpose();
    Ps[r] = Ps[l] - Rs[r] * T_r_to_l;

    std::map<int, Eigen::Vector3d> sfm_cloud; 
    for (size_t i = 0; i < pts_l.size(); i++)
    {
        if (mask.at<uchar>(i) == 0) continue;

        int feature_id = common_feature_ids[i];
        Eigen::Vector2d pl(pts_l[i].x, pts_l[i].y);
        Eigen::Vector2d pr(pts_r[i].x, pts_r[i].y);

        Eigen::Vector3d P_w = triangulatePoint(Rs[l], Ps[l], Rs[r], Ps[r], pl, pr);
        
        double depth_l = (Rs[l].transpose() * (P_w - Ps[l])).z();
        double depth_r = (Rs[r].transpose() * (P_w - Ps[r])).z();
        
        if (depth_l > 0.1 && depth_r > 0.1)
        {
            sfm_cloud[feature_id] = P_w;
        }
    }

    for (int i = 1; i < r; i++)
    {
        std::vector<cv::Point3f> pts_3d;
        std::vector<cv::Point2f> pts_2d;

        for (const auto &it : feature_manager)
        {
            int feature_id = it.first;
            const auto &feature = it.second;
            
            if (sfm_cloud.find(feature_id) != sfm_cloud.end() && i >= feature.start_frame && (size_t)(i - feature.start_frame) < feature.feature_per_frame.size())
            {
                Eigen::Vector3d P_w = sfm_cloud[feature_id];
                pts_3d.push_back(cv::Point3f(P_w.x(), P_w.y(), P_w.z()));
                
                Eigen::Vector2d uv = feature.feature_per_frame[i - feature.start_frame].point;
                pts_2d.push_back(cv::Point2f(uv.x(), uv.y()));
            }
        }

        if (pts_3d.size() < 10)
        {
            std::cerr << "[SfM 失败] 中间帧 PnP 关联 3D 点太少！" << std::endl;
            return false;
        }

        cv::Mat rvec, tvec;
        cv::Mat K = (cv::Mat_<double>(3,3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
        cv::solvePnP(pts_3d, pts_2d, K, cv::Mat(), rvec, tvec, false, cv::SOLVEPNP_EPNP);
        
        cv::Mat R_cv;
        cv::Rodrigues(rvec, R_cv);

        Eigen::Matrix3d R_c_w;
        Eigen::Vector3d T_c_w;
        cv2eigen(R_cv, tvec, R_c_w, T_c_w);

        Rs[i] = R_c_w.transpose();
        Ps[i] = -R_c_w.transpose() * T_c_w;
    }

    for (auto &it : feature_manager)
    {
        int feature_id = it.first;
        if (sfm_cloud.find(feature_id) != sfm_cloud.end())
        {
            Eigen::Vector3d P_w = sfm_cloud[feature_id];
            int start_f = it.second.start_frame;
            double depth = (Rs[start_f].transpose() * (P_w - Ps[start_f])).z();
            it.second.estimated_depth = 1.0 / depth;
        }
        else
        {
            it.second.estimated_depth = 1.0;
        }
    }

    std::cout << "[SfM 成功] 滑窗纯视觉初始结构构建完毕！" << std::endl;
    return true;
}

// ==========================================
// 4. 核心图优化与视觉残差计算
// ==========================================
void Estimator::optimization()
{
    std::cout << "[Ceres Optimizer] Constructing Non-linear Factor Graph Optimization..." << std::endl;

    ceres::Problem problem;
    ceres::LossFunction *loss_function = new ceres::HuberLoss(1.0);

    double para_Pose[WINDOW_SIZE + 1][7];

    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        para_Pose[i][0] = Ps[i].x();
        para_Pose[i][1] = Ps[i].y();
        para_Pose[i][2] = Ps[i].z();
        Eigen::Quaterniond q(Rs[i]);
        para_Pose[i][3] = q.x();
        para_Pose[i][4] = q.y();
        para_Pose[i][5] = q.z();
        para_Pose[i][6] = q.w();
        problem.AddParameterBlock(para_Pose[i], 7);
    }

    double para_Ex_Pose[7] = {TIC.x(), TIC.y(), TIC.z(), RIC(0, 0), RIC(1, 0), RIC(2, 0), 1.0};
    problem.AddParameterBlock(para_Ex_Pose, 7);
    problem.SetParameterBlockConstant(para_Ex_Pose);

    // 强行锁死滑窗第 0 帧位姿作为世界坐标系基准
    problem.SetParameterBlockConstant(para_Pose[0]);

    int visual_factor_cnt = 0;
    for (auto &it : feature_manager)
    {
        auto &feature = it.second;
        if (feature.feature_per_frame.size() < 2)
            continue;

        int i = feature.start_frame;
        Eigen::Vector2d pts_i = feature.feature_per_frame[0].point;

        for (size_t idx = 1; idx < feature.feature_per_frame.size(); idx++)
        {
            int j = i + idx;
            if (j > WINDOW_SIZE)
                continue;

            Eigen::Vector2d pts_j = feature.feature_per_frame[idx].point;

            // 调用 projection_factor.h 的重重投影误差代价函数
            ceres::CostFunction *f = ProjectionInverseDepthFactor::Create(pts_i, pts_j);
            problem.AddResidualBlock(f, loss_function, para_Pose[i], para_Pose[j], para_Ex_Pose, &feature.estimated_depth);
            visual_factor_cnt++;
        }
    }
    std::cout << "[Ceres Optimizer] Added " << visual_factor_cnt << " Visual Projection Factors into Graph." << std::endl;

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR; // Schur 消元极大加速纯视觉问题求解
    options.max_num_iterations = 10;
    options.max_solver_time_in_seconds = 0.04;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    // 优化状态写回
    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        Ps[i] = Eigen::Vector3d(para_Pose[i][0], para_Pose[i][1], para_Pose[i][2]);
        Eigen::Quaterniond q(para_Pose[i][6], para_Pose[i][3], para_Pose[i][4], para_Pose[i][5]);
        Rs[i] = q.toRotationMatrix();
    }
    std::cout << "[Ceres Optimizer] Optimization done. Trajectory updated. Current X: " << Ps[WINDOW_SIZE].x() << " Y: " << Ps[WINDOW_SIZE].y() << " Z: " << Ps[WINDOW_SIZE].z() << std::endl;
}

// ==========================================
// 5. 滑动窗口维护（边缘化老帧）
// ==========================================
void Estimator::slideWindow()
{
    std::cout << "[Slide Window] Marginalizing the oldest frame (Frame 0)..." << std::endl;

    for (int i = 0; i < WINDOW_SIZE; i++)
    {
        Headers[i] = Headers[i + 1];
        Ps[i] = Ps[i + 1];
        Rs[i] = Rs[i + 1];
    }

    Headers[WINDOW_SIZE] = 0.0;
    Ps[WINDOW_SIZE] = Ps[WINDOW_SIZE - 1];
    Rs[WINDOW_SIZE] = Rs[WINDOW_SIZE - 1];

    for (auto it = feature_manager.begin(); it != feature_manager.end();)
    {
        auto &feature = it->second;

        if (feature.start_frame == 0)
        {
            if (feature.feature_per_frame.size() > 1)
            {
                feature.feature_per_frame.erase(feature.feature_per_frame.begin());
                feature.start_frame = 0;
                it++;
            }
            else
            {
                it = feature_manager.erase(it);
            }
        }
        else
        {
            feature.start_frame--;
            it++;
        }
    }
}