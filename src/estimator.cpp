#include "estimator.h"
#include "projection_factor.h"
#include "imu_factor.h"
#include <ceres/ceres.h>
#include "utility.h"

// 三维向量转反对称矩阵的辅助函数
Eigen::Matrix3d skewSymmetric(const Eigen::Vector3d &v)
{
    Eigen::Matrix3d m;
    m << 0, -v.z(), v.y(),
        v.z(), 0, -v.x(),
        -v.y(), v.x(), 0;
    return m;
}

Estimator::Estimator()
{
    std::cout << "[Estimator] Initializing VIO Backend Estimator..." << std::endl;
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

    // 初始化预积分缓存队列
    for (int i = 0; i < WINDOW_SIZE; i++)
    {
        dt_buf[i] = 0.0;
        delta_p_buf[i].setZero();
        delta_v_buf[i].setZero();
        delta_q_buf[i].setIdentity();
    }

    feature_manager.clear();
}

void Estimator::processMeasurement(const MeasurementPackage &package)
{
    std::cout << "\n[Backend Estimator] ====== Process New Sync Packet ======" << std::endl;
    std::cout << "[Backend Estimator] Image Frame Timestamp: " << std::fixed << std::setprecision(6) << package.image.timestamp << " s" << std::endl;
    std::cout << "[Backend Estimator] Linked IMU counts between frames: " << package.imus.size() << std::endl;

    // 1. 初始化当前两帧之间的局部预积分增量量
    Eigen::Vector3d delta_p = Eigen::Vector3d::Zero();
    Eigen::Vector3d delta_v = Eigen::Vector3d::Zero();
    Eigen::Quaterniond delta_q = Eigen::Quaterniond::Identity();
    double dt_sum = 0.0;

    int current_frame_idx = (frame_count < WINDOW_SIZE) ? frame_count : WINDOW_SIZE;

    // 2. 循环处理并累积两帧之间的原始高频惯性数据
    for (const auto &imu : package.imus)
    {
        double dt = 0.005;
        if (!first_imu)
            dt = imu.timestamp - last_imu_time;
        else
            first_imu = false;
        last_imu_time = imu.timestamp;

        // 全局递推（用于提供优化的状态初始迭代值）
        processIMU(dt, imu.acc, imu.gyr);

        // 局部增量预积分计算
        Eigen::Vector3d un_gyr = imu.gyr - Bgs[current_frame_idx];
        Eigen::Vector3d un_acc = imu.acc - Bas[current_frame_idx];

        dt_sum += dt;
        delta_q = delta_q * utility::quaternionExpMap(un_gyr * dt);

        // 依照动力学方程在中值进行局部测量累积
        delta_v += delta_q * un_acc * dt;
        delta_p += delta_v * dt + (delta_q * un_acc) * (0.5 * dt * dt);
    }

    // 3. 把这一段通过高频 IMU 真实算出来的核心物理约束，缓存到队列中
    if (frame_count > 0)
    {
        int buf_idx = (frame_count < WINDOW_SIZE) ? (frame_count - 1) : (WINDOW_SIZE - 1);
        dt_buf[buf_idx] = dt_sum;
        delta_p_buf[buf_idx] = delta_p;
        delta_v_buf[buf_idx] = delta_v;
        delta_q_buf[buf_idx] = delta_q;
    }

    // 4. 推进视觉状态机并触发正规优化
    processImage(package.image.feature_points, package.image.timestamp);
}

void Estimator::processIMU(double dt, const Eigen::Vector3d &linear_acceleration, const Eigen::Vector3d &angular_velocity)
{
    if (solver_flag == INITIAL)
    {
        int i = frame_count;
        Eigen::Vector3d un_gyr = angular_velocity - Bgs[i];
        Rs[i] = Rs[i] * (Eigen::Matrix3d::Identity() + skewSymmetric(un_gyr) * dt);
        Eigen::Vector3d un_acc = Rs[i] * (linear_acceleration - Bas[i]) + Eigen::Vector3d(0, 0, -9.80766);
        Ps[i] = Ps[i] + Vs[i] * dt + 0.5 * un_acc * dt * dt;
        Vs[i] = Vs[i] + un_acc * dt;
    }
}

void Estimator::processImage(const std::map<int, Eigen::Vector2d> &image_msg, double header_time)
{
    std::cout << "[Backend Estimator] Feature points count in this frame: " << image_msg.size() << std::endl;

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
            new_feature.estimated_depth = 1.0;
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
        std::cout << "[Backend Estimator] Scaling Slide Window... Current frame count: " << frame_count << "/" << WINDOW_SIZE << std::endl;
    }
    else
    {
        Headers[WINDOW_SIZE] = header_time;

        if (solver_flag == INITIAL)
        {
            std::cout << "[Backend Estimator] Window is FULL. Trying to Initialize..." << std::endl;
            if (initialStructure())
            {
                solver_flag = NON_LINEAR;
                std::cout << "[Backend Estimator] SUCCESS! VIO Backend System Initialized!" << std::endl;
            }
            slideWindow();
        }
        else
        {
            optimization();
            slideWindow();
        }
    }
}

bool Estimator::initialStructure()
{
    if (frame_count >= WINDOW_SIZE)
    {
        for (int i = 0; i <= WINDOW_SIZE; ++i)
        {
            Vs[i] = Eigen::Vector3d(0.0, 0.0, 0.0);
        }
        return true;
    }
    return false;
}

void Estimator::optimization()
{
    std::cout << "[Ceres Optimizer] Constructing Non-linear Factor Graph Optimization..." << std::endl;

    ceres::Problem problem;
    ceres::LossFunction *loss_function = new ceres::HuberLoss(1.0);

    // 1. 参数块格式化转换
    double para_Pose[WINDOW_SIZE + 1][7];
    double para_Speed[WINDOW_SIZE + 1][3];

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

        para_Speed[i][0] = Vs[i].x();
        para_Speed[i][1] = Vs[i].y();
        para_Speed[i][2] = Vs[i].z();
        problem.AddParameterBlock(para_Speed[i], 3);
    }

    double para_Ex_Pose[7] = {TIC.x(), TIC.y(), TIC.z(), RIC(0, 0), RIC(1, 0), RIC(2, 0), 1.0};
    problem.AddParameterBlock(para_Ex_Pose, 7);
    problem.SetParameterBlockConstant(para_Ex_Pose);

    // 💡【核心基准固定】：强行将滑窗第 0 帧锁死在基准原点，作为基准参照物防止整体漂移坍塌
    problem.SetParameterBlockConstant(para_Pose[0]);

    // 2. 💡【终极修改】：注入相邻两帧间最真实的惯性运动因子
    int imu_factor_cnt = 0;
    for (int i = 0; i < WINDOW_SIZE; i++)
    {
        int j = i + 1;

        // 从类缓冲区中提取当前时间包内真实的 IMU 积分度量值
        double dT = dt_buf[i];
        Eigen::Vector3d real_dp = delta_p_buf[i];
        Eigen::Vector3d real_dv = delta_v_buf[i];
        Eigen::Quaterniond real_dq = delta_q_buf[i];

        // 引入由真实物理增量生成的 IMU 约束因子
        ceres::CostFunction *imu_f = IMUFactor::Create(real_dp, real_dv, real_dq, dT);
        problem.AddResidualBlock(imu_f, nullptr, para_Pose[i], para_Speed[i], para_Pose[j], para_Speed[j]);
        imu_factor_cnt++;
    }
    std::cout << "[Ceres Optimizer] Added " << imu_factor_cnt << " REAL IMU Pre-integration Factors into Graph." << std::endl;

    // 3. 注入视觉重投影因子
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

            ceres::CostFunction *f = ProjectionInverseDepthFactor::Create(pts_i, pts_j);
            problem.AddResidualBlock(f, loss_function, para_Pose[i], para_Pose[j], para_Ex_Pose, &feature.estimated_depth);
            visual_factor_cnt++;
        }
    }
    std::cout << "[Ceres Optimizer] Added " << visual_factor_cnt << " Visual Projection Factors into Graph." << std::endl;

    // 4. 配置并求解
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.max_num_iterations = 10;
    options.max_solver_time_in_seconds = 0.04;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    // 5. 状态写回
    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        Ps[i] = Eigen::Vector3d(para_Pose[i][0], para_Pose[i][1], para_Pose[i][2]);
        Eigen::Quaterniond q(para_Pose[i][6], para_Pose[i][3], para_Pose[i][4], para_Pose[i][5]);
        Rs[i] = q.toRotationMatrix();

        Vs[i] = Eigen::Vector3d(para_Speed[i][0], para_Speed[i][1], para_Speed[i][2]);
    }
    std::cout << "[Ceres Optimizer] Optimization done. Trajectory updated. Current X: " << Ps[WINDOW_SIZE].x() << " Y: " << Ps[WINDOW_SIZE].y() << " Z: " << Ps[WINDOW_SIZE].z() << std::endl;
}

void Estimator::slideWindow()
{
    std::cout << "[Slide Window] Marginalizing the oldest frame (Frame 0) to maintain real-time execution..." << std::endl;

    // 1. 滑窗状态数组与预积分增量缓存同步整体前移一位
    for (int i = 0; i < WINDOW_SIZE; i++)
    {
        Headers[i] = Headers[i + 1];
        Ps[i] = Ps[i + 1];
        Vs[i] = Vs[i + 1];
        Rs[i] = Rs[i + 1];
        Bas[i] = Bas[i + 1];
        Bgs[i] = Bgs[i + 1];

        // 💡【核心同步】：预积分测量容器平移
        if (i < WINDOW_SIZE - 1)
        {
            dt_buf[i] = dt_buf[i + 1];
            delta_p_buf[i] = delta_p_buf[i + 1];
            delta_v_buf[i] = delta_v_buf[i + 1];
            delta_q_buf[i] = delta_q_buf[i + 1];
        }
    }

    Headers[WINDOW_SIZE] = 0.0;
    Ps[WINDOW_SIZE] = Ps[WINDOW_SIZE - 1];
    Vs[WINDOW_SIZE] = Vs[WINDOW_SIZE - 1];
    Rs[WINDOW_SIZE] = Rs[WINDOW_SIZE - 1];
    Bas[WINDOW_SIZE] = Bas[WINDOW_SIZE - 1];
    Bgs[WINDOW_SIZE] = Bgs[WINDOW_SIZE - 1];

    // 清洗并重置滑窗尾部的预积分项
    dt_buf[WINDOW_SIZE - 1] = 0.0;
    delta_p_buf[WINDOW_SIZE - 1].setZero();
    delta_v_buf[WINDOW_SIZE - 1].setZero();
    delta_q_buf[WINDOW_SIZE - 1].setIdentity();

    // 2. 更新特征点管理器生命周期
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