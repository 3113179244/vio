#include "estimator.h"
#include "projection_factor.h"
#include <ceres/ceres.h>
#include <Eigen/Dense>
#include <iostream>
#include <map>

// 辅助函数：三维向量转反对称矩阵（虽然纯视觉不用，但保留以维持结构完整）
Eigen::Matrix3d skewSymmetric(const Eigen::Vector3d &v)
{
    Eigen::Matrix3d m;
    m << 0, -v.z(), v.y(),
         v.z(),    0, -v.x(),
        -v.y(),  v.x(),    0;
    return m;
}

// 构造函数：初始化滑窗内状态量
Estimator::Estimator()
{
    std::cout << "[Estimator] 正在初始化纯视觉滑窗后端估计器..." << std::endl;
    solver_flag = INITIAL;
    frame_count = 0;

    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        Ps[i].setZero();
        Rs[i].setIdentity();
    }
    feature_manager.clear();
}

// 后端处理数据包的总入口
void Estimator::processMeasurement(const MeasurementPackage &package)
{
    std::cout << "\n[Backend] ====== 开始处理同步数据包 ======" << std::endl;
    
    // 纯视觉模式下，跳过 processIMU 递推，直接推进视觉状态机
    processImage(package.image.feature_points, package.image.timestamp);
}

// 视觉状态机更新：管理滑窗帧数，触发优化与滑窗边缘化
void Estimator::processImage(const std::map<int, Eigen::Vector2d> &image_msg, double header_time)
{
    std::cout << "[Backend] 当前帧追踪到的特征点数量: " << image_msg.size() << std::endl;

    // 1. 将当前帧的特征点观测信息，录入到全局特征管理器（Feature Manager）中
    int current_frame_idx = (frame_count < WINDOW_SIZE) ? frame_count : WINDOW_SIZE;
    for (const auto &pts : image_msg)
    {
        int feature_id = pts.first;
        Eigen::Vector2d point_uv = pts.second; // 已经是前端去畸变后的归一化平面坐标

        if (feature_manager.find(feature_id) == feature_manager.end())
        {
            // 如果是新出现的特征点，为其创建管理器节点
            FeaturePerId new_feature;
            new_feature.feature_id = feature_id;
            new_feature.start_frame = current_frame_idx; // 记录它的起始帧
            new_feature.estimated_depth = 1.0;          // 初始深度默认设为 1.0
            new_feature.feature_per_frame.push_back({point_uv});
            feature_manager[feature_id] = new_feature;
        }
        else
        {
            // 如果是老特征点，追加当前帧的观测记录
            feature_manager[feature_id].feature_per_frame.push_back({point_uv});
        }
    }

    // 2. 判断滑动窗口是否已满
    if (frame_count < WINDOW_SIZE)
    {
        Headers[frame_count] = header_time;
        frame_count++;
        std::cout << "[Backend] 窗口未满，正在收集帧数... 当前进度: " << frame_count << "/" << WINDOW_SIZE << std::endl;
    }
    else
    {
        Headers[WINDOW_SIZE] = header_time;

        // 如果系统还未建立初始结构，先进行简单初始化
        if (solver_flag == INITIAL)
        {
            std::cout << "[Backend] 窗口已满，触发系统初始化..." << std::endl;
            if (initialStructure())
            {
                solver_flag = NON_LINEAR; // 切换至正规非线性优化状态
                std::cout << "[Backend] 系统初始化成功，进入非线性优化阶段！" << std::endl;
            }
            slideWindow(); // 边缘化老帧，腾出空间
        }
        else
        {
            // 核心步骤：进入基于视觉残差的非线性图优化
            optimization();
            // 优化完成后，滑动窗口，剔除或平移老状态
            slideWindow();
        }
    }
}

// 简化的纯视觉初始化结构
bool Estimator::initialStructure()
{
    if (frame_count >= WINDOW_SIZE)
    {
        // 纯视觉在没有IMU尺度时存在尺度不确定性，此处先默认位姿为初始递推值
        return true;
    }
    return false;
}

// =========================================================================
// 核心函数：构建非线性最小二乘问题，计算视觉残差并求解更新
// =========================================================================
void Estimator::optimization()
{
    std::cout << "[Optimizer] >>>>>>>> 开始构建纯视觉因子图优化 <<<<<<<<" << std::endl;

    ceres::Problem problem;
    
    // 创建 Huber 鲁棒核函数，用于剔除前端光流追踪的误匹配点（Outliers）
    ceres::LossFunction *loss_function = new ceres::HuberLoss(1.0);

    // 1. 将 Eigen 矩阵和四元数的状态量转为 Ceres 接收的 double 数组参数块
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
        
        // 向 Ceres 添加滑窗内第 i 帧的位姿参数块
        problem.AddParameterBlock(para_Pose[i], 7);
    }

    // 2. 将外参（Camera 到 IMU 的 R 和 T）转为参数块，并将其设置为【常量】不参与优化
    double para_Ex_Pose[7] = {TIC.x(), TIC.y(), TIC.z(), RIC(0, 0), RIC(1, 0), RIC(2, 0), 1.0};
    problem.AddParameterBlock(para_Ex_Pose, 7);
    problem.SetParameterBlockConstant(para_Ex_Pose);

    // 3. 固定滑窗的第 0 帧位姿为【绝对基准】（不进行优化），防止全局轨迹整体发生无约束漂移
    problem.SetParameterBlockConstant(para_Pose[0]);

    // 4. 遍历特征点管理器，计算并注入所有视觉重投影残差因子
    int visual_factor_cnt = 0;
    for (auto &it : feature_manager)
    {
        auto &feature = it.second;
        
        // 如果一个特征点被滑窗内观测到的次数小于 2 次（即无法构成三角化约束），则跳过
        if (feature.feature_per_frame.size() < 2)
            continue;

        // 获取该特征点第一次被看到的起始帧索引
        int i = feature.start_frame;
        // 提取参考帧 i 对应的归一化像素坐标 pts_i
        Eigen::Vector2d pts_i = feature.feature_per_frame[0].point;

        // 遍历该特征点在后续每一帧 j 中的观测记录，分别与参考帧 i 建立视觉重投影残差
        for (size_t idx = 1; idx < feature.feature_per_frame.size(); idx++)
        {
            int j = i + idx;
            if (j > WINDOW_SIZE)
                continue;

            // 提取当前帧 j 观测到的去畸变归一化像素坐标 pts_j
            Eigen::Vector2d pts_j = feature.feature_per_frame[idx].point;

            // 利用我们在 projection_factor.h 中实现的静态工厂方法创建视觉残差代价函数
            // 该代价函数内部会通过逆深度反投影、空间坐标系变换、再重投影，最终计算出 residuals 
            ceres::CostFunction *f = ProjectionInverseDepthFactor::Create(pts_i, pts_j);
            
            // 将视觉重投影残差块加入 Ceres 求解器
            // 关联的优化变量有：参考帧位姿、当前帧位姿、外参、以及该特征点的逆深度
            problem.AddResidualBlock(f, loss_function, para_Pose[i], para_Pose[j], para_Ex_Pose, &feature.estimated_depth);
            visual_factor_cnt++;
        }
    }
    std::cout << "[Optimizer] 成功向非线性图优化中注入 " << visual_factor_cnt << " 个视觉重投影约束因子。" << std::endl;

    // 5. 配置 Ceres 求解器选项并调用 Solve 进行迭代优化
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR; // 采用 Schur 消元法（Schur Complement）优先消去深度，极大加速求解
    options.max_num_iterations = 10;                // 设置最大迭代次数
    options.max_solver_time_in_seconds = 0.04;      // 限制单次最大求解时间，确保实时性

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    // 6. 优化结束，将优化后的最新参数块写回至后端核心状态量中，更新相机运动轨迹
    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        Ps[i] = Eigen::Vector3d(para_Pose[i][0], para_Pose[i][1], para_Pose[i][2]);
        Eigen::Quaterniond q(para_Pose[i][6], para_Pose[i][3], para_Pose[i][4], para_Pose[i][5]);
        Rs[i] = q.toRotationMatrix();
    }
    std::cout << "[Optimizer] 视觉非线性迭代完成。最新滑窗尾部相机位置 -> X: " 
              << Ps[WINDOW_SIZE].x() << " Y: " << Ps[WINDOW_SIZE].y() << " Z: " << Ps[WINDOW_SIZE].z() << std::endl;
}

// =========================================================================
// 滑动窗口边缘化：剔除最老帧（Frame 0），保证系统内存和计算量的恒定
// =========================================================================
void Estimator::slideWindow()
{
    std::cout << "[Slide Window] 正在边缘化最老帧 (Frame 0) 以维持实时性..." << std::endl;

    // 1. 滑动窗口内的相机状态量整体向前平移一位（扔掉第 0 帧）
    for (int i = 0; i < WINDOW_SIZE; i++)
    {
        Headers[i] = Headers[i + 1];
        Ps[i] = Ps[i + 1];
        Rs[i] = Rs[i + 1];
    }

    // 2. 填充并复位滑窗尾部的空闲位置
    Headers[WINDOW_SIZE] = 0.0;
    Ps[WINDOW_SIZE] = Ps[WINDOW_SIZE - 1];
    Rs[WINDOW_SIZE] = Rs[WINDOW_SIZE - 1];

    // 3. 核心步骤：维护地图特征点的生命周期与观测树结构
    for (auto it = feature_manager.begin(); it != feature_manager.end();)
    {
        auto &feature = it->second;

        // 如果该点的起始帧是已经被剔除的第 0 帧
        if (feature.start_frame == 0)
        {
            // 如果它不仅在第 0 帧被看到，在后面的帧也留有观测记录
            if (feature.feature_per_frame.size() > 1)
            {
                // 抹去它在第 0 帧的观测，观测队列整体前移
                feature.feature_per_frame.erase(feature.feature_per_frame.begin());
                // 此时它在当前新滑窗中的起始帧依然变成了新的 0 帧
                feature.start_frame = 0;
                it++;
            }
            else
            {
                // 如果它只在老第 0 帧被看到过，说明已经滑出窗口且无后续观测，直接从内存中删除
                it = feature_manager.erase(it);
            }
        }
        else
        {
            // 如果该点的起始帧不在第 0 帧，因为整个窗口前移了，其起始帧索引需要自减 1
            feature.start_frame--;
            it++;
        }
    }
}