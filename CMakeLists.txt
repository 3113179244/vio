cmake_minimum_required(VERSION 3.10) 
project(vio) 

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON) 

find_package(Eigen3 REQUIRED) 

# 寻找 OpenCV 库（前端光流与图像处理所需）
find_package(OpenCV REQUIRED)

# 寻找 Ceres 优化库（后端非线性优化所需）
find_package(Ceres REQUIRED)

# 寻找 Pangolin 库（3D 交互式可视化所需）
find_package(Pangolin REQUIRED)

find_package(Sophus REQUIRED)
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include 
    ${EIGEN3_INCLUDE_DIR} 
    ${OpenCV_INCLUDE_DIRS}
    ${CERES_INCLUDE_DIRS}
    ${PANGOLIN_INCLUDE_DIRS}
    ${Sophus_INCLUDE_DIRS}
)

add_library(${PROJECT_NAME}_lib STATIC

    # ==== 视觉前端模块 (Front-end) ====
    src/feature_tracker.cpp    # 负责处理图像、提取 KLT 角点并使用光流法跨帧追踪特征点

    # ==== 惯性数据预处理 (IMU Pre-integration) ====
    src/imu_factor.cpp         # 负责两帧图像间的IMU数据预积分、计算雅可比矩阵、协方差及Ceres残差定义

    # ==== 数据管理与时间戳同步 (Data Hub) ====
    src/data_manager.cpp       # 负责线程安全的 IMU/图像 队列管理，并替代 ROS 实现高低频数据的时间戳对齐

    # ==== 后端优化估计器 (Back-end Estimator) ====
    src/estimator.cpp          # VIO的主大脑，负责管理滑动窗口、构建并求解 Ceres 非线性优化问题（融合视觉与IMU）

    # ==== 系统在线初始化模块 (Initialization) ====
    src/initial_sfm.cpp        # 负责在没有尺度时，通过纯视觉特征点建立局部三维点云和滑窗内相机轨迹 (SfM)

    src/initial_alignment.cpp  # 负责将纯视觉 SfM 轨迹与 IMU 预积分对齐，求解系统绝对尺度、初始速度和重力方向

    # ==== 配置参数与通用工具 (System Utilities) ====
    src/parameters.cpp         # 负责使用 OpenCV 的 FileStorage 解析 yaml 配置文件，加载系统全局硬件及算法参数
    
    src/utility.cpp            # 负责存放通用的数学与坐标转换函数（如四元数、旋转矩阵、李代数、时间戳转换等）

    # ==== 3.D 交互可视化模块 (Visualization) ====
    src/viewer.cpp             # 负责启动独立绘图线程，利用 Pangolin 库实时绘制相机运动轨迹和三维地图点云
)

add_executable(main src/main.cpp) 

target_link_libraries(main 
    ${PROJECT_NAME}_lib
    pthread 
    ${OpenCV_LIBS}
    ${CERES_LIBRARIES}
    ${PANGOLIN_LIBRARIES}
    Sophus::Sophus
)