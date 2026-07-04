#include "parameters.h"
#include <fstream>
#include <sstream>

// 定义所有全局变量
std::string IMU_TOPIC = "";
std::string IMAGE_TOPIC = "";
std::string OUTPUT_PATH = "";

std::string MODEL_TYPE = "";
std::string CAMERA_NAME = "";
int IMAGE_WIDTH = 0;
int IMAGE_HEIGHT = 0;

double FX = 0.0, FY = 0.0, CX = 0.0, CY = 0.0;
double K2 = 0.0, K3 = 0.0, K4 = 0.0, K5 = 0.0;

int ESTIMATE_EXTRINSIC = 0;
Eigen::Matrix3d RIC = Eigen::Matrix3d::Identity();
Eigen::Vector3d TIC = Eigen::Vector3d::Zero();

int MAX_CNT = 0;
int MIN_DIST = 0;
int WINDOW_SIZE = 0;
double FREQ = 0.0;
double F_THRESHOLD = 0.0;
int SHOW_TRACK = 0;
int EQUALIZE = 0;

double MAX_SOLVER_TIME = 0.0;
int MAX_NUM_ITERATIONS = 0;
double KEYFRAME_PARALLAX = 0.0;

double ACC_N = 0.0, GYR_N = 0.0;
double ACC_W = 0.0, GYR_W = 0.0;
double G_NORM = 0.0;

int LOOP_CLOSURE = 0;
int LOAD_PREVIOUS_POSE_GRAPH = 0;
int FAST_RELOCALIZATION = 0;
std::string POSE_GRAPH_SAVE_PATH = "";

int ESTIMATE_TD = 0;
double TD = 0.0;

int ROLLING_SHUTTER = 0;
double ROLLING_SHUTTER_TR = 0.0;

int SAVE_IMAGE = 0;
int VISUALIZE_IMU_FORWARD = 0;
int VISUALIZE_CAMERA = 0;

bool readParameters(const std::string& config_file_path) {
    std::cout << "[Parameters] Loading config file: " << config_file_path << std::endl;

    cv::FileStorage fs;
    try {
        fs.open(config_file_path, cv::FileStorage::READ);
    } catch (const cv::Exception& e) {
        std::cerr << "[Error] OpenCV FileStorage open failed: " << e.what() << std::endl;
        return false;
    }

    if (!fs.isOpened()) {
        std::cerr << "[Error] Failed to open YAML configuration file at: " << config_file_path << std::endl;
        return false;
    }

    // 1. 基础公共参数 / 话题 / 路径
    IMU_TOPIC = (std::string)fs["imu_topic"];
    IMAGE_TOPIC = (std::string)fs["image_topic"];
    OUTPUT_PATH = (std::string)fs["output_path"];

    // 2. 相机标定与基础信息
    MODEL_TYPE = (std::string)fs["model_type"];
    CAMERA_NAME = (std::string)fs["camera_name"];
    IMAGE_WIDTH = (int)fs["image_width"];
    IMAGE_HEIGHT = (int)fs["image_height"];

    // 3. 嵌套读取投影参数 (内参和畸变)
    cv::FileNode proj_params = fs["projection_parameters"];
    if (!proj_params.empty()) {
        K2 = (double)proj_params["k2"];
        K3 = (double)proj_params["k3"];
        K4 = (double)proj_params["k4"];
        K5 = (double)proj_params["k5"];
        FX = (double)proj_params["mu"];
        FY = (double)proj_params["mv"];
        CX = (double)proj_params["u0"];
        CY = (double)proj_params["v0"];
    } else {
        std::cerr << "[Warning] 'projection_parameters' node not found!" << std::endl;
    }

    // 4. 相机-IMU 外参
    ESTIMATE_EXTRINSIC = (int)fs["estimate_extrinsic"];
    if (ESTIMATE_EXTRINSIC == 0 || ESTIMATE_EXTRINSIC == 1) {
        cv::Mat cv_R, cv_T;
        fs["extrinsicRotation"] >> cv_R;
        fs["extrinsicTranslation"] >> cv_T;

        if (!cv_R.empty() && !cv_T.empty() && cv_R.rows == 3 && cv_R.cols == 3 && cv_T.rows == 3 && cv_T.cols == 1) {
            for (int i = 0; i < 3; i++) {
                TIC(i) = cv_T.at<double>(i, 0);
                for (int j = 0; j < 3; j++) {
                    RIC(i, j) = cv_R.at<double>(i, j);
                }
            }
        } else {
            std::cerr << "[Warning] Extrinsic matrix is empty or has incorrect dimensions!" << std::endl;
        }
    }

    // 5. 特征提取与追踪参数
    MAX_CNT = (int)fs["max_cnt"];
    MIN_DIST = (int)fs["min_dist"];
    WINDOW_SIZE = (int)fs["window_size"];
    FREQ = (double)fs["freq"];
    F_THRESHOLD = (double)fs["F_threshold"];
    SHOW_TRACK = (int)fs["show_track"];
    EQUALIZE = (int)fs["equalize"];

    // 6. 优化控制参数
    MAX_SOLVER_TIME = (double)fs["max_solver_time"];
    MAX_NUM_ITERATIONS = (int)fs["max_num_iterations"];
    KEYFRAME_PARALLAX = (double)fs["keyframe_parallax"];

    // 7. IMU 噪声与重力参数
    ACC_N = (double)fs["acc_n"];
    GYR_N = (double)fs["gyr_n"];
    ACC_W = (double)fs["acc_w"];
    GYR_W = (double)fs["gyr_w"];
    G_NORM = (double)fs["g_norm"];

    // 8. 回环检测参数
    LOOP_CLOSURE = (int)fs["loop_closure"];
    LOAD_PREVIOUS_POSE_GRAPH = (int)fs["load_previous_pose_graph"]; 
    FAST_RELOCALIZATION = (int)fs["fast_relocalization"];
    POSE_GRAPH_SAVE_PATH = (std::string)fs["pose_graph_save_path"];
    FAST_RELOCALIZATION = (int)fs["fast_relocalization"];
    POSE_GRAPH_SAVE_PATH = (std::string)fs["pose_graph_save_path"];

    // 9. 在线时间同步参数
    ESTIMATE_TD = (int)fs["estimate_td"];
    TD = (double)fs["td"];

    // 10. 快门参数
    ROLLING_SHUTTER = (int)fs["rolling_shutter"];
    ROLLING_SHUTTER_TR = (double)fs["rolling_shutter_tr"];

    // 11. 可视化与调试参数
    SAVE_IMAGE = (int)fs["save_image"];
    VISUALIZE_IMU_FORWARD = (int)fs["visualize_imu_forward"];
    VISUALIZE_CAMERA = (int)fs["visualize_camera"];

    std::cout << "[Parameters] All configuration parameters loaded successfully." << std::endl;
    return true;
}