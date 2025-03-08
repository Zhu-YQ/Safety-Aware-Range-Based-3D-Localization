#include <deque>
#include <fstream>
#include <iostream>

#include <Eigen/Core>
#include <opencv2/opencv.hpp>

#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <nav_msgs/Path.h>
#include <ros/ros.h>

#include "filter/ekf_trans.hpp"
#include "uwb_loc/BAResult.h"
#include "uwb_loc/Frame.h"
#include "uwb_loc/MyUWB.h"

// std::queue<uwb_loc::MyUWB> uwb_msg_queue;
std::deque<uwb_loc::MyUWB> uwb_msg_queue;
void UWBMsgCallback(const uwb_loc::MyUWB& msg)
{
    // uwb_msg_queue.push(msg);
    uwb_msg_queue.push_back(msg);
}

std::vector<Eigen::Vector3d> aps_vec;
std::vector<double> bias_vec;
void BAResultMsgCallback(const uwb_loc::BAResult& msg)
{
    if (aps_vec.empty()) {
        for (int i = 0; i < msg.num_anchors; i++) {
            aps_vec.emplace_back(0, 0, 0);
            bias_vec.push_back(0);
        }
    }
    for (int i = 0; i < msg.num_anchors; i++) {
        double x = msg.aps_arr[i * 3 + 0];
        double y = msg.aps_arr[i * 3 + 1];
        double z = msg.aps_arr[i * 3 + 2];
        Eigen::Vector3d q_i(x, y, z);
        aps_vec.at(i) = q_i;
        bias_vec.at(i) = msg.bias_arr[i];
    }
}

geometry_msgs::PoseWithCovarianceStamped safety_bound;
void BoundingResultCallback(const geometry_msgs::PoseWithCovarianceStamped& msg)
{
    safety_bound = msg;
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "tracking_node");
    ros::NodeHandle nh;

    std::string config_file_path;
    nh.param<std::string>("config_file_path", config_file_path, "");
    // ROS_INFO_STREAM("using config file: " << config_file_path << '\n');

    /// read config
    cv::FileStorage fs_read(config_file_path, cv::FileStorage::READ);

    // subscribe sensor msgs
    std::string uwb_topic;
    fs_read["uwb_topic"] >> uwb_topic;
    ros::Subscriber uwb_sub = nh.subscribe(uwb_topic, 2000, UWBMsgCallback);
    // subscribe ba results
    ros::Subscriber ba_result_sub = nh.subscribe("/ba_result", 1, BAResultMsgCallback);
    // subscribe bounding results
    ros::Subscriber bounding_result_sub = nh.subscribe("/safety_bound", 10, BoundingResultCallback);

    // pubilsh results
    ros::Publisher frame_pub
        = nh.advertise<uwb_loc::Frame>("/frame", 20);
    ros::Publisher path_pub = nh.advertise<nav_msgs::Path>("/esti_path", 20);

    // save results
    std::string save_path;
    fs_read["save_path"] >> save_path;

    std::string traj_save_path = save_path + "/esti_traj.txt";
    std::ofstream traj_save_fs;
    traj_save_fs.open(traj_save_path, std::ios::out);
    traj_save_fs.setf(std::ios::fixed, std::ios::floatfield);
    traj_save_fs.precision(16);

    std::string cov_save_path = save_path + "/esti_cov.txt";
    std::ofstream cov_save_fs;
    cov_save_fs.open(cov_save_path, std::ios::out);
    cov_save_fs.setf(std::ios::fixed, std::ios::floatfield);
    cov_save_fs.precision(16);

    // params of filter
    int num_anchors = 0;
    fs_read["num_anchors"] >> num_anchors;

    double nlos_thresh = 0;
    fs_read["nlos_thresh"] >> nlos_thresh;

    double initial_robot_px_var = 0;
    fs_read["initial_robot_px_var"] >> initial_robot_px_var;
    double initial_robot_py_var = 0;
    fs_read["initial_robot_py_var"] >> initial_robot_py_var;
    double initial_robot_pz_var = 0;
    fs_read["initial_robot_pz_var"] >> initial_robot_pz_var;

    double initial_robot_v_var = 0;
    fs_read["initial_robot_v_var"] >> initial_robot_v_var;
    double initial_robot_a_var = 0;
    fs_read["initial_robot_a_var"] >> initial_robot_a_var;

    double motion_robot_jx_var = 0;
    fs_read["motion_robot_jx_var"] >> motion_robot_jx_var;
    double motion_robot_jy_var = 0;
    fs_read["motion_robot_jy_var"] >> motion_robot_jy_var;
    double motion_robot_jz_var = 0;
    fs_read["motion_robot_jz_var"] >> motion_robot_jz_var;

    double tag_height = 0;
    fs_read["tag_height"] >> tag_height;

    double noise_var = 0;
    fs_read["noise_var"] >> noise_var;

    /// core part
    EKFtrans ekf(num_anchors, nlos_thresh, initial_robot_px_var, initial_robot_py_var,
        initial_robot_pz_var, initial_robot_v_var, initial_robot_a_var);

    nav_msgs::Path path;
    path.header.frame_id = "world";

    double last_stamp = 0;
    // const double DT_THRESH = 0.02; // 50Hz
    const double DT_THRESH = 0.05; // 20Hz
    // const double DT_THRESH = 0.1; // 10Hz

    bool init_done = false;
    const int INIT_Y_THRESH = 10;
    std::vector<Eigen::VectorXd> y_for_init_vec;

    double sum_time = 0;
    int num_sum = 0;
    while (ros::ok()) {
        ros::spinOnce();

        if (uwb_msg_queue.empty() || aps_vec.empty()) {
            continue;
        }

        // get data
        auto uwb_msg = uwb_msg_queue.front();
        uwb_msg_queue.pop_front();

        auto dis_arr = uwb_msg.dis_arr;
        Eigen::VectorXd y;
        y.setZero(num_anchors, 1);
        for (size_t i = 0; i < num_anchors; i++) {
            y(i) = dis_arr.at(i);
        }

        // uwb_loc::MyUWB uwb_msg = uwb_msg_queue[8];
        // Eigen::VectorXd y;
        // y.setZero(num_anchors, 1);
        // std::vector<std::vector<double>> dis_arr_win;
        // for (size_t j = 0; j < num_anchors; j++) {
        //     dis_arr_win.push_back({});
        //     for (size_t i = 0; i < 9; i++) {
        //         dis_arr_win[j].push_back(uwb_msg_queue[i].dis_arr[j]);
        //     }
        //     std::sort(dis_arr_win[j].begin(), dis_arr_win[j].end());
        //     y(j) = dis_arr_win[j][4];
        // }
        // uwb_msg_queue.pop_front();

        // initialization
        if (!init_done) {
            y_for_init_vec.push_back(y);
            if (y_for_init_vec.size() < INIT_Y_THRESH) {
                continue;
            }

            Eigen::VectorXd y_mean;
            y_mean.setZero(y.rows(), y.cols());
            for (auto& y_i : y_for_init_vec) {
                y_mean += y_i;
            }
            y_mean /= INIT_Y_THRESH;
            ekf.initTagPosition(y_mean, tag_height, aps_vec);

            init_done = true;
        }

        // limit sampling interval
        double stamp = uwb_msg.stamp;

        if (last_stamp == 0) {
            last_stamp = stamp;
        }
        if (stamp - last_stamp < DT_THRESH) {
            continue;
        }

        double dt = stamp - last_stamp;
        last_stamp = stamp;

        // bias compensation
        for (int i = 0; i < num_anchors; i++) {
            y(i) /= (1 + bias_vec.at(i));
        }

        const double start_time = ros::Time::now().toSec();

        ekf.predict(dt, motion_robot_jx_var, motion_robot_jy_var, motion_robot_jz_var);
        ekf.update_seq(y, noise_var, aps_vec);

        const double end_time = ros::Time::now().toSec();
        sum_time += end_time - start_time;
        num_sum++;

        const Eigen::Vector3d robot_position = ekf.getTagPosition();
        const auto robot_position_bound = ekf.getTagPositionBound();
        const auto robot_position_P = ekf.getTagPositionCov();

        // check safety
        Eigen::Vector3d safety_bound_center;
        safety_bound_center << safety_bound.pose.pose.position.x, safety_bound.pose.pose.position.y, safety_bound.pose.pose.position.z;
        Eigen::Matrix3d safety_bound_extrema;
        safety_bound_extrema(0, 0) = safety_bound.pose.covariance[0];
        safety_bound_extrema(0, 1) = safety_bound_extrema(1, 0) = safety_bound.pose.covariance[1];
        safety_bound_extrema(0, 2) = safety_bound_extrema(2, 0) = safety_bound.pose.covariance[2];
        safety_bound_extrema(1, 1) = safety_bound.pose.covariance[7];
        safety_bound_extrema(1, 2) = safety_bound_extrema(2, 1) = safety_bound.pose.covariance[8];
        safety_bound_extrema(2, 2) = safety_bound.pose.covariance[14];
        if ((robot_position - safety_bound_center).transpose() * safety_bound_extrema.inverse() * (robot_position - safety_bound_center) > 1) {
            ROS_WARN("Safety warning!\n");
        }

        // public
        // frame
        uwb_loc::Frame frame;
        frame.stamp = stamp;
        for (int i = 0; i < 3; i++) {
            frame.p_esti[i] = robot_position(i);
        }
        for (int i = 0; i < num_anchors; i++) {
            frame.dis_arr[i] = y(i);
        }
        frame_pub.publish(frame);

        // visualization
        geometry_msgs::PoseStamped pose_stamped;

        pose_stamped.pose.position.x = robot_position(0);
        pose_stamped.pose.position.y = robot_position(1);
        pose_stamped.pose.position.z = robot_position(2);
        pose_stamped.pose.orientation.w = 1;

        pose_stamped.header.stamp = ros::Time(uwb_msg.stamp);
        pose_stamped.header.frame_id = "world";

        path.header.stamp = ros::Time(uwb_msg.stamp);
        path.poses.push_back(pose_stamped);
        if (path.poses.size() > 70) {
            path.poses.assign(path.poses.begin() + 1, path.poses.end());
        }

        path_pub.publish(path);

        // record position
        std::stringstream ss;
        ss.setf(std::ios::fixed, std::ios::floatfield);
        ss.precision(9);
        ss << uwb_msg.stamp << " ";
        ss.precision(16);
        ss << robot_position(0) << " ";
        ss << robot_position(1) << " ";
        ss << robot_position(2) << " ";
        ss << "0 0 0 1\n";

        traj_save_fs << ss.str();

        ss.str("");

        // record cov
        ss.precision(9);
        ss << uwb_msg.stamp << " ";
        ss.precision(16);
        // ss << robot_position_bound[0] << " " << robot_position_bound[1] << " " << robot_position_bound[2] << "\n";
        ss << robot_position_P(0, 0) << " " << robot_position_P(0, 1) << " " << robot_position_P(0, 2) << " ";
        ss << robot_position_P(1, 0) << " " << robot_position_P(1, 1) << " " << robot_position_P(1, 2) << " ";
        ss << robot_position_P(2, 0) << " " << robot_position_P(2, 1) << " " << robot_position_P(2, 2) << "\n";

        cov_save_fs << ss.str();
    }

    // std::cout << "avg tracking time: " << sum_time / num_sum << '\n';

    traj_save_fs.close();
    cov_save_fs.close();

    return 0;
}