#include <fstream>
#include <iostream>

#include <Eigen/Core>
#include <opencv2/opencv.hpp>

#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <nav_msgs/Path.h>
#include <ros/ros.h>

#include "filter/smf_trans.hpp"

#include "uwb_loc/BAResult.h"
#include "uwb_loc/Frame.h"
#include "uwb_loc/MyUWB.h"

std::queue<uwb_loc::MyUWB> uwb_msg_queue;
void UWBMsgCallback(const uwb_loc::MyUWB& msg) { uwb_msg_queue.push(msg); }

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

int main(int argc, char** argv)
{
    ros::init(argc, argv, "bounding_node");
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

    // pubilsh results
    ros::Publisher ellipse_pub = nh.advertise<geometry_msgs::PoseWithCovarianceStamped>("/safety_bound", 100);

    // save results
    std::string save_path;
    fs_read["save_path"] >> save_path;

    std::string center_save_path = save_path + "/esti_center.txt";
    std::ofstream center_save_fs;
    center_save_fs.open(center_save_path, std::ios::out);
    center_save_fs.setf(std::ios::fixed, std::ios::floatfield);
    center_save_fs.precision(16);

    std::string extrema_save_path = save_path + "/esti_extrema.txt";
    std::ofstream extrema_save_fs;
    extrema_save_fs.open(extrema_save_path, std::ios::out);
    extrema_save_fs.setf(std::ios::fixed, std::ios::floatfield);
    extrema_save_fs.precision(16);

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

    double motion_robot_jx_bound = 0;
    fs_read["motion_robot_jx_bound"] >> motion_robot_jx_bound;
    double motion_robot_jy_bound = 0;
    fs_read["motion_robot_jy_bound"] >> motion_robot_jy_bound;
    double motion_robot_jz_bound = 0;
    fs_read["motion_robot_jz_bound"] >> motion_robot_jz_bound;

    double tag_height = 0;
    fs_read["tag_height"] >> tag_height;

    double noise_var = 0;
    fs_read["noise_var"] >> noise_var;

    /// core part
    ESMFtrans esmf(num_anchors, nlos_thresh, initial_robot_px_var,
        initial_robot_py_var, initial_robot_pz_var,
        initial_robot_v_var, initial_robot_a_var);

    nav_msgs::Path path;
    path.header.frame_id = "world";

    double last_stamp = 0;

    // const double DT_THRESH = 0.05; // 20Hz
    // const double DT_THRESH = 0.1; // 10Hz
    const double DT_THRESH = 0.2; // 5Hz

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
        uwb_msg_queue.pop();

        auto dis_arr = uwb_msg.dis_arr;
        Eigen::VectorXd y;
        y.setZero(num_anchors, 1);
        for (size_t i = 0; i < num_anchors; i++) {
            y(i) = dis_arr.at(i);
        }

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

            esmf.initTagPosition(y_mean, tag_height, aps_vec);

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

        esmf.predict(dt, motion_robot_jx_bound, motion_robot_jy_bound, motion_robot_jz_bound);
        esmf.update_seq(y, noise_var, aps_vec);

        const double end_time = ros::Time::now().toSec();
        sum_time += end_time - start_time;
        num_sum++;

        const auto robot_position_bound = esmf.getTagPositionBound();

        // public
        geometry_msgs::PoseWithCovarianceStamped ellipse_msg;
        ellipse_msg.header.stamp = ros::Time::now();
        ellipse_msg.header.frame_id = "world";
        ellipse_msg.pose.pose.position.x = esmf.x_(0);
        ellipse_msg.pose.pose.position.y = esmf.x_(3);
        ellipse_msg.pose.pose.position.z = esmf.x_(6);
        ellipse_msg.pose.pose.orientation.x = 0;
        ellipse_msg.pose.pose.orientation.y = 0;
        ellipse_msg.pose.pose.orientation.z = 0;
        ellipse_msg.pose.pose.orientation.w = 1;
        ellipse_msg.pose.covariance[0] = esmf.P_(0, 0);
        ellipse_msg.pose.covariance[1] = ellipse_msg.pose.covariance[6] = esmf.P_(0, 3);
        ellipse_msg.pose.covariance[2] = ellipse_msg.pose.covariance[12] = esmf.P_(0, 6);
        ellipse_msg.pose.covariance[7] = esmf.P_(3, 3);
        ellipse_msg.pose.covariance[8] = ellipse_msg.pose.covariance[13] = esmf.P_(3, 6);
        ellipse_msg.pose.covariance[14] = esmf.P_(6, 6);
        ellipse_pub.publish(ellipse_msg);

        // record
        // center
        std::stringstream ss;
        ss.setf(std::ios::fixed, std::ios::floatfield);
        ss.precision(9);
        ss << uwb_msg.stamp << " ";
        ss.precision(16);
        ss << esmf.x_(0) << " " << esmf.x_(3) << " " << esmf.x_(6) << " 0 0 0 1\n";

        center_save_fs << ss.str();

        // shape matrix
        ss.str("");
        ss.precision(9);
        ss << uwb_msg.stamp << " ";
        ss.precision(16);
        ss << esmf.P_(0, 0) << " " << esmf.P_(0, 3) << " " << esmf.P_(0, 6) << " ";
        ss << esmf.P_(3, 0) << " " << esmf.P_(3, 3) << " " << esmf.P_(3, 6) << " ";
        ss << esmf.P_(6, 0) << " " << esmf.P_(6, 3) << " " << esmf.P_(6, 6) << "\n";

        extrema_save_fs << ss.str();
    }

    // std::cout << "avg bounding time: " << sum_time / num_sum << '\n';

    center_save_fs.close();
    extrema_save_fs.close();

    return 0;
}