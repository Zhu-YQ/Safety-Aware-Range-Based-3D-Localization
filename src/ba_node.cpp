#include <Eigen/Core>
#include <fstream>
#include <iostream>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include <ros/ros.h>
#include <visualization_msgs/Marker.h>

#include "./ba.hpp"
#include "uwb_loc/BAResult.h"
#include "uwb_loc/Frame.h"

std::queue<uwb_loc::Frame> frame_msg_queue;
void FrameMsgCallback(const uwb_loc::Frame& msg)
{
    frame_msg_queue.push(msg);
    if (frame_msg_queue.size() > 10) {
        frame_msg_queue.pop();
    }
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "ba_node");
    ros::NodeHandle nh;

    std::string config_file_path;
    nh.param<std::string>("config_file_path", config_file_path, "");
    ROS_INFO_STREAM("Using config file: " << config_file_path << '\n');

    /// read config
    cv::FileStorage fs_read(config_file_path, cv::FileStorage::READ);

    // subscribe frame (from tracking node)
    ros::Subscriber frame_sub = nh.subscribe("/frame", 200, FrameMsgCallback);
    // publish anchor positions
    ros::Publisher aps_pub = nh.advertise<uwb_loc::BAResult>("/ba_result", 20);
    ros::Publisher marker_pub = nh.advertise<visualization_msgs::Marker>("/aps_marker", 20);

    // save results
    std::string save_path;
    fs_read["save_path"] >> save_path;

    std::string aps_save_path = save_path + "/esti_aps.txt";
    std::ofstream aps_save_fs;
    aps_save_fs.open(aps_save_path, std::ios::out);
    aps_save_fs.setf(std::ios::fixed, std::ios::floatfield);
    aps_save_fs.precision(16);

    // std::string bias_save_path = save_path + "/esti_bias.txt";
    // std::ofstream bias_save_fs;
    // bias_save_fs.open(bias_save_path, std::ios::out);
    // bias_save_fs.setf(std::ios::fixed, std::ios::floatfield);
    // bias_save_fs.precision(16);

    // anchor
    int num_anchors = 0;
    fs_read["num_anchors"] >> num_anchors;

    cv::Mat aps_mat;
    fs_read["aps_mat"] >> aps_mat;

    std::vector<Eigen::Vector3d> initial_aps_vec;
    for (size_t i = 0; i < num_anchors; i++) {
        initial_aps_vec.emplace_back(aps_mat.at<double>(i, 0), aps_mat.at<double>(i, 1), aps_mat.at<double>(i, 2));
    }

    cv::Mat aps_range_cv_mat;
    fs_read["aps_range_mat"] >> aps_range_cv_mat;
    Eigen::MatrixXd aps_range_mat;
    cv::cv2eigen<double>(aps_range_cv_mat, aps_range_mat);

    int frame_window_size = 0;
    fs_read["frame_window_size"] >> frame_window_size;

    double noise_var = 0;
    fs_read["noise_var"] >> noise_var;

    double bias_walk_var = 0;
    fs_read["bias_walk_var"] >> bias_walk_var;

    cv::Mat bias_cv_mat;
    fs_read["bias"] >> bias_cv_mat;
    std::vector<double> bias_vec;
    for (int i = 0; i < num_anchors; i++) {
        bias_vec.push_back(bias_cv_mat.at<double>(i, 0));
    }

    /// core part
    BA ba(initial_aps_vec, aps_range_mat, bias_vec, frame_window_size, noise_var, bias_walk_var);

    ros::Rate rate(10);
    while (ros::ok()) {
        ros::spinOnce();

        // BA
        if (!frame_msg_queue.empty()) {
            ba.addFrame(frame_msg_queue.front());
            frame_msg_queue.pop();
        }

        ba.solve();
        const std::vector<Eigen::Vector3d> esti_aps_vec = ba.getAnchorsPositions();
        const std::vector<double> esti_bias_vec = ba.getBias();

        // publish ba result
        uwb_loc::BAResult ba_result_msg;
        double stamp = ros::Time::now().toSec();
        if (std::fabs(stamp - 0) < 1e-3) {
            continue;
        }
        ba_result_msg.stamp = stamp;
        ba_result_msg.num_anchors = num_anchors;
        for (int i = 0; i < esti_aps_vec.size(); i++) {
            const Eigen::Vector3d q_i = esti_aps_vec.at(i);
            for (int j = 0; j < 3; j++) {
                ba_result_msg.aps_arr[i * 3 + j] = q_i(j);
            }
        }
        for (int i = 0; i < esti_bias_vec.size(); i++) {
            ba_result_msg.bias_arr[i] = esti_bias_vec.at(i);
        }
        aps_pub.publish(ba_result_msg);

        // publish anchor markers
        for (int i = 0; i < esti_aps_vec.size(); i++) {
            visualization_msgs::Marker marker;
            marker.header.frame_id = "world";
            marker.header.stamp = ros::Time::now();

            marker.ns = "anchors";
            marker.id = i;
            marker.type = visualization_msgs::Marker::SPHERE;
            marker.action = visualization_msgs::Marker::ADD;

            const Eigen::Vector3d q_i = esti_aps_vec.at(i);
            marker.pose.position.x = q_i(0);
            marker.pose.position.y = q_i(1);
            marker.pose.position.z = q_i(2);
            marker.pose.orientation.x = 0.0;
            marker.pose.orientation.y = 0.0;
            marker.pose.orientation.z = 0.0;
            marker.pose.orientation.w = 1.0;

            marker.scale.x = 0.2;
            marker.scale.y = 0.2;
            marker.scale.z = 0.2;

            marker.color.r = 1.0;
            marker.color.g = 0.54;
            marker.color.b = 0.0;
            marker.color.a = 1.0;

            marker.lifetime = ros::Duration();

            marker_pub.publish(marker);
        }

        // record
        std::stringstream ss;
        ss.setf(std::ios::fixed, std::ios::floatfield);

        ss.precision(9);
        ss << stamp;
        ss.precision(16);
        for (const auto& q_i : esti_aps_vec) {
            ss << " " << q_i(0) << " " << q_i(1) << " " << q_i(2);
        }
        ss << "\n";
        aps_save_fs << ss.str();

        // ss.str("");
        // ss.precision(9);
        // ss << stamp;
        // ss.precision(16);
        // for (const auto& bias : esti_bias_vec) {
        //     ss << " " << bias;
        // }
        // ss << '\n';
        // bias_save_fs << ss.str();

        rate.sleep();
    }

    aps_save_fs.close();
    // bias_save_fs.close();

    return 0;
}