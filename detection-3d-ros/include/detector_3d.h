#include <ros/ros.h>
#include <actionlib/client/simple_action_client.h>

#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <darknet_ros_msgs/CheckForObjectsAction.h>
#include <darknet_ros_msgs/BoundingBox.h>
#include <darknet_ros_msgs/BoundingBoxes.h>


#include <pcl/io/pcd_io.h>
#include <pcl/common/common.h>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#include "pcl_processor.hpp"

#ifndef DETECTOR_3D
#define DETECTOR_3D


namespace detector_3d {


class Detector3d {
public:
    Detector3d(ros::NodeHandle&);
    ~Detector3d() {};

    // Realsense topic subscriber
    void img_callback(const sensor_msgs::Image::ConstPtr& msg);
    void pc_callback(const sensor_msgs::PointCloud2::ConstPtr& msg);

    // Darknet action callbacks
    void action_done_callback(const actionlib::SimpleClientGoalState& state,
                              const darknet_ros_msgs::CheckForObjectsResultConstPtr& result);
    void action_active_callback() {};
    void action_feedback_callback(const darknet_ros_msgs::CheckForObjectsFeedbackConstPtr& feedback) {};

    // pcl_pipeline
    std::vector<Box3d> pcl_pipeline();


private:

    // ROS
    ros::NodeHandle& nh;
    ros::Subscriber img_sub;
    ros::Subscriber pc_sub;
    actionlib::SimpleActionClient<darknet_ros_msgs::CheckForObjectsAction> darknet_client;

    sensor_msgs::PointCloud2::ConstPtr last_pc_msgs;
    darknet_ros_msgs::BoundingBoxes last_2d_boxes_msgs;

    // PCL
    PCLProcessor<pcl::PointXYZ> pcl_processor;
};


}

#endif