#include <ros/ros.h>
#include <actionlib/client/simple_action_client.h>
#include <cv_bridge/cv_bridge.h>

#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <darknet_ros_msgs/CheckForObjectsAction.h>
#include <darknet_ros_msgs/BoundingBox.h>
#include <darknet_ros_msgs/BoundingBoxes.h>
#include <vision_msgs/Detection3D.h>
#include <vision_msgs/Detection3DArray.h>



#include <pcl/io/pcd_io.h>
#include <pcl/common/common.h>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#include "pcl_processor.hpp"
#include "cv_processor.h"

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
                              const darknet_ros_msgs::CheckForObjectsResultConstPtr& result) {};
    void action_active_callback() {};
    void action_feedback_callback(const darknet_ros_msgs::CheckForObjectsFeedbackConstPtr& feedback) {};



private:

    // pcl_pipeline
    std::vector<Box3d> pcl_pipeline();
    
    // Box projection on 2d
    std::vector<Box2d> project_box_2d(std::vector<Box3d>& boxes_3d, int width, int height);


    // ROS
    ros::NodeHandle& nh;
    ros::Subscriber img_sub;
    ros::Subscriber pc_sub;
    ros::Publisher box_pub;
    actionlib::SimpleActionClient<darknet_ros_msgs::CheckForObjectsAction> darknet_client;


    sensor_msgs::PointCloud2::ConstPtr last_pc_msgs;
    std::vector<Box2d> last_detection_2d;

    // PCL
    PCLProcessor<pcl::PointXYZ> pcl_processor;
};


}

#endif