#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>

#include "detector_3d.h"

int main(int argc, char** argv) {

    // init ros
    ros::init(argc, argv, "detection_3d_node");
    ros::NodeHandle nh;

    // init detector instance
    detector_3d::Detector3d detector3d(nh);

    // ros spin
    ros::spin();

    ROS_INFO("Detection-3d node is Down!");

    return 0;
}