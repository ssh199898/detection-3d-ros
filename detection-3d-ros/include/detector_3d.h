#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>

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

    void img_callback(const sensor_msgs::Image::ConstPtr& msg);
    void pc_callback(const sensor_msgs::PointCloud2::ConstPtr& msg);

private:

    // ROS
    ros::NodeHandle& nh;
    ros::Subscriber img_sub;
    ros::Subscriber pc_sub;

    // PCL
    PCLProcessor<pcl::PointXYZ> pcl_processor;

};


}

#endif