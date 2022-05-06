#include <pcl_ros/point_cloud.h>

#include "detector_3d.h"


namespace detector_3d {


// Constructor
Detector3d::Detector3d(ros::NodeHandle& nh):
    nh(nh), 
    pcl_processor() {

    // ROS
    img_sub = nh.subscribe("/camera/color/image_raw", 1, &Detector3d::img_callback, this);
    pc_sub = nh.subscribe("/camera/depth/color/points", 1, &Detector3d::pc_callback, this);

    ROS_INFO("Detector Instance is Up");
}


// Image Callback
void Detector3d::img_callback(const sensor_msgs::Image::ConstPtr& msgs) {
    // TODO: detector action
}

// Pointcloud Callback
void Detector3d::pc_callback(const sensor_msgs::PointCloud2::ConstPtr& msgs) {

    // Convert ROS Message to PCL
    pcl::PointCloud<pcl::PointXYZ>::Ptr inputCloud (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg<pcl::PointXYZ>(*msgs, *inputCloud);

    // Pipeline
    //   1. Preprocessing (axis conversion, RoI, downsampling)
    Box3d roi = Box3d(0, -0.2, -0.2, 0.4, 0.2, 0.2);
    inputCloud = pcl_processor.rotate(inputCloud);
    inputCloud = pcl_processor.downsampleRoI(roi, inputCloud, 0.01); // downsample in meter

    //   2. Segmentation & Clustering
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> cloudClusters = pcl_processor.cluster(inputCloud, 0.05, 20, 500);

    //   3. Visualization
    bool render_input = true;
    bool render_cluster = true;
    bool render_bbox = true;
    std::vector<Color> colors = { Color(1, 0, 1), Color(0, 1, 1), Color(1, 1, 0), Color(0, 1, 0) };

    //      a. clear previous view
    pcl_processor.clearViewer();

    //      b. render input
    if (render_input)
        pcl_processor.renderCloud(inputCloud, "input cloud", Color(1, 0, 0));

    //      c. render cloud & bbox per cluster
    for (int i=0; i<cloudClusters.size(); i++) {

        auto cluster = cloudClusters[i];
        std::string size_str = "[" + std::to_string(i) + "] cluster size: " + std::to_string(cluster->points.size());
        ROS_INFO("%s", size_str.c_str());
        Box3d box = pcl_processor.getBoundingBox(cluster);

        if (render_cluster)
            pcl_processor.renderCloud(cluster, "clusterCloud"+std::to_string(i), colors[i]);

        if (render_bbox)
            pcl_processor.renderBox(box, i, colors[i], 0.1);
    }

    //      c. RoI & dead zone area
    //pcl_processor.renderBox(roi, -1, Color(1, 0, 0), 1); // Region of Interest
    //pcl_processor.renderSphere(pcl::PointXYZ(0, 0, 0), 0.18, 0, Color(0.5, 0.5, 0), 0.1);
    pcl_processor.spinOnceViewer();
}


}