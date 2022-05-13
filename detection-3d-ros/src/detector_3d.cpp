#include <pcl_ros/point_cloud.h>
#include <boost/format.hpp>

#include "detector_3d.h"


namespace detector_3d {


// Constructor
Detector3d::Detector3d(ros::NodeHandle& nh):
    nh(nh), 
    pcl_processor(),
    darknet_client("/darknet_ros/check_for_objects", true)
    {

    // ROS
    img_sub = nh.subscribe("/camera/color/image_raw", 1, &Detector3d::img_callback, this);
    pc_sub = nh.subscribe("/camera/depth/color/points", 1, &Detector3d::pc_callback, this);



    ROS_INFO("Detector Instance is Up");
}


// TODO: Image Callback
void Detector3d::img_callback(const sensor_msgs::Image::ConstPtr& msgs) {

    // Connect to darknet action server
    darknet_client.waitForServer();

    // 1. Create action goal object
    darknet_ros_msgs::CheckForObjectsGoal goal;
    goal.image = (*msgs);     


    // 2. Send goal (bind to std::function to pass)
    auto done_cb = boost::bind(&Detector3d::action_done_callback, this, _1, _2);
    auto active_cb = boost::bind(&Detector3d::action_active_callback, this);
    auto feedback_cb = boost::bind(&Detector3d::action_feedback_callback, this, _1);
    darknet_client.sendGoal(goal, done_cb, active_cb, feedback_cb);

    // 3. Process Point Cloud while 2d image being detected.
    auto boxes = pcl_pipeline();

    // 4. Wait for action
    actionlib::SimpleClientGoalState state_result = darknet_client.getState();  
    while (!state_result.isDone()) {
        state_result = darknet_client.getState();
    };

    if (state_result == actionlib::SimpleClientGoalState::SUCCEEDED)
        ROS_INFO("Detection Successed");
    else 
        ROS_INFO("Detection Failed");

    // TODO:: 5. last_2d_boxes is now updated... fuse boxes here.
    for (int i=0; i < last_2d_boxes_msgs.bounding_boxes.size(); i++) {
        auto box = last_2d_boxes_msgs.bounding_boxes[i];
        cout << boost::format("(Box%d) xm:%d, ym:%d, xM:%d, yM:%d") % i % box.xmin % box.ymin % box.xmax % box.ymax << endl;
    }
    

}

// Pointcloud Callback
void Detector3d::pc_callback(const sensor_msgs::PointCloud2::ConstPtr& msgs) {
    last_pc_msgs = msgs; 
}


// Darknet Action Done Callback
void Detector3d::action_done_callback(const actionlib::SimpleClientGoalState& state,
                                      const darknet_ros_msgs::CheckForObjectsResultConstPtr& result) {

    // Save detection result
    this->last_2d_boxes_msgs = result->bounding_boxes;
    pcl_processor.spinOnceViewer();

}


// Apply pcl pipeline to last pointcloud msgs
std::vector<Box3d> Detector3d::pcl_pipeline() {
    // Convert ROS Message to PCL
    pcl::PointCloud<pcl::PointXYZ>::Ptr inputCloud (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg<pcl::PointXYZ>(*last_pc_msgs, *inputCloud);

    // Pipeline
    //   1. Preprocessing (axis conversion, RoI, downsampling)
    Box3d roi = Box3d(0, -0.2, -0.2, 0.4, 0.2, 0.2);
    inputCloud = pcl_processor.rotate(inputCloud);
    inputCloud = pcl_processor.downsampleRoI(roi, inputCloud, 0.01); // downsample in meter

    //   2. Segmentation & Clustering
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> cloudClusters = pcl_processor.cluster(inputCloud, 0.05, 20, 500);


    //   4. Visualization & Get B-Box
    bool render_input = true;
    bool render_cluster = true;
    bool render_bbox = true;
    bool render_roi = false;
    bool render_shpere = false;
    std::vector<Color> colors = { Color(1, 0, 1), Color(0, 1, 1), Color(1, 1, 0), Color(0, 1, 0) };
    std::vector<Box3d> boxes;

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

        // Get Bbox
        Box3d box = pcl_processor.getBoundingBox(cluster);
        boxes.push_back(box);     

        if (render_cluster)
            pcl_processor.renderCloud(cluster, "clusterCloud"+std::to_string(i), colors[i]);

        if (render_bbox)
            pcl_processor.renderBox(box, i, colors[i], 0.1);
    }

    //      d. RoI & dead zone area
    if (render_roi)
        pcl_processor.renderBox(roi, -1, Color(1, 0, 0), 1); // Region of Interest
    if (render_shpere)
        pcl_processor.renderSphere(pcl::PointXYZ(0, 0, 0), 0.18, 0, Color(0.5, 0.5, 0), 0.1);


    pcl_processor.spinOnceViewer();

    return boxes;
}




}