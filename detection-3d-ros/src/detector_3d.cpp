#include <pcl_ros/point_cloud.h>
#include <boost/format.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>


#include "detector_3d.h"

namespace detector_3d {


// Constructor
Detector3d::Detector3d(ros::NodeHandle& nh):
    nh(nh), 
    pcl_processor(),
    darknet_client("/darknet_ros/check_for_objects", true),
    last_pc_msgs(new sensor_msgs::PointCloud2())
    {

    // ROS
    img_sub = nh.subscribe("/camera/color/image_raw", 1, &Detector3d::img_callback, this);
    pc_sub = nh.subscribe("/camera/depth/color/points", 1, &Detector3d::pc_callback, this);
    box_pub = nh.advertise<vision_msgs::Detection3DArray>("/detection_3d/detection_3d", 1);

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
    auto pc_boxes_3d = pcl_pipeline();
    
    // 4. Wait for action
    ros::Time begin = ros::Time::now();
    actionlib::SimpleClientGoalState state_result = darknet_client.getState();  
    while (!state_result.isDone()) {

        ros::Time cur = ros::Time::now();        
        if (cur.sec-begin.sec > 5.0) {
            darknet_client.cancelGoal();
            ROS_WARN("Inference took more than 5 seconds... Canceling");
            break;
        }

        state_result = darknet_client.getState();
        pcl_processor.spinOnceViewer();
    };


    // 5. Convert boxes type
    auto result = darknet_client.getResult();
    std::vector<Box2d> interest_detection_2d;
    std::vector<Box2d> uninterest_detection_2d;
    
    for (auto box : result->bounding_boxes.bounding_boxes) {
         Box2d new_box(box.xmin, box.ymin, box.xmax, box.ymax);
        if (box.id == 1) {
            uninterest_detection_2d.push_back(new_box);
        } else {
            interest_detection_2d.push_back(new_box);
        }
    }

    // 6. last_2d_boxes is now updated.
    cv_bridge::CvImagePtr cv_ptr;
    cv_ptr = cv_bridge::toCvCopy(msgs, sensor_msgs::image_encodings::BGR8);
    int width = cv_ptr->image.size().width;
    int height = cv_ptr->image.size().height;

    //   a. Project box onto 2d image
    auto pc_boxes_2d = CVProcessor::project_box_2d(pc_boxes_3d, width, height);


    //   b. draw boxes!
    CVProcessor::draw_boxes_2d(cv_ptr->image, interest_detection_2d, cv::Scalar(255, 0, 255), 2); // red apple
    CVProcessor::draw_boxes_2d(cv_ptr->image, uninterest_detection_2d, cv::Scalar(0, 255, 0), 2); // green apple
    CVProcessor::draw_boxes_2d(cv_ptr->image, pc_boxes_2d, cv::Scalar(0, 0, 255), 3); // points


    //   c. check IoU and select valid boxes only
    std::vector<bool> valid_pc_box(pc_boxes_2d.size(), false);
    float iou_threshold = 0.3;

    for (int i=0; i < pc_boxes_2d.size(); i++) {
        float best_iou = 0;

        for (int j=0; j < interest_detection_2d.size(); j++) {
            // check IoU and get box with highest IoU and threshold
            float iou = CVProcessor::get_iou(pc_boxes_2d[i], interest_detection_2d[j]);
            if (iou > best_iou && iou > iou_threshold) {
                valid_pc_box[i] = true;
                best_iou = iou;
            }
        }
    }

    //   d. extract valid 3d boxes only;
    std::vector<Box3d> valid_boxes_3d;
    std::vector<Box2d> valid_boxes_2d;
    for (int i=0; i < valid_pc_box.size(); i++) {
        if (valid_pc_box[i] == true) {
            valid_boxes_3d.push_back(pc_boxes_3d[i]);
            valid_boxes_2d.push_back(pc_boxes_2d[i]);
        }
    }

    //   e. redraw valid boxes...
    CVProcessor::draw_boxes_2d(cv_ptr->image, valid_boxes_2d, cv::Scalar(255, 255, 0), 3);
    cv::imshow("det_3d debug", cv_ptr->image);
    cv::waitKey(3);


    // 7. publish 3d point topic
    vision_msgs::Detection3DArray detection_3d_msgs;
    for (auto box : valid_boxes_3d) {
        vision_msgs::Detection3D box_msg;
        
        float center_x = (box.x_min+box.x_max)/2.0;
        float center_y = (box.y_min+box.y_max)/2.0;
        float center_z = (box.z_min+box.z_max)/2.0;

        float size_x = (box.x_max-box.x_min);
        float size_y = (box.y_max-box.y_min);
        float size_z = (box.z_max-box.z_min);

        box_msg.bbox.center.position.x = center_x;
        box_msg.bbox.center.position.y = center_y;
        box_msg.bbox.center.position.z = center_z;
        
        box_msg.bbox.size.x = size_x;
        box_msg.bbox.size.y = size_y;
        box_msg.bbox.size.z = size_z;

        detection_3d_msgs.detections.push_back(box_msg);
    }

    box_pub.publish(detection_3d_msgs);

}

// Pointcloud Callback
void Detector3d::pc_callback(const sensor_msgs::PointCloud2::ConstPtr& msgs) {
    last_pc_msgs = msgs; 
    pcl_processor.spinOnceViewer();
}


// Apply pcl pipeline to last pointcloud msgs
std::vector<Box3d> Detector3d::pcl_pipeline() {
    // Convert ROS Message to PCL
    pcl::PointCloud<pcl::PointXYZ>::Ptr rawCloud (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg<pcl::PointXYZ>(*last_pc_msgs, *rawCloud);

    // Pipeline
    //   1. Preprocessing (axis conversion, RoI, downsampling)
    Box3d roi = Box3d(0, -0.2, -0.2, 0.4, 0.2, 0.2);
    rawCloud = pcl_processor.rotate(rawCloud);
    pcl::PointCloud<pcl::PointXYZ>::Ptr inputCloud = pcl_processor.downsampleRoI(roi, rawCloud, 0.005); // downsample unit in meter

    //   2. Segmentation & Clustering
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> cloudClusters = pcl_processor.cluster(inputCloud, 0.0065, 20, 100);


    //   4. Visualization & Get B-Box
    bool render_raw = false;
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
    if (render_raw)
        pcl_processor.renderCloud(rawCloud, "raw cloud", Color(0.5, 0.5, 0.5));
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