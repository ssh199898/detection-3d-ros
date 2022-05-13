#include <cmath>
#include <vector>

#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/crop_box.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/extract_clusters.h>

#include "pcl_processor.h"


namespace detector_3d {

// Constructor
template <typename PointT>
PCLProcessor<PointT>::PCLProcessor():
    viewer(new pcl::visualization::PCLVisualizer("3D Viewer")) {
    
    // init camera angle
    viewer->initCameraParameters();
    viewer->setCameraPosition(-5, -5, 4, 1, 1, 0);
    viewer->addCoordinateSystem(0.1);
    viewer->setCameraFieldOfView(0.4);

}


// Rotation function
template <typename PointT>
typename pcl::PointCloud<PointT>::Ptr PCLProcessor<PointT>::rotate(typename pcl::PointCloud<PointT>::Ptr& inputCloud) {
    
    // axis swap matrix
    Eigen::Matrix4f xz_swap;
    xz_swap << 0, 0, 1, 0,
               0, 1, 0, 0,
               1, 0, 0, 0,
               0, 0, 0, 1;

    // rotation matrix
    float th = -M_PI/2;
    Eigen::Matrix4f x_rot;
    x_rot << 1, 0, 0, 0,
            0, cos(th), -sin(th), 0,
            0, sin(th), cos(th), 0,
            0, 0, 0, 1;
    
    // y value flip matrix
    Eigen::Matrix4f y_flip;
    y_flip << 1, 0, 0, 0,
            0, -1, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1;


    Eigen::Matrix4f transfrom_mat = y_flip * x_rot * xz_swap;

    // new point cloud
    typename pcl::PointCloud<PointT>::Ptr transformed (new pcl::PointCloud<pcl::PointXYZ>());
    pcl::transformPointCloud(*inputCloud, *transformed, transfrom_mat);

    return transformed;
}


// Downsample & Crop Box function
template <typename PointT>
typename pcl::PointCloud<PointT>::Ptr PCLProcessor<PointT>::downsampleRoI(Box3d roi, typename pcl::PointCloud<PointT>::Ptr& inputCloud, float voxelRes) {
    
    // Pool RoI first before downsampling. (downsampleing may cause overflow)
    typename pcl::PointCloud<PointT>::Ptr cloudRoI (new pcl::PointCloud<PointT>);
    
    Eigen::Vector4f minPoint, maxPoint;
    minPoint << roi.x_min, roi.y_min, roi.z_min;
    maxPoint << roi.x_max, roi.y_max, roi.z_max;

    pcl::CropBox<PointT> region(true);
    region.setMin(minPoint);
    region.setMax(maxPoint);
    region.setInputCloud(inputCloud);
    region.filter(*cloudRoI);


    // new point cloud
    typename pcl::PointCloud<PointT>::Ptr cloudDownsample (new pcl::PointCloud<PointT>);
    pcl::VoxelGrid<PointT>vg;
    vg.setInputCloud(cloudRoI);
    vg.setLeafSize(voxelRes, voxelRes, voxelRes);
    vg.filter(*cloudDownsample);


    return cloudDownsample;
}  


// Clustering function
template <typename PointT>
std::vector<typename pcl::PointCloud<PointT>::Ptr> PCLProcessor<PointT>::cluster(typename pcl::PointCloud<PointT>::Ptr& inputCloud, float clusterTolerance, int minSize, int maxSize) {

    // vector that contains multiple clusters
    std::vector<typename pcl::PointCloud<PointT>::Ptr> clusters;

    // do not perfrom clustering on empty cluster
    if (inputCloud->points.size() < minSize)
        return clusters;

    // 1. Make KD-Tree for search efficiency
    typename pcl::search::KdTree<PointT>::Ptr kdtree (new pcl::search::KdTree<PointT>); // KdTree node has indice as data
    kdtree->setInputCloud(inputCloud);

    // 2. Perform clustering using extractor
    std::vector<pcl::PointIndices> cluster_indices; // vector to save point indices of single cluster
    pcl::EuclideanClusterExtraction<PointT> ec;
    ec.setClusterTolerance(clusterTolerance);
    ec.setMinClusterSize(minSize);
    ec.setMaxClusterSize(maxSize);
    ec.setSearchMethod(kdtree);
    ec.setInputCloud(inputCloud);
    ec.extract(cluster_indices); // pull indices

    // 3. Gather clustered points using indices
    for (pcl::PointIndices getIndices : cluster_indices) {
        // New cloud instance
        typename pcl::PointCloud<PointT>::Ptr cloudCluster (new pcl::PointCloud<PointT>);

        // Gathering points from indices into new instance
        for (int i : getIndices.indices)
            cloudCluster->points.push_back(inputCloud->points[i]);

        // Setup meta data (unordered pc...)
        cloudCluster->width = cloudCluster->points.size();
        cloudCluster->height = 1;
        cloudCluster->is_dense = true;

        // push new cluster into vector!
        clusters.push_back(cloudCluster);
    }

    return clusters;
}


// Bounding box function
template <typename PointT>
Box3d PCLProcessor<PointT>::getBoundingBox(typename pcl::PointCloud<PointT>::Ptr& cluster) {

    // Find min, max point 
    PointT minPoint, maxPoint;
    pcl::getMinMax3D(*cluster, minPoint, maxPoint);

    Box3d box(minPoint.x, minPoint.y, minPoint.z, maxPoint.x, maxPoint.y, maxPoint.z);
    return box;
} 


// Point cloud visualization function
template <typename PointT>
void PCLProcessor<PointT>::renderCloud(typename pcl::PointCloud<PointT>::Ptr& cloud, std::string name, Color color) {
    
    viewer->addPointCloud<PointT>(cloud, name);
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, name);
    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_COLOR, color.r, color.g, color.b, name);
}


// Render Box function
template <typename PointT>
void PCLProcessor<PointT>::renderBox(Box3d box, int id, Color color, float opacity) {

 	if(opacity > 1.0)
		opacity = 1.0;
	if(opacity < 0.0)
		opacity = 0.0;
	
    // Draw box polygon
	std::string cube = "box"+std::to_string(id);
    viewer->addCube(box.x_min, box.x_max, box.y_min, box.y_max, box.z_min, box.z_max, color.r, color.g, color.b, cube);
    viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_REPRESENTATION, pcl::visualization::PCL_VISUALIZER_REPRESENTATION_WIREFRAME, cube); 
    viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, color.r, color.g, color.b, cube);
    viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY, opacity, cube);
    
    // Fill box
    std::string cubeFill = "boxFill"+std::to_string(id);
    viewer->addCube(box.x_min, box.x_max, box.y_min, box.y_max, box.z_min, box.z_max, color.r, color.g, color.b, cubeFill);
    viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_REPRESENTATION, pcl::visualization::PCL_VISUALIZER_REPRESENTATION_SURFACE, cubeFill); 
    viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, color.r, color.g, color.b, cubeFill);
    viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY, opacity*0.3, cubeFill);
}


// Render Sphere function
template <typename PointT>
void PCLProcessor<PointT>::renderSphere(PointT center, float radius, int id, Color color, float opacity) {
    
    if(opacity > 1.0)
		opacity = 1.0;
	if(opacity < 0.0)
		opacity = 0.0;
	
    std::string sphereFill = "sphereFill"+std::to_string(id);
    viewer->addSphere(center, radius, color.r, color.g, color.b, sphereFill);
    viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_REPRESENTATION, pcl::visualization::PCL_VISUALIZER_REPRESENTATION_SURFACE, sphereFill); 
    viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, color.r, color.g, color.b, sphereFill);
    viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY, opacity, sphereFill);
}


// Clearing viewer function
template <typename PointT>
void PCLProcessor<PointT>::clearViewer() {
    viewer->removeAllPointClouds();
    viewer->removeAllShapes();
}


// Spin once viewer
template <typename PointT>
void PCLProcessor<PointT>::spinOnceViewer() {
    viewer->spinOnce();
}

}
