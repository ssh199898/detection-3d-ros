#ifndef PCL_PROCESSOR_H
#define PCL_PROCESSOR_H


#include <pcl/io/pcd_io.h>
#include <pcl/common/common.h>
#include <pcl/visualization/pcl_visualizer.h>

namespace detector_3d {

class Color {
public: 
    Color(float r, float g, float b):
        r(r), g(g), b(b) {};

    float r;
    float g;
    float b;

};


class Box3d {
public:
    Box3d(float x_min, float y_min, float z_min, float x_max, float y_max, float z_max):
        x_min(x_min), y_min(y_min), z_min(z_min), x_max(x_max), y_max(y_max), z_max(z_max) {};
    ~Box3d() {};

    float x_min;
	float y_min;
	float z_min;
	float x_max;
	float y_max;
	float z_max;
};


template <typename PointT>
class PCLProcessor {
public:
    PCLProcessor();
    ~PCLProcessor() {};

    // Filter
    typename pcl::PointCloud<PointT>::Ptr rotate(typename pcl::PointCloud<PointT>::Ptr&);
    typename pcl::PointCloud<PointT>::Ptr downsampleRoI(Box3d, typename pcl::PointCloud<PointT>::Ptr&, float voxelRes=0.005);
    typename pcl::PointCloud<PointT>::Ptr segmentPlane(typename pcl::PointCloud<PointT>::Ptr&);
    std::vector<typename pcl::PointCloud<PointT>::Ptr> cluster(typename pcl::PointCloud<PointT>::Ptr&, float, int, int);
    Box3d getBoundingBox(typename pcl::PointCloud<PointT>::Ptr&);


    // Viz
    void clearViewer();
    void spinOnceViewer();
    void renderCloud(typename pcl::PointCloud<PointT>::Ptr&, std::string, Color);
    void renderBox(Box3d, int, Color, float);    
    void renderSphere(PointT, float, int, Color, float);

private:
    pcl::visualization::PCLVisualizer::Ptr viewer;

};



}

#endif