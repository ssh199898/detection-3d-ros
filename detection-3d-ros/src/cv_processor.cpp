#include <Eigen/Eigen>
#include <boost/algorithm/clamp.hpp>


#include "cv_processor.h"

namespace detector_3d {

// Project boxes onto plane
std::vector<Box2d> CVProcessor::project_box_2d(std::vector<Box3d>& boxes_3d, int width, int height) {

    // D435i Depth FOV
    // VGA: H: 75+-3, V: 62+-1 (used here)
    // HD: H: 87+-3, V: 58+-1

    const float focal = 1.0; // 1 meter (or can be thought as relative to 1)
    const float scene_hor = sin(35.0 * M_PI / 180.0);
    const float scene_ver = sin(27.0 * M_PI / 180.0);
    
    const float center_x = 0.5; // position of center in scene is (0.5, 0.5) while 3d is (0, 0, 0)
    const float center_y = 0.5;


    // converting global xyz into camera xyz...
    std::vector<Box3d> converted_3d;
    for (auto box_3d : boxes_3d) {
        Box3d new_box(box_3d.y_min, box_3d.z_min, box_3d.x_min, box_3d.y_max, box_3d.z_max, box_3d.x_max);
        converted_3d.push_back(new_box);
    }


    std::vector<Box2d> boxes_2d;
    for (auto box_3d : converted_3d) {

        Eigen::Vector4f min;
        Eigen::Vector4f max;
        min << box_3d.x_min, box_3d.y_min, box_3d.z_min, 1;
        max << box_3d.x_max, box_3d.y_max, box_3d.z_min, 1;
    
        // Translation from depth to color sensor (-1 cm in x)
        Eigen::Matrix4f translation;
        translation << 1, 0, 0, -0.017,
                    0, 1, 0, 0,
                    0, 0, 1, 0,
                    0, 0, 0, 1;

        // Projection matrix (using box_3d.z_min for z-distance to box)
        float z_dist = (box_3d.z_min - 0.001); // 1cm compensation
        Eigen::MatrixXf projection(3, 4);
        projection << focal / z_dist, 0, 0, 0,
                    0, focal / z_dist, 0, 0,
                    0, 0, 1, 0;

        // Normalization matrix (0~1 value)
        Eigen::Matrix3f normalization;
        normalization << 0.5/scene_hor, 0, 0,
                        0, 0.5/scene_ver, 0,
                        0, 0, 1;

        // Projection to 2d camera coordinate
        auto min_2d = normalization * projection * translation * min;
        auto max_2d = normalization * projection * translation * max;

        // Now converting to pixelwise coord (0~1). Flipping.
        Eigen::Vector3f pix_min_2d;
        Eigen::Vector3f pix_max_2d;
        float box_z = box_3d.z_min;
        pix_min_2d << -max_2d(0) + center_x, -max_2d(1) + center_y, 1;
        pix_max_2d << -min_2d(0) + center_x, -min_2d(1) + center_y, 1;

        // Enlarge box a bit
        Eigen::Vector3f center_2d = (pix_min_2d + pix_max_2d) / 2.0;
        Eigen::Vector3f l_pix_min_2d = (pix_min_2d - center_2d) * 1.2 + center_2d;
        Eigen::Vector3f l_pix_max_2d = (pix_max_2d - center_2d) * 1.2 + center_2d;

        // Save in pixel
        Box2d box_2d(l_pix_min_2d(0)*width, l_pix_min_2d(1)*height, l_pix_max_2d(0)*width, l_pix_max_2d(1)*height);
        boxes_2d.push_back(box_2d);              
    }

    return boxes_2d;
}

void CVProcessor::draw_boxes_2d(cv::Mat& image, std::vector<Box2d>& boxes_2d, cv::Scalar color, int thickness) {
    for (auto box : boxes_2d) {
        cv::rectangle(image, cv::Rect(cv::Point((int)box.x_min, (int)box.y_min), cv::Point((int)box.x_max, (int)box.y_max)), color, thickness, 8, 0);    
    }
}

// Intersection over Union. Assume all boxes are "corners" represented.
float CVProcessor::get_iou(const Box2d& box1, const Box2d& box2) {

    int x1 = std::max(box1.x_min, box2.x_min);
    int y1 = std::max(box1.y_min, box2.y_min);
    int x2 = std::min(box1.x_max, box2.x_max);
    int y2 = std::min(box1.y_max, box2.y_max);
    
    int box1_area = std::abs((box1.x_max-box1.x_min) * (box1.y_max-box1.y_min));
    int box2_area = std::abs((box2.x_max-box2.x_min) * (box2.y_max-box2.y_min));

    int intersection = boost::algorithm::clamp(x2-x1, 0, 9999) * boost::algorithm::clamp(y2-y1, 0, 9999);
    float iou = (float)intersection / (float)(box1_area + box2_area - intersection);

    return iou;
}





}