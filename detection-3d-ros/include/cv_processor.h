#ifndef CV_PROCESSOR_H
#define CV_PROCESSOR_H

#include "pcl_processor.h"


namespace detector_3d {
    
class Box2d {
public:
    Box2d(int x_min, int y_min, int x_max, int y_max):
        x_min(x_min), y_min(y_min), x_max(x_max), y_max(y_max) {};
    ~Box2d() {};

    int x_min;
	int y_min;
	int x_max;
	int y_max;
};



class CVProcessor {
public:
    static std::vector<Box2d> project_box_2d(std::vector<Box3d>& boxes_3d, int width, int height);
    static void draw_boxes_2d(std::vector<Box2d>& boxes_2d);
    static float get_iou(Box2d box1, Box2d box2);


};



}

#endif