cmake_minimum_required(VERSION 3.5.1)
project(detection-3d-ros)

add_compile_options(-std=c++11)


# Required packages (catkin)
find_package(catkin REQUIRED COMPONENTS
    roscpp
    std_msgs
    sensor_msgs
    cv_bridge
    pcl_ros
    geometry_msgs
    actionlib_msgs
    actionlib
    darknet_ros_msgs
    message_generation
)

# Required packages (CMake)
find_package(
    PCL REQUIRED
)
find_package(
    OpenCV REQUIRED
)


# ROS messages
# add_message_files()
# add_service_files()
# add_action_files()
# generate_messages(DEPENDENCIES )


# catkin macro
catkin_package(
    INCLUDE_DIRS include
    LIBRARIES ${PROJECT_NAME}
    CATKIN_DEPENDS roscpp std_msgs sensor_msgs cv_bridge pcl_ros geometry_msgs actionlib actionlib_msgs darknet_ros_msgs
    DEPENDS PCL OpenCV
)


add_definitions(
    ${PCL_DEFINITIONS}
    ${OpenCV_DEFINITIONS}
)

include_directories(
    include
    ${catkin_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

add_executable(detection_3d_node 
    src/detection_3d_node.cpp
    src/detector_3d.cpp
    src/cv_processor.cpp
)

add_dependencies(detection_3d_node
    ${detection_3d_node_EXPORTED_TARGETS} 
    ${catkin_EXPORTED_TARGETS}
)

target_link_libraries(detection_3d_node
    ${catkin_LIBRARIES}
    ${PCL_LIBRARIES}
    ${OpenCV_LIBRARIES}
)
