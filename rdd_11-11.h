// Road_Defects_Detector.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API
#include <opencv2/opencv.hpp>   // Include OpenCV API
#include "cv-helpers.hpp"
#include <climits>

#include <librealsense2/rsutil.h>
#include "rs_types.hpp"
#include "rs_frame.hpp"
#include "rs_options.hpp"

//#define NOMINMAX
#include <algorithm> 
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/thread/thread.hpp>
#include <string>
//#include <windows.h>
#include <cstdio>
#include <limits>

// PCL Headers
/*
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/passthrough.h>
#include <pcl/common/common_headers.h>
#include <pcl/features/normal_3d.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/console/parse.h>
#include <boost/thread/thread.hpp>
#include <pcl/io/io.h>
*/
#include <typeinfo>
#include <pcl/filters/passthrough.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/visualization/cloud_viewer.h>

#include "render.cpp"
#include "processPointClouds.cpp"
#include "objectDetection2D.cpp"
#include "dataStructures.h"

#include <pcl/filters/project_inliers.h>
#include <pcl/surface/concave_hull.h>

#include <pcl/common/pca.h>
#include <pcl/common/transforms.h>
#include <pcl/common/distances.h>

// object detection
string yoloBasePath = "C:/Users/hedey/source/repos/Road_Defects_Detector/yolo/";
//string yoloClassesFile = yoloBasePath + "coco.names";
string yoloClassesFile = yoloBasePath + "Proj_obj-13.names";
//string yoloClassesFile = yoloBasePath + "obj.names";
//string yoloModelConfiguration = yoloBasePath + "yolov3.cfg";
string yoloModelConfiguration = yoloBasePath + "yolov3_proj.cfg";
//string yoloModelConfiguration = yoloBasePath + "yolov4_custom_train.cfg";
//string yoloModelWeights = yoloBasePath + "yolov3.weights";
//string yoloModelWeights = yoloBasePath + "yolov3_proj_last_357.weights";
//string yoloModelWeights = yoloBasePath + "yolov4_custom_train_best.weights";
//string yoloModelWeights = yoloBasePath + "yolov3_proj_last_2000.weights";
//string yoloModelWeights = yoloBasePath + "yolov3_proj_last_2700.weights";
//string yoloModelWeights = yoloBasePath + "yolov3_proj_last_4000.weights";
string yoloModelWeights = yoloBasePath + "yolov3_proj_last_6700.weights";

string bag_path = "D:/Eslam/20211023_111231.bag";
//string bag_path = "C:/Users/hedey/Documents/20211008_130436.bag";
//string bag_path = "C:/Users/hedey/Documents/20210925_172641.bag";//28G
//cfg.enable_device_from_file("C:/Users/hedey/Downloads/d435i_walk_around.bag");
//cfg.enable_device_from_file("C:/Users/hedey/Downloads/d435i_walking.bag");
//cfg.enable_device_from_file("C:/Users/hedey/Documents/20210912_171555.bag");
//cfg.enable_device_from_file("C:/Users/hedey/Documents/20211008_130436.bag");
//cfg.enable_device_from_file("C:/Users/hedey/Documents/20210925_172641.bag");//28G
//cfg.enable_device_from_file("C:/Users/hedey/Documents/20211008_133012.bag");
//cfg.enable_device_from_file("C:/Users/hedey/Documents/20211008_132819.bag");

std::tuple<float, float> min_max(std::vector<float> vec)
{
    float min = FLT_MAX, max = -FLT_MAX;
    int size = vec.size();
    //std::cout << min << ", " << max << ", " << size << std::endl;
    for (int i = 0; i < size; i++)
    {
        if (vec[i] < min) {
            min = vec[i];
        }

        if (vec[i] > max) {
            max = vec[i];
        }
    }
    return std::tuple<float, float>(min, max);
}

std::tuple<int, int, int> RGB_Texture(rs2::video_frame texture, rs2::texture_coordinate Texture_XY)
{
    // Get Width and Height coordinates of texture
    int width = texture.get_width();  // Frame width in pixels
    int height = texture.get_height(); // Frame height in pixels

    // Normals to Texture Coordinates conversion
    int x_value = std::min(std::max(int(Texture_XY.u * width + .5f), 0), width - 1);
    int y_value = std::min(std::max(int(Texture_XY.v * height + .5f), 0), height - 1);

    int bytes = x_value * texture.get_bytes_per_pixel();   // Get # of bytes per pixel
    int strides = y_value * texture.get_stride_in_bytes(); // Get line width in bytes
    int Text_Index = (bytes + strides);

    const auto New_Texture = reinterpret_cast<const uint8_t*>(texture.get_data());

    // RGB components to save in tuple
    int NT1 = New_Texture[Text_Index];
    int NT2 = New_Texture[Text_Index + 1];
    int NT3 = New_Texture[Text_Index + 2];

    return std::tuple<int, int, int>(NT1, NT2, NT3);
}

template<typename PointT>
typename pcl::PointCloud<PointT>::Ptr PCL_Conversion(const rs2::points& points, const rs2::video_frame& color) {

    // Object Declaration (Point Cloud)
    typename pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>);

    // Declare Tuple for RGB value Storage (<t0>, <t1>, <t2>)
    std::tuple<uint8_t, uint8_t, uint8_t> RGB_Color;

    //================================
    // PCL Cloud Object Configuration
    //================================
    // Convert data captured from Realsense camera to Point Cloud
    auto sp = points.get_profile().as<rs2::video_stream_profile>();

    cloud->width = static_cast<uint32_t>(sp.width());
    cloud->height = static_cast<uint32_t>(sp.height());
    cloud->is_dense = false;
    cloud->points.resize(points.size());

    auto Texture_Coord = points.get_texture_coordinates();
    auto Vertex = points.get_vertices();

    // Iterating through all points and setting XYZ coordinates
    // and RGB values
    for (int i = 0; i < points.size(); i++)
    {
        //===================================
        // Mapping Depth Coordinates
        // - Depth data stored as XYZ values
        //===================================
        cloud->points[i].x = Vertex[i].x;
        cloud->points[i].y = Vertex[i].y;
        cloud->points[i].z = Vertex[i].z;

        // Obtain color texture for specific point
        RGB_Color = RGB_Texture(color, Texture_Coord[i]);

        // Mapping Color (BGR due to Camera Model)
        cloud->points[i].r = std::get<2>(RGB_Color); // Reference tuple<2>
        cloud->points[i].g = std::get<1>(RGB_Color); // Reference tuple<1>
        cloud->points[i].b = std::get<0>(RGB_Color); // Reference tuple<0>

    }

    return cloud; // PCL RGB Point Cloud generated
}

//setAngle: SWITCH CAMERA ANGLE {XY, TopDown, Side, FPS}
void initCamera(CameraAngle setAngle, pcl::visualization::PCLVisualizer::Ptr& viewer)
{

    viewer->setBackgroundColor(0, 0, 0);

    // set camera position and angle
    viewer->initCameraParameters();
    // distance away in meters
    int distance = 5;

    switch (setAngle)
    {
    case XY: viewer->setCameraPosition(distance, distance, distance, -1, 0, 1); break;
        //case TopDown: viewer->setCameraPosition(0, -distance, 0, 0, -1, 1); break;
    case TopDown: viewer->setCameraPosition(0, -distance, 0, -1, -1, 1); break;
        //case TopDown: viewer->setCameraPosition(-distance, -distance, -distance, 0, -1, 1); break;
    case Side: viewer->setCameraPosition(0, -distance, 0, 0, 0, 1); break;
    case FPS: viewer->setCameraPosition(0, 0, -10, 0, -1, 0); break;
        /*
        case XY: viewer->setCameraPosition(-distance, -distance, distance, 1, 1, 0); break;
        case TopDown : viewer->setCameraPosition(0, 0, distance, 1, 0, 1); break;
        case Side : viewer->setCameraPosition(0, -distance, 0, 0, 0, 1); break;
        case FPS : viewer->setCameraPosition(-10, 0, 0, 0, 0, 1);
        */
    }

    if (setAngle != FPS)
        viewer->addCoordinateSystem(1.0);
}

std::tuple<float, float, float> RGB_Heatmap(int minimum, int maximum, int dp_value)
{
    //minimum, maximum = float(minimum), float(maximum)
    //std::cout << "value: "<<float(value)<<", minimum: "<<float(minimum)<<", maximum: "<<float(maximum)<<std::endl;
    float r, g, b;//ratio, 
    //std::cout << value << std::endl;
    if (dp_value < 0)
    {
        /*
        r = 0;
        b = float(min(255, (int)(255 * (1 - abs((float(value) - float(minimum)) / float(minimum)))))) / 256.0;
        //std::cout << b << std::endl;
        //std::cout << (1 - abs((float(value) - float(minimum)) / float(minimum)))<<std::endl;
        g = float(min(255, (int)(255 * abs((float(value) - float(minimum)) / float(minimum))))) / 256.0;
        */
        b = 0;
        r = float(std::min(255, (int)(255 * (1 - abs((float(dp_value) - float(minimum)) / float(minimum)))))) / 255.0;
        g = float(std::min(255, (int)(255 * abs((float(dp_value) - float(minimum)) / float(minimum))))) / 255.0;
        //std::cout << g << std::endl;
    }
    else
    {
        /*
        b = 0;
        r = float(min(255, (int)(255 * (1 - abs((float(value) - float(maximum)) / float(maximum)))))) / 256.0;
        //std::cout << b << std::endl;
        //std::cout << (1 - abs((float(value) - float(maximum)) / float(maximum))) << std::endl;
        g = float(min(255, (int)(255 * abs((float(value) - float(maximum)) / float(maximum))))) / 256.0;
        */
        r = 0;
        b = float(std::min(255, (int)(255 * (1 - abs((float(dp_value) - float(maximum)) / float(maximum)))))) / 255.0;
        g = float(std::min(255, (int)(255 * abs((float(dp_value) - float(maximum)) / float(maximum))))) / 255.0;
        //std::cout << g << std::endl;

    }
    /*
    float r = float(max(0, int(255 * (1 - ratio))))/256.0;
    float b = float(max(0, int(255 * (ratio - 1))))/256.0;
    float g = (255 - b - r)/256.0;
    */
    //g = (255 - b - r) / 256.0;
    return std::tuple<float, float, float>(r, g, b);
}

std::string zero_padding(string s, int n_zero)
{
    std::string new_string;
    return new_string = std::string(n_zero - s.length(), '0') + s;
}


int main(int argc, char* argv[]) try
{
    typedef pcl::PointXYZRGB Cloud_Type;
    // Declare depth colorizer for pretty visualization of depth data
    rs2::colorizer color_map;

    // Declare RealSense pipeline, encapsulating the actual device and sensors
    rs2::pipeline pipe;
    // Start streaming with default recommended configuration
    //pipe.start();
    rs2::config cfg;
    cfg.enable_device_from_file(bag_path);

    //pipe.start(cfg); // Load from file
    rs2::pipeline_profile profile = pipe.start(cfg);

    const rs2::stream_profile color_profile = profile.get_stream(RS2_STREAM_COLOR);
    const rs2::stream_profile depth_profile = profile.get_stream(RS2_STREAM_DEPTH);
    rs2::align align_to_depth(RS2_STREAM_DEPTH);
    rs2::align align_to_color(RS2_STREAM_COLOR);
    rs2::frameset data;
    rs2::frameset data_aligned_to_color;

    static struct rs2_intrinsics depth_intrin;
    static struct rs2_intrinsics color_intrin;
    static struct rs2_extrinsics depth_extrin_to_color;
    static struct rs2_extrinsics color_extrin_to_depth;
    depth_intrin = depth_profile.as<rs2::video_stream_profile>().get_intrinsics();
    color_intrin = color_profile.as<rs2::video_stream_profile>().get_intrinsics();
    depth_extrin_to_color = depth_profile.as<rs2::video_stream_profile>().get_extrinsics_to(color_profile);
    color_extrin_to_depth = color_profile.as<rs2::video_stream_profile>().get_extrinsics_to(depth_profile);

    rs2::device selected_device = profile.get_device();
    // get playback device and disable realtime mode
    auto playback = selected_device.as<rs2::playback>();
    playback.set_real_time(false);

    using namespace cv;
    const auto depth_frame = "Depth Image";
    const auto color_frame = "Color Image";
    //namedWindow(depth_frame, WINDOW_AUTOSIZE);
    //namedWindow(color_frame, WINDOW_AUTOSIZE);
    namedWindow(depth_frame, WINDOW_NORMAL);
    //namedWindow(color_frame, WINDOW_NORMAL);
    Cloud_Type minPt, maxPt;

    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
    pcl::PointCloud<Cloud_Type>::Ptr newCloud(new pcl::PointCloud<Cloud_Type>);
    CameraAngle setAngle = TopDown; //XY, FPS, Side, TopDown
    initCamera(setAngle, viewer);

    boost::shared_ptr<pcl::visualization::PCLVisualizer> openCloud;

    int f = 0;

    //while (waitKey(1) < 0 && getWindowProperty(window_name, WND_PROP_AUTOSIZE) >= 0)
    while (true)
        //while(!viewer->wasStopped()) && waitKey(1)< 0 && getWindowProperty(window_name, WND_PROP_AUTOSIZE) >= 0
    {

        // Define two align objects. One will be used to align
        // to depth viewport and the other to color.
        // Creating align object is an expensive operation
        // that should not be performed in the main loop

        for(int k=0;k<5;k++)
            data = pipe.wait_for_frames();

        //rs2::frameset data = pipe.wait_for_frames(); // Wait for next set of frames from the camera
        data = pipe.wait_for_frames();
        auto startTime_alignment_process = std::chrono::steady_clock::now();
        data_aligned_to_color = align_to_color.process(data);
        /*
        rs2::frameset data_aligned_to_depth = align_to_depth.process(data);
        rs2::depth_frame depth_aligned_to_depth = data_aligned_to_depth.get_depth_frame();//.apply_filter(color_map);
        rs2::depth_frame depth2_aligned_to_depth = data_aligned_to_depth.get_depth_frame().apply_filter(color_map);
        rs2::frame RGB_aligned_to_depth = data_aligned_to_depth.get_color_frame();
        */

        rs2::depth_frame depth_pc = data.get_depth_frame();//.apply_filter(color_map);
        rs2::frame RGB_pc = data.get_color_frame();

        rs2::depth_frame depth = data_aligned_to_color.get_depth_frame();//.apply_filter(color_map);
        rs2::frame depth2 = data_aligned_to_color.get_depth_frame().apply_filter(color_map);
        rs2::frame RGB = data_aligned_to_color.get_color_frame();

        auto endTime_alignment = std::chrono::steady_clock::now();
        auto elapsedTime_alignment = std::chrono::duration_cast<std::chrono::milliseconds>(endTime_alignment - startTime_alignment_process);
        std::cout << "Frame alignment took " << elapsedTime_alignment.count() << " milliseconds" << std::endl;

        //auto depth_meters = depth_frame_to_meters(depth);
        //std::cout << "Depth Meters Mat: " << depth_meters.cols << ", " << depth_meters.rows << std::endl;

        //auto inrist = rs2::video_stream_profile(depth_aligned_to_depth.get_profile()).get_intrinsics();

        // Query frame size (width and height)
        const int w_rgb = RGB_pc.as<rs2::video_frame>().get_width();
        const int h_rgb = RGB_pc.as<rs2::video_frame>().get_height();
        const int w_pc = depth_pc.as<rs2::video_frame>().get_width();
        const int h_pc = depth_pc.as<rs2::video_frame>().get_height();
        const int w = depth.as<rs2::video_frame>().get_width();
        const int h = depth.as<rs2::video_frame>().get_height();

        // Create OpenCV matrix of size (w,h) from the colorized depth data
        Mat image(Size(w, h), CV_8UC3, (void*)depth2.get_data(), Mat::AUTO_STEP);

        Mat imageRGB(Size(w_rgb, h_rgb), CV_8UC3, (void*)RGB.get_data(), Mat::AUTO_STEP);
        //Mat imageRGB(Size(w_rgb, h_rgb), CV_8UC3, (void*)RGB_pc.get_data(), Mat::AUTO_STEP);

        // Update the window with new data
        imshow(depth_frame, image);
        //imshow(color_frame, imageRGB);

        float confThreshold = 0.2;//.9, 0.5
        float nmsThreshold = 0.4;
        bool bVis = true;
        std::vector<BoundingBox> bBoxes;
        //detectObjects((dataBuffer.end() - 1)->cameraImg, (dataBuffer.end() - 1)->boundingBoxes, confThreshold, nmsThreshold,
        detectObjects(imageRGB, bBoxes, confThreshold, nmsThreshold,
            yoloBasePath, yoloClassesFile, yoloModelConfiguration, yoloModelWeights, bVis);

        // Clear viewer
        //viewer->removeAllPointClouds();
        //viewer->removeAllShapes();
        // Time pointcloud process
        auto startTime_pc_processing = std::chrono::steady_clock::now();

        // Declare pointcloud object, for calculating pointclouds and texture mappings
        rs2::pointcloud pc;

        // Map Color texture to each point
        pc.map_to(RGB_pc);

        // Generate Point Cloud
        auto points = pc.calculate(depth_pc);

        // Convert generated Point Cloud to PCL Formatting
        pcl::PointCloud<Cloud_Type>::Ptr cloud = PCL_Conversion<Cloud_Type>(points, RGB_pc);

        //========================================
        // Filter PointCloud (PassThrough Method)
        //========================================
        /*
        pcl::PassThrough<Cloud_Type> Cloud_Filter; // Create the filtering object
        Cloud_Filter.setInputCloud(cloud);           // Input generated cloud to filter
        Cloud_Filter.setFilterFieldName("z");        // Set field name to Z-coordinate
        Cloud_Filter.setFilterLimits(0.0, 1.0);      // Set accepted interval values
        Cloud_Filter.filter(*newCloud);              // Filtered Cloud Outputted
        */
        if (f == 0)
        {
            std::cout << "Color Frame Width: " << w_rgb << ", " << "Color Frame Height: " << h_rgb << std::endl;
            std::cout << "Original Depth Frame Width: " << w_pc << ", " << "Original Depth Frame Height: " << h_pc << std::endl;
            std::cout << "Aligned Depth Frame Width: " << w << ", " << "Aligned Depth Frame Height: " << h << std::endl;
            pcl::getMinMax3D(*cloud, minPt, maxPt);
            std::cout << "Max x: " << maxPt.x << std::endl;
            std::cout << "Max y: " << maxPt.y << std::endl;
            std::cout << "Max z: " << maxPt.z << std::endl;
            std::cout << "Min x: " << minPt.x << std::endl;
            std::cout << "Min y: " << minPt.y << std::endl;
            std::cout << "Min z: " << minPt.z << std::endl;
            f = 1;
        }

        // Clear viewer
        viewer->removeAllPointClouds();
        viewer->removeAllShapes();
        // Load pcd and run obstacle detection process
        renderPointCloud(viewer, cloud, "bagCloud");
        viewer->spinOnce(); // Allow user to rotate point cloud and view it
        if (bBoxes.size() > 0)
        {
            for (auto bBox : bBoxes)
            {
                auto startTime_bb_conversion_projection = std::chrono::steady_clock::now();
                float Point3d_tl[3], Point3d_tr[3], Point3d_bl[3], Point3d_br[3];
                float Point3d_tl_depth[3], Point3d_tr_depth[3], Point3d_bl_depth[3], Point3d_br_depth[3];
                int x_tl = std::max(bBox.roi.x, 0) , x_tr = std::min(bBox.roi.x + bBox.roi.width, w_rgb-1), x_bl = std::max(bBox.roi.x, 0), x_br = std::min(bBox.roi.x + bBox.roi.width, w_rgb-1);
                int y_tl = std::max(bBox.roi.y, 0), y_tr = std::max(bBox.roi.y, 0), y_bl = std::min(bBox.roi.y + bBox.roi.height, h_rgb-1), y_br = std::min(bBox.roi.y + bBox.roi.height, h_rgb-1);
                std::cout << "(" << x_tl << ", " << y_tl << ")" << ", " << "(" << x_br << ", " << y_br << ")" << std::endl;
                
                float pixel_tl[2] = { float(x_tl),float(y_tl) };
                float pixel_tr[2] = { float(x_tr),float(y_tr) };
                float pixel_bl[2] = { float(x_bl),float(y_bl) };
                float pixel_br[2] = { float(x_br),float(y_br) };

                float pixel_distance_in_meters_tl = depth.get_distance(x_tl, y_tl);
                float pixel_distance_in_meters_tr = depth.get_distance(x_tr, y_tr);
                float pixel_distance_in_meters_bl = depth.get_distance(x_bl, y_bl);
                float pixel_distance_in_meters_br = depth.get_distance(x_br, y_br);

                //std::cout << pixel_distance_in_meters_tl << std::endl; //<< ", " << depth_meters.at<float>(x_tl, y_tl)
                rs2_deproject_pixel_to_point(Point3d_tl, &color_intrin, pixel_tl, pixel_distance_in_meters_tl);
                rs2_transform_point_to_point(Point3d_tl_depth, &color_extrin_to_depth, Point3d_tl);
                //std::cout << "tl: " << Point3d_tl[0] << ", " << Point3d_tl[1] << ", " << Point3d_tl[2] << ", " << std::endl;
                std::cout << "tl_depth: " << Point3d_tl_depth[0] << ", " << Point3d_tl_depth[1] << ", " << Point3d_tl_depth[2] << ", " << std::endl;
                rs2_deproject_pixel_to_point(Point3d_tr, &color_intrin, pixel_tr, pixel_distance_in_meters_tr);
                rs2_transform_point_to_point(Point3d_tr_depth, &color_extrin_to_depth, Point3d_tr);
                //std::cout << "tr: " << Point3d_tr[0] << ", " << Point3d_tr[1] << ", " << Point3d_tr[2] << ", " << std::endl;
                std::cout << "tr_depth: " << Point3d_tr_depth[0] << ", " << Point3d_tr_depth[1] << ", " << Point3d_tr_depth[2] << ", " << std::endl;
                rs2_deproject_pixel_to_point(Point3d_bl, &color_intrin, pixel_bl, pixel_distance_in_meters_bl);
                rs2_transform_point_to_point(Point3d_bl_depth, &color_extrin_to_depth, Point3d_bl);
                //std::cout << "bl: " << Point3d_bl[0] << ", " << Point3d_bl[1] << ", " << Point3d_bl[2] << ", " << std::endl;
                std::cout << "bl_depth: " << Point3d_bl_depth[0] << ", " << Point3d_bl_depth[1] << ", " << Point3d_bl_depth[2] << ", " << std::endl;
                rs2_deproject_pixel_to_point(Point3d_br, &color_intrin, pixel_br, pixel_distance_in_meters_br);
                rs2_transform_point_to_point(Point3d_br_depth, &color_extrin_to_depth, Point3d_br);
                //std::cout << "br: " << Point3d_br[0] << ", " << Point3d_br[1] << ", " << Point3d_br[2] << ", " << std::endl;
                std::cout << "br_depth: " << Point3d_br_depth[0] << ", " << Point3d_br_depth[1] << ", " << Point3d_br_depth[2] << ", " << std::endl;
                auto endTime_projection = std::chrono::steady_clock::now();
                auto elapsedTime_projection = std::chrono::duration_cast<std::chrono::milliseconds>(endTime_projection - startTime_bb_conversion_projection);
                std::cout << "Bbox" + zero_padding(std::to_string(bBox.boxID), 2)+ " Conversion/Projection took " << elapsedTime_projection.count() << " milliseconds" << std::endl;

                //std::vector<float> x_vec = { planarPoint3d_tl[0], planarPoint3d_tr[0], planarPoint3d_bl[0], planarPoint3d_br[0] };
                //std::vector<float> y_vec = { planarPoint3d_tl[1], planarPoint3d_tr[1], planarPoint3d_bl[1], planarPoint3d_br[1] };
                //std::vector<float> z_vec = { planarPoint3d_tl[2], planarPoint3d_tr[2], planarPoint3d_bl[2], planarPoint3d_br[2] };
                std::vector<float> z_vec_depth = { Point3d_tl_depth[2], Point3d_tr_depth[2], Point3d_bl_depth[2], Point3d_br_depth[2] };
                //std::tuple<float, float> x_min_max = min_max(x_vec);
                //std::tuple<float, float> y_min_max = min_max(y_vec);
                //std::tuple<float, float> z_min_max = min_max(z_vec);
                std::tuple<float, float> z_min_max_depth = min_max(z_vec_depth);
                //std::cout << "xmin: " << get<0>(x_min_max) << ", xmax: " << get<1>(x_min_max) << std::endl;
                //std::cout << "ymin: " << get<0>(y_min_max) << ", ymax: " << get<1>(y_min_max) << std::endl;
                //std::cout << "zmin: " << get<0>(z_min_max) << ", zmax: " << get<1>(z_min_max) << std::endl;
                //std::cout << "zmin: " << get<0>(z_min_max_depth) << ", zmax: " << get<1>(z_min_max_depth) << std::endl;
                ProcessPointClouds<Cloud_Type>* pointProcessorI = new ProcessPointClouds<Cloud_Type>();
                //pcl::PointCloud<Cloud_Type>::Ptr filterCloud = pointProcessorI->FilterCloud(cloud, 0.1f, Eigen::Vector4f(-1.75, -3.5, 1, 1), Eigen::Vector4f(1.75, 4.5, 8, 1)); //0.25f
                //pcl::PointCloud<Cloud_Type>::Ptr filterCloud = pointProcessorI->FilterCloud(cloud, 0.1f, Eigen::Vector4f(-22, -8, 18, 1), Eigen::Vector4f(-5, -2, 41, 1)); //0.25f
                //pcl::PointCloud<Cloud_Type>::Ptr filterCloud = pointProcessorI->FilterCloud(cloud, 0.1f, Eigen::Vector4f(planarPoint3d_br[0], planarPoint3d_br[1], planarPoint3d_br[2], 1), Eigen::Vector4f(planarPoint3d_tl[0], planarPoint3d_tl[1], planarPoint3d_tl[2], 1)); //0.25f
                //pcl::PointCloud<Cloud_Type>::Ptr filterCloud = pointProcessorI->FilterCloud(cloud, 0.1f, Eigen::Vector4f(get<0>(x_min_max), get<0>(y_min_max), get<0>(z_min_max), 1), Eigen::Vector4f(get<1>(x_min_max), get<1>(y_min_max), get<1>(z_min_max), 1)); //0.25f
                //pcl::PointCloud<Cloud_Type>::Ptr filterCloud = pointProcessorI->FilterCloud(cloud, 0.1f, Eigen::Vector4f(std::min(x_bl,x_tl), std::min(y_tl,y_tr), get<0>(z_min_max), 1), Eigen::Vector4f(std::max(x_tr,x_br), std::max(y_bl,y_br), get<1>(z_min_max), 1)); //0.25f
                //pcl::PointCloud<Cloud_Type>::Ptr filterCloud = pointProcessorI->FilterCloud(cloud, 0.1f, Eigen::Vector4f(std::max(std::min(planarPoint3d_bl_depth[0], planarPoint3d_tl_depth[0]), minPt.x), std::max(std::min(planarPoint3d_tl_depth[1], planarPoint3d_tr_depth[1]), minPt.y), std::max(get<0>(z_min_max_depth), minPt.z), 1), Eigen::Vector4f(std::min(std::max(planarPoint3d_tr_depth[0], planarPoint3d_br_depth[0]), maxPt.x), std::min(std::max(planarPoint3d_bl_depth[1], planarPoint3d_br_depth[1]), maxPt.y), std::min(get<1>(z_min_max_depth), maxPt.z), 1)); //0.25f
                //pcl::PointCloud<Cloud_Type>::Ptr filterCloud = pointProcessorI->FilterCloud(cloud, 0.1f, Eigen::Vector4f(std::min(planarPoint3d_bl_depth[0], planarPoint3d_tl_depth[0]), std::min(planarPoint3d_tl_depth[1], planarPoint3d_tr_depth[1]), get<0>(z_min_max_depth), 1), Eigen::Vector4f(std::max(planarPoint3d_tr_depth[0], planarPoint3d_br_depth[0]), std::max(planarPoint3d_bl_depth[1], planarPoint3d_br_depth[1]), get<1>(z_min_max_depth), 1)); //0.25f
                //pcl::PointCloud<Cloud_Type>::Ptr filterCloud = pointProcessorI->FilterCloud(cloud, 0.1f, Eigen::Vector4f(std::max(std::min(Point3d_bl_depth[0], Point3d_tl_depth[0]), minPt.x), std::max(std::min(Point3d_tl_depth[1], Point3d_tr_depth[1]), minPt.y), get<0>(z_min_max_depth), 1), Eigen::Vector4f(std::min(std::max(Point3d_tr_depth[0], Point3d_br_depth[0]), maxPt.x), std::min(std::max(Point3d_bl_depth[1], Point3d_br_depth[1]), maxPt.y), get<1>(z_min_max_depth), 1)); //0.25f
                pcl::PointCloud<Cloud_Type>::Ptr filterCloud = pointProcessorI->FilterCloud(cloud, 0.1f, Eigen::Vector4f(std::max(std::min(Point3d_bl_depth[0], Point3d_tl_depth[0]), minPt.x), std::max(std::min(Point3d_tl_depth[1], Point3d_tr_depth[1]), minPt.y), std::max(std::min(Point3d_bl_depth[2], Point3d_br_depth[2]), minPt.z), 1), Eigen::Vector4f(std::min(std::max(Point3d_tr_depth[0], Point3d_br_depth[0]), maxPt.x), std::min(std::max(Point3d_bl_depth[1], Point3d_br_depth[1]), maxPt.y), std::min(std::max(Point3d_tl_depth[2], Point3d_tr_depth[2]), maxPt.z), 1)); //Best
                std::cout << std::max(std::min(Point3d_bl_depth[0], Point3d_tl_depth[0]), minPt.x) << ", " << std::max(std::min(Point3d_tl_depth[1], Point3d_tr_depth[1]), minPt.y) << ", " << std::max(std::min(Point3d_bl_depth[2], Point3d_br_depth[2]), minPt.z) << std::endl;
                std::cout << std::min(std::max(Point3d_tr_depth[0], Point3d_br_depth[0]), maxPt.x) << ", " << std::min(std::max(Point3d_bl_depth[1], Point3d_br_depth[1]), maxPt.y) << "," << std::min(std::max(Point3d_tl_depth[2], Point3d_tr_depth[2]), maxPt.z) << std::endl;
                pcl::PointCloud<Cloud_Type>::Ptr ROI_min_max(new pcl::PointCloud<Cloud_Type>);
                Cloud_Type ROImin, ROImax;
                ROImin.x = std::max(std::min(Point3d_bl_depth[0], Point3d_tl_depth[0]), minPt.x);
                ROImin.y = std::max(std::min(Point3d_tl_depth[1], Point3d_tr_depth[1]), minPt.y);
                ROImin.z = std::max(std::min(Point3d_bl_depth[2], Point3d_br_depth[2]), minPt.z);
                ROImax.x = std::min(std::max(Point3d_tr_depth[0], Point3d_br_depth[0]), maxPt.x);
                ROImax.y = std::min(std::max(Point3d_bl_depth[1], Point3d_br_depth[1]), maxPt.y);
                ROImax.z = std::min(std::max(Point3d_tl_depth[2], Point3d_tr_depth[2]), maxPt.z);
                ROI_min_max->points.push_back(ROImin);
                ROI_min_max->points.push_back(ROImax);
                renderPointCloud(viewer, ROI_min_max, "ROIminmax" + zero_padding(std::to_string(bBox.boxID), 3), Color(0, 1, 1), 10);
                //pcl::PointCloud<Cloud_Type>::Ptr filterCloud = pointProcessorI->FilterCloud(cloud, 0.1f, Eigen::Vector4f(std::max(std::max(Point3d_bl_depth[0], Point3d_tl_depth[0]), minPt.x), std::max(std::max(Point3d_tl_depth[1], Point3d_tr_depth[1]), minPt.y), std::max(std::max(Point3d_bl_depth[2], Point3d_br_depth[2]), minPt.z), 1), Eigen::Vector4f(std::min(std::min(Point3d_tr_depth[0], Point3d_br_depth[0]), maxPt.x), std::min(std::min(Point3d_bl_depth[1], Point3d_br_depth[1]), maxPt.y), std::min(std::min(Point3d_tl_depth[2], Point3d_tr_depth[2]), maxPt.z), 1)); 
                //std::cout << std::max(std::max(Point3d_bl_depth[0], Point3d_tl_depth[0]), minPt.x) << ", " << std::max(std::max(Point3d_tl_depth[1], Point3d_tr_depth[1]), minPt.y) << ", " << std::max(std::max(Point3d_bl_depth[2], Point3d_br_depth[2]), minPt.z) << std::endl;
                //std::cout << std::min(std::min(Point3d_tr_depth[0], Point3d_br_depth[0]), maxPt.x) << ", " << std::min(std::min(Point3d_bl_depth[1], Point3d_br_depth[1]), maxPt.y) << "," << std::min(std::min(Point3d_tl_depth[2], Point3d_tr_depth[2]), maxPt.z) << std::endl;
                //pcl::PointCloud<Cloud_Type>::Ptr filterCloud = pointProcessorI->FilterCloud(cloud, 0.1f, Eigen::Vector4f(planarPoint3d_bl_depth[0], planarPoint3d_tr_depth[1], get<0>(z_min_max_depth), 1), Eigen::Vector4f(planarPoint3d_br_depth[0], std::min(std::max(planarPoint3d_bl_depth[1], planarPoint3d_br_depth[1]), maxPt.y), get<1>(z_min_max_depth), 1)); //0.25f
                //pcl::PointCloud<Cloud_Type>::Ptr filterCloud = pointProcessorI->FilterCloud(cloud, 0.1f, Eigen::Vector4f(Point3d_bl_depth[0], Point3d_tr_depth[1], Point3d_br_depth[2], 1), Eigen::Vector4f(Point3d_br_depth[0], Point3d_br_depth[1], Point3d_tr_depth[2], 1)); //0.25f
                if (filterCloud->points.size() > 0)
                {
                    renderPointCloud(viewer, filterCloud, "filterCloud" + zero_padding(std::to_string(bBox.boxID), 3), Color(1, 0, 1));
                    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
                    //std::pair<typename pcl::PointCloud<RGB_Cloud>::Ptr, typename pcl::PointCloud<RGB_Cloud>::Ptr> segmentCloud = pointProcessorI->SegmentPlane(cloud, 3000, 0.01, coefficients); // Possible pavement defects //.01
                    std::pair<typename pcl::PointCloud<Cloud_Type>::Ptr, typename pcl::PointCloud<Cloud_Type>::Ptr> segmentCloud = pointProcessorI->SegmentPlane(filterCloud, 3000, 0.01, coefficients); // Possible pavement defects //.01
                    std::cerr << "Model coefficients: " << coefficients->values[0] << " "
                        << coefficients->values[1] << " "
                        << coefficients->values[2] << " "
                        << coefficients->values[3] << std::endl;
                    //std::pair<typename pcl::PointCloud<PointT>::Ptr, typename pcl::PointCloud<PointT>::Ptr> segmentCloud = pointProcessorI->SegmentPlane(filterCloud, 3000, 0.05); // Possible pavement defects
                    //renderPointCloud(viewer, segmentCloud.first, "obstCloud", Color(1, 0, 1));
                    renderPointCloud(viewer, segmentCloud.second, "planeCloud2" + zero_padding(std::to_string(bBox.boxID), 3), Color(1, 1, 0));
                    if (segmentCloud.first->points.size() > 0)
                    {
                        std::vector<pcl::PointCloud<Cloud_Type>::Ptr> cloudClusters = pointProcessorI->Clustering(segmentCloud.first, 0.02, 1000, 1000000);
                        int clusterId = 0;
                        //std::vector<Color> colors = { Color(1,0,0), Color(1,1,0), Color(0,0,1) };

                        for (pcl::PointCloud<Cloud_Type>::Ptr cluster : cloudClusters)
                        {
                            std::cout << "cluster size ";
                            pointProcessorI->numPoints(cluster);

                            double min_depth = 10000;
                            double max_depth = -10000;
                            map<int, vector<Cloud_Type>> depth_estimation;
                            //map<int, pcl::PointCloud<PointT>::Ptr> depth_estimation;
                            for (Cloud_Type point : cluster->points)
                            {
                                double ptp_dist = pcl::pointToPlaneDistanceSigned(point,
                                    coefficients->values[0],
                                    coefficients->values[1],
                                    coefficients->values[2],
                                    coefficients->values[3]);
                                if (ptp_dist < min_depth && ptp_dist != coefficients->values[3])
                                    min_depth = ptp_dist;

                                if (ptp_dist > max_depth && ptp_dist != coefficients->values[3])
                                    max_depth = ptp_dist;
                                int rounded_depth;
                                rounded_depth = floor(ptp_dist * 1000.0 + .5);
                                depth_estimation[rounded_depth].push_back(point);
                                //depth_estimation[rounded_depth]->points.push_back(point);
                            }
                            if (min_depth > 0 && max_depth > 0)
                            {
                                //std::cout << "Pothole's depth ranges from: " << min_depth << " to: " << max_depth << ", Max Depth is: " << abs(floor(max_depth * 1000.0 + .5)) << " mm." << std::endl;
                                std::cout << "Defect's height ranges from: " << min_depth << " to: " << max_depth << ", Max Height is: " << abs(floor(max_depth * 1000.0 + .5)) << " mm." << std::endl;
                            }
                            else if (min_depth < 0 && max_depth < 0)
                            {
                                std::cout << "Pothole's depth ranges from: " << min_depth << " to: " << max_depth << ", Max Depth is: " << abs(floor(min_depth * 1000.0 + .5)) << " mm." << std::endl;
                                //std::cout << "Defect's height ranges from: " << min_depth << " to: " << max_depth << ", Max Height is: " << abs(floor(max_depth * 1000.0 + .5)) << " mm." << std::endl;
                            }
                            else
                                std::cout << "Inconsistent Depth (+ & -), ranging from: " << min_depth << " to: " << max_depth << std::endl;

                            pcl::PointCloud<Cloud_Type>::Ptr min_depth_Cloud(new pcl::PointCloud<Cloud_Type>);
                            int rounded_max = floor(max_depth * 1000.0 + .5);
                            int rounded_min = floor(min_depth * 1000.0 + .5);
                            int max_level = 150;
                            int min_level = -150;
                            int j = 0;
                            std::vector<int> depths;
                            int capacity = 0;
                            for (auto it = depth_estimation.begin(); it != depth_estimation.end(); ++it)
                            {
                                pcl::PointCloud<Cloud_Type>::Ptr depth_contour_Cloud(new pcl::PointCloud<Cloud_Type>);
                                if (min_depth != 10000 && max_depth != -10000)
                                {
                                    std::tuple<float, float, float> RGB_Color;
                                    //std::cout << int(it->first) << std::endl;
                                    RGB_Color = RGB_Heatmap(min_level, max_level, int(it->first));
                                    if ((it->first > 0 && it->first == rounded_min) || (it->first < 0 && it->first == rounded_max))
                                    {
                                        for (Cloud_Type point : it->second)
                                            min_depth_Cloud->points.push_back(point);

                                        renderPointCloud(viewer, min_depth_Cloud, "minDepth" + zero_padding(std::to_string(bBox.boxID), 3) + zero_padding(std::to_string(clusterId), 4) + zero_padding(std::to_string(j), 4), Color(get<0>(RGB_Color), get<1>(RGB_Color), get<2>(RGB_Color)));
                                        //std::cout << "Size of nearest contour to plane is: " << min_depth_Cloud->points.size() << std::endl;
                                        //std::cout << "Colors: " << get<0>(RGB_Color) << ", " << get<1>(RGB_Color) << ", " << get<2>(RGB_Color) << std::endl;
                                    }
                                    else
                                    {
                                        for (Cloud_Type point : it->second)
                                            depth_contour_Cloud->points.push_back(point);

                                        renderPointCloud(viewer, depth_contour_Cloud, "depthContour" + zero_padding(std::to_string(bBox.boxID), 3) + zero_padding(std::to_string(clusterId), 4) + zero_padding(std::to_string(j), 4), Color(get<0>(RGB_Color), get<1>(RGB_Color), get<2>(RGB_Color)));
                                        /*
                                        std::cout << "Contour size is: " << depth_contour_Cloud->points.size() << std::endl;
                                        //std::cout << "Colors: " << get<0>(RGB_Color) <<", " << get<1>(RGB_Color) << ", " << get<2>(RGB_Color) << std::endl;
                                        cout << endl;
                                        cout << "Press [Q] in viewer to continue. " << endl;
                                        */
                                    }
                                    depths.push_back(it->first);
                                    if (min_depth < 0 && max_depth < 0)
                                    {
                                        capacity += (it->first)* depth_estimation[it->first].size();
                                    }
                                }
                                j++;
                            }
                            std::cout << "Pothole's estimated capacity in mm is: " << capacity <<" ." << std::endl;
                            pcl::PointCloud<Cloud_Type>::Ptr five_perc_depth_Cloud(new pcl::PointCloud<Cloud_Type>);
                            pcl::PointCloud<Cloud_Type>::Ptr tfive_perc_depth_Cloud(new pcl::PointCloud<Cloud_Type>);
                            int five_perc;
                            int tfive_perc;
                            if (rounded_max > 0)
                            {
                                std::sort(depths.begin(), depths.end());
                                //renderPointCloud(viewer, five_perc_depth_Cloud, "five_perc_depth_Cloud" + zero_padding(std::to_string(clusterId), 4), Color(get<0>(RGB_Color), get<1>(RGB_Color), get<2>(RGB_Color)));
                            }
                            else if (rounded_min < 0)
                            {
                                std::sort(depths.begin(), depths.end(), greater<int>());
                            }
                            five_perc = depths[int(.05 * depths.size())];
                            tfive_perc = depths[int(.25 * depths.size())];
                            //std::cout << five_perc<<endl;
                            for (Cloud_Type point : depth_estimation[five_perc])
                                five_perc_depth_Cloud->points.push_back(point);
                            for (Cloud_Type point : depth_estimation[tfive_perc])
                                tfive_perc_depth_Cloud->points.push_back(point);
                            //
                            //renderPointCloud(viewer, cluster, "obstCloud" + std::to_string(clusterId), colors[clusterId]);
                            //do
                            //{
                            //    cout << '\n' << "Press a key to render a cluster ...";
                            //} while (cin.get() != '\n');
                            /*
                            pcl::PointCloud<PointT>::Ptr min_depth_Cloud(new pcl::PointCloud<PointT>);
                            for (PointT point : depth_estimation[floor(min_depth * 1000.0 + .5)])
                                min_depth_Cloud->points.push_back(point);
                            renderPointCloud(viewer, min_depth_Cloud, "minDepth", Color(1, 0, 0));
                            */

                            //viewer->spin(); // Allow user to rotate point cloud and view it
                            /*
                            if (min_depth != 10000 && max_depth != -10000)
                            {
                                if (min_depth_Cloud->points.size() > 20)
                                {
                                    std::cout << "Size of nearest contour to plane is: " << min_depth_Cloud->points.size() << std::endl;
                                    pcl::ModelCoefficients::Ptr coefficientsCircle(new pcl::ModelCoefficients);
                                    std::pair<typename pcl::PointCloud<Cloud_Type>::Ptr, typename pcl::PointCloud<Cloud_Type>::Ptr> segmentCircle = pointProcessorI->SegmentCircle2D(min_depth_Cloud, 5000, 0.01, coefficientsCircle); // Possible pavement defects //.01
                                    std::cerr << "Circle's coefficients are: " << coefficientsCircle->values[0] << " "
                                        << coefficientsCircle->values[1] << " "
                                        << coefficientsCircle->values[2] << " "
                                        << std::endl;
                                    //std::pair<typename pcl::PointCloud<PointT>::Ptr, typename pcl::PointCloud<PointT>::Ptr> segmentCloud = pointProcessorI->SegmentPlane(filterCloud, 3000, 0.05); // Possible pavement defects
                                    //renderPointCloud(viewer, segmentCloud.first, "obstCloud", Color(1, 0, 1));
                                    renderPointCloud(viewer, segmentCircle.second, "CircleCloud" + zero_padding(std::to_string(bBox.boxID), 3) + zero_padding(std::to_string(clusterId), 4), Color(1, 1, 1));
                                }
                                if (five_perc_depth_Cloud->points.size() > 20)
                                {
                                    std::cout << "Size of the 5 percentile contour is: " << five_perc_depth_Cloud->points.size() << std::endl;
                                    pcl::ModelCoefficients::Ptr coefficientsCircle_5(new pcl::ModelCoefficients);
                                    std::pair<typename pcl::PointCloud<Cloud_Type>::Ptr, typename pcl::PointCloud<Cloud_Type>::Ptr> segmentCircle_5 = pointProcessorI->SegmentCircle2D(five_perc_depth_Cloud, 5000, 0.01, coefficientsCircle_5); // Possible pavement defects //.01
                                    std::cerr << "Circle's coefficients are: " << coefficientsCircle_5->values[0] << " "
                                        << coefficientsCircle_5->values[1] << " "
                                        << coefficientsCircle_5->values[2] << " "
                                        << std::endl;
                                    //std::pair<typename pcl::PointCloud<PointT>::Ptr, typename pcl::PointCloud<PointT>::Ptr> segmentCloud = pointProcessorI->SegmentPlane(filterCloud, 3000, 0.05); // Possible pavement defects
                                    //renderPointCloud(viewer, segmentCloud.first, "obstCloud", Color(1, 0, 1));
                                    renderPointCloud(viewer, segmentCircle_5.second, "CircleCloud_5" + zero_padding(std::to_string(bBox.boxID), 3) + zero_padding(std::to_string(clusterId), 4), Color(.95, .95, .95));
                                }
                                if (tfive_perc_depth_Cloud->points.size() > 20)
                                {
                                    std::cout << "Size of the 25 percentile contour is: " << tfive_perc_depth_Cloud->points.size() << std::endl;
                                    pcl::ModelCoefficients::Ptr coefficientsCircle_25(new pcl::ModelCoefficients);
                                    std::pair<typename pcl::PointCloud<Cloud_Type>::Ptr, typename pcl::PointCloud<Cloud_Type>::Ptr> segmentCircle_25 = pointProcessorI->SegmentCircle2D(tfive_perc_depth_Cloud, 5000, 0.01, coefficientsCircle_25); // Possible pavement defects //.01
                                    std::cerr << "Circle's coefficients are: " << coefficientsCircle_25->values[0] << " "
                                        << coefficientsCircle_25->values[1] << " "
                                        << coefficientsCircle_25->values[2] << " "
                                        << std::endl;
                                    //std::pair<typename pcl::PointCloud<PointT>::Ptr, typename pcl::PointCloud<PointT>::Ptr> segmentCloud = pointProcessorI->SegmentPlane(filterCloud, 3000, 0.05); // Possible pavement defects
                                    //renderPointCloud(viewer, segmentCloud.first, "obstCloud", Color(1, 0, 1));
                                    renderPointCloud(viewer, segmentCircle_25.second, "CircleCloud_25" + zero_padding(std::to_string(bBox.boxID), 3) + zero_padding(std::to_string(clusterId), 4), Color(.75, .75, .75));
                                }
                            }
                            */

                            //}
                            //if(render_box)
                            //{
                            //Box box = pointProcessor.BoundingBox(cluster);
                            //renderBox(viewer, box, clusterId);
                            //}
                            pcl::PointCloud<Cloud_Type>::Ptr cloud_projected(new pcl::PointCloud<Cloud_Type>);
                            // Project the model inliers
                            pcl::ProjectInliers<Cloud_Type> proj;
                            proj.setModelType(pcl::SACMODEL_PLANE);
                            // proj.setIndices (inliers);
                            proj.setInputCloud(cluster);
                            proj.setModelCoefficients(coefficients);
                            proj.filter(*cloud_projected);
                            std::cerr << "PointCloud after projection has: "
                                << cloud_projected->size() << " data points." << std::endl;

                            // Create a Concave Hull representation of the projected inliers
                            pcl::PointCloud<Cloud_Type>::Ptr cloud_hull(new pcl::PointCloud<Cloud_Type>);
                            pcl::ConcaveHull<Cloud_Type> chull;
                            chull.setInputCloud(cloud_projected);
                            chull.setAlpha(0.1);
                            chull.reconstruct(*cloud_hull);

                            std::cerr << "Concave hull has: " << cloud_hull->size()
                                << " data points." << std::endl;
                            renderPointCloud(viewer, cloud_hull, "cloud_hull" + zero_padding(std::to_string(bBox.boxID), 3) + zero_padding(std::to_string(clusterId), 4), Color(0, 0, 0));
                            float concave_hull_area = calculatePolygonArea(*cloud_hull);
                            std::cout << "Area within concave hull is: " << concave_hull_area << std::endl;
                            float avg_area_per_point = concave_hull_area / cloud_projected->size();
                            float volume = avg_area_per_point * capacity/1000;
                            std::cout << "Volume is: " << volume << " (" << 1000 * volume << " litres.)"<<std::endl;
                            // Compute principal directions
                            Eigen::Vector4f pcaCentroid;
                            pcl::compute3DCentroid(*cloud_projected, pcaCentroid);
                            Eigen::Matrix3f covariance;
                            computeCovarianceMatrixNormalized(*cloud_projected, pcaCentroid, covariance);
                            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigen_solver(covariance, Eigen::ComputeEigenvectors);
                            Eigen::Matrix3f eigenVectorsPCA = eigen_solver.eigenvectors();
                            eigenVectorsPCA.col(2) = eigenVectorsPCA.col(0).cross(eigenVectorsPCA.col(1));  /// This line is necessary for proper orientation in some cases. The numbers come out the same without it, but
                                                                                                            ///    the signs are different and the box doesn't get correctly oriented in some cases.
                            /* // Note that getting the eigenvectors can also be obtained via the PCL PCA interface with something like:
                            pcl::PointCloud<pcl::PointXYZ>::Ptr cloudPCAprojection (new pcl::PointCloud<pcl::PointXYZ>);
                            pcl::PCA<pcl::PointXYZ> pca;
                            pca.setInputCloud(cloudSegmented);
                            pca.project(*cloudSegmented, *cloudPCAprojection);
                            std::cerr << std::endl << "EigenVectors: " << pca.getEigenVectors() << std::endl;
                            std::cerr << std::endl << "EigenValues: " << pca.getEigenValues() << std::endl;
                            // In this case, pca.getEigenVectors() gives similar eigenVectors to eigenVectorsPCA.
                            */
                            // Transform the original cloud to the origin where the principal components correspond to the axes.
                            Eigen::Matrix4f projectionTransform(Eigen::Matrix4f::Identity());
                            projectionTransform.block<3, 3>(0, 0) = eigenVectorsPCA.transpose();
                            projectionTransform.block<3, 1>(0, 3) = -1.f * (projectionTransform.block<3, 3>(0, 0) * pcaCentroid.head<3>());
                            pcl::PointCloud<Cloud_Type>::Ptr cloudPointsProjected(new pcl::PointCloud<Cloud_Type>);
                            pcl::transformPointCloud(*cloud_projected, *cloudPointsProjected, projectionTransform);
                            // Get the minimum and maximum points of the transformed cloud.
                            Cloud_Type minPoint, maxPoint;
                            pcl::getMinMax3D(*cloudPointsProjected, minPoint, maxPoint);
                            const Eigen::Vector3f meanDiagonal = 0.5f * (maxPoint.getVector3fMap() + minPoint.getVector3fMap());



                            // Final transform
                            const Eigen::Quaternionf bboxQuaternion(eigenVectorsPCA); //Quaternions are a way to do rotations https://www.youtube.com/watch?v=mHVwd8gYLnI
                            const Eigen::Vector3f bboxTransform = eigenVectorsPCA * meanDiagonal + pcaCentroid.head<3>();
                            //viewer->addCube(bboxTransform, bboxQuaternion, maxPoint.x - minPoint.x, maxPoint.y - minPoint.y, maxPoint.z - minPoint.z, "bbox" + zero_padding(std::to_string(bBox.boxID), 3) + zero_padding(std::to_string(clusterId), 4));

                            // This viewer has 4 windows, but is only showing images in one of them as written here.
                            /*
                            pcl::visualization::PCLVisualizer* visu;
                            visu = new pcl::visualization::PCLVisualizer(argc, argv, "PlyViewer");
                            int mesh_vp_1, mesh_vp_2, mesh_vp_3, mesh_vp_4;
                            visu->createViewPort(0.0, 0.5, 0.5, 1.0, mesh_vp_1);
                            visu->createViewPort(0.5, 0.5, 1.0, 1.0, mesh_vp_2);
                            visu->createViewPort(0.0, 0, 0.5, 0.5, mesh_vp_3);
                            visu->createViewPort(0.5, 0, 1.0, 0.5, mesh_vp_4);
                            visu->addPointCloud(cloudSegmented, ColorHandlerXYZ(cloudSegmented, 30, 144, 255), "bboxedCloud", mesh_vp_3);
                            visu->addCube(bboxTransform, bboxQuaternion, maxPoint.x - minPoint.x, maxPoint.y - minPoint.y, maxPoint.z - minPoint.z, "bbox", mesh_vp_3);
                            */
                            Cloud_Type minPoint_z, maxPoint_z;
                            pcl::PointCloud<Cloud_Type>::Ptr pca_end_points_z(new pcl::PointCloud<Cloud_Type>);
                            pcl::PointCloud<Cloud_Type>::Ptr pca_end_points_z_out(new pcl::PointCloud<Cloud_Type>);
                            //minPoint_x.x = minPoint.x, minPoint_x.y = 0.5*(minPoint.y+maxPoint.y), minPoint_x.z = 0.5*(minPoint.z+maxPoint.z);
                            minPoint_z.x = 0.0, minPoint_z.y = 0.0, minPoint_z.z = minPoint.z;
                            //maxPoint_x.x = maxPoint.x, maxPoint_x.y = 0.5*(minPoint.y+maxPoint.y), maxPoint_x.z = 0.5*(minPoint.z+maxPoint.z);
                            maxPoint_z.x = 0.0, maxPoint_z.y = 0.0, maxPoint_z.z = maxPoint.z;
                            pca_end_points_z->points.push_back(minPoint_z);
                            pca_end_points_z->points.push_back(maxPoint_z);
                            pcl::transformPointCloud(*pca_end_points_z, *pca_end_points_z_out, bboxTransform, bboxQuaternion);
                            renderPointCloud(viewer, pca_end_points_z_out, "pca_end_points_z" + zero_padding(std::to_string(bBox.boxID), 3) + zero_padding(std::to_string(clusterId), 4), Color(0, 0, 1), 10);
                            float distance_z = pcl::euclideanDistance(pca_end_points_z_out->points[0], pca_end_points_z_out->points[1]);
                            std::cout << "Diameter (z) equals: " << distance_z << " m." << std::endl;

                            Cloud_Type minPoint_y, maxPoint_y;
                            pcl::PointCloud<Cloud_Type>::Ptr pca_end_points_y(new pcl::PointCloud<Cloud_Type>);
                            pcl::PointCloud<Cloud_Type>::Ptr pca_end_points_y_out(new pcl::PointCloud<Cloud_Type>);
                            //minPoint_x.x = minPoint.x, minPoint_x.y = 0.5*(minPoint.y+maxPoint.y), minPoint_x.z = 0.5*(minPoint.z+maxPoint.z);
                            minPoint_y.x = 0.0, minPoint_y.y = minPoint.y, minPoint_y.z = 0.0;
                            //maxPoint_x.x = maxPoint.x, maxPoint_x.y = 0.5*(minPoint.y+maxPoint.y), maxPoint_x.z = 0.5*(minPoint.z+maxPoint.z);
                            maxPoint_y.x = 0.0, maxPoint_y.y = maxPoint.y, maxPoint_y.z = 0.0;
                            pca_end_points_y->points.push_back(minPoint_y);
                            pca_end_points_y->points.push_back(maxPoint_y);
                            pcl::transformPointCloud(*pca_end_points_y, *pca_end_points_y_out, bboxTransform, bboxQuaternion);
                            renderPointCloud(viewer, pca_end_points_y_out, "pca_end_points_y" + zero_padding(std::to_string(bBox.boxID), 3) + zero_padding(std::to_string(clusterId), 4), Color(1, 0, 0), 10);
                            float distance_y = pcl::euclideanDistance(pca_end_points_y_out->points[0], pca_end_points_y_out->points[1]);
                            std::cout << "Diameter (y) equals: " << distance_y << " m." << std::endl;

                            Cloud_Type minPoint_x, maxPoint_x;
                            pcl::PointCloud<Cloud_Type>::Ptr pca_end_points_x(new pcl::PointCloud<Cloud_Type>);
                            pcl::PointCloud<Cloud_Type>::Ptr pca_end_points_x_out(new pcl::PointCloud<Cloud_Type>);
                            //minPoint_x.x = minPoint.x, minPoint_x.y = 0.5*(minPoint.y+maxPoint.y), minPoint_x.z = 0.5*(minPoint.z+maxPoint.z);
                            minPoint_x.x = minPoint.x, minPoint_x.y = 0.0, minPoint_x.z = 0.0;
                            //maxPoint_x.x = maxPoint.x, maxPoint_x.y = 0.5*(minPoint.y+maxPoint.y), maxPoint_x.z = 0.5*(minPoint.z+maxPoint.z);
                            maxPoint_x.x = maxPoint.x, maxPoint_x.y = 0.0, maxPoint_x.z = 0.0;
                            pca_end_points_x->points.push_back(minPoint_x);
                            pca_end_points_x->points.push_back(maxPoint_x);
                            pcl::transformPointCloud(*pca_end_points_x, *pca_end_points_x_out, bboxTransform, bboxQuaternion);
                            renderPointCloud(viewer, pca_end_points_x_out, "pca_end_points_x" + zero_padding(std::to_string(bBox.boxID), 3) + zero_padding(std::to_string(clusterId), 4), Color(1, 1, 1), 10);
                            float distance_x = pcl::euclideanDistance(pca_end_points_x_out->points[0], pca_end_points_x_out->points[1]);
                            std::cout << "Diameter (x) equals: " << distance_x << " m." << std::endl;
                            /*
                            pcl::PointCloud<Cloud_Type>::Ptr pca_end_points(new pcl::PointCloud<Cloud_Type>);
                            pcl::PointCloud<Cloud_Type>::Ptr pca_end_points_out(new pcl::PointCloud<Cloud_Type>);
                            pca_end_points->points.push_back(minPoint);
                            pca_end_points->points.push_back(maxPoint);
                            pcl::transformPointCloud(*pca_end_points, *pca_end_points_out, bboxTransform, bboxQuaternion);
                            renderPointCloud(viewer, pca_end_points_out, "pca_end_points" + zero_padding(std::to_string(bBox.boxID), 3) + zero_padding(std::to_string(clusterId), 4), Color(0, 0, 1), 10);
                            //viewer->addLine(pca_end_points_out->points[0].x, pca_end_points_out->points[0].y, pca_end_points_out->points[0].z, pca_end_points_out->points[0].x- pca_end_points_out->points[1].x, pca_end_points_out->points[0].y - pca_end_points_out->points[1].y, pca_end_points_out->points[0].z - pca_end_points_out->points[1].z);
                            */

                            ++clusterId;
                            viewer->spinOnce(); // Allow user to rotate point cloud and view it

                        }

                    }
                }
            }
        }
        auto endTime = std::chrono::steady_clock::now();
        auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime_pc_processing);
        std::cout << "Point cloud processing took " << elapsedTime.count() << " milliseconds" << std::endl;

    }

    return EXIT_SUCCESS;
}
catch (const rs2::error& e)
{
    std::cerr << "RealSense error calling " << e.get_failed_function() << "(" << e.get_failed_args() << "):\n    " << e.what() << std::endl;
    system("pause");
    return EXIT_FAILURE;
}
catch (const std::exception& e)
{
    std::cerr << e.what() << std::endl;
    system("pause");
    return EXIT_FAILURE;
}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file