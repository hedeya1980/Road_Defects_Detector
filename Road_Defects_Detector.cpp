// Road_Defects_Detector.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#define CV_IMWRITE_JPEG_QUALITY 1

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

#include <filesystem>
#include <pcl/kdtree/kdtree_flann.h>

namespace fs = std::filesystem;
std::string test_input_path = "G:/My Drive/examples/Egypt_examples";
//std::string test_output_path = "C:/Users/hedey/source/repos/Road_Defects_Detector/Test_output";


// object detection
/*
string yoloBasePath = "C:/Users/hedey/OneDrive/Documents/Research_papers/STDF/yolo/";
string yoloClassesFile = yoloBasePath + "Proj_obj-13.names";
string yoloModelConfiguration = yoloBasePath + "yolov3_proj.cfg";
string yoloModelWeights = yoloBasePath + "yolov3_proj_best_2541-mAP.weights";
*/

//string yoloClassesFile = yoloBasePath + "coco.names";
//string yoloClassesFile = yoloBasePath + "obj.names";
//string yoloModelConfiguration = yoloBasePath + "yolov3.cfg";
//string yoloModelConfiguration = yoloBasePath + "yolov4_custom_train.cfg";
//string yoloModelWeights = yoloBasePath + "yolov3.weights";
//string yoloModelWeights = yoloBasePath + "yolov3_proj_last_357.weights";
//string yoloModelWeights = yoloBasePath + "yolov4_custom_train_best.weights";
//string yoloModelWeights = yoloBasePath + "yolov3_proj_last_2000.weights";
//string yoloModelWeights = yoloBasePath + "yolov3_proj_last_2700.weights";
//string yoloModelWeights = yoloBasePath + "yolov3_proj_last_4000.weights";
//string yoloModelWeights = yoloBasePath + "yolov3_proj_last_6700.weights";
//string yoloModelWeights = yoloBasePath + "yolov3_proj_10000.weights";
//string yoloModelWeights = yoloBasePath + "yolov3_proj_last_12000.weights";

//string bag_path = "E:/20210925_172641.bag";//28G
//string bag_path = "C:/Users/hedey/Documents/eng.hedeya/eslam test.bag";
//string bag_path = "C:/Users/hedey/Documents/eng.hedeya/20211023_111231.bag";
string bag_path = "C:/Users/hedey/Documents/20211008_130710.bag";//important: 2 good potholes
//string bag_path = "C:/Users/hedey/Documents/20211008_130835.bag";//important: shadows - 1 pothole
//string bag_path = "C:/Users/hedey/Documents/20211127_150031.bag";//last - cracks
//string bag_path = "C:/Users/hedey/Documents/20211008_130932.bag";
//string bag_path = "D:/Eslam/20211023_111231.bag";
//string bag_path = "D:/Eslam/eslam test.bag";
//string bag_path = "C:/Users/hedey/Documents/20211008_132819.bag";
//string bag_path = "C:/Users/hedey/Documents/20211008_133012.bag";
//string bag_path = "C:/Users/hedey/Documents/20211008_130436.bag";
//string bag_path = "C:/Users/hedey/Documents/20210925_172641.bag";//28G
//cfg.enable_device_from_file("C:/Users/hedey/Downloads/d435i_walk_around.bag");
//cfg.enable_device_from_file("C:/Users/hedey/Downloads/d435i_walking.bag");
//cfg.enable_device_from_file("C:/Users/hedey/Documents/20210912_171555.bag");
//cfg.enable_device_from_file("C:/Users/hedey/Documents/20211008_130436.bag");
//cfg.enable_device_from_file("C:/Users/hedey/Documents/20210925_172641.bag");//28G
//cfg.enable_device_from_file("C:/Users/hedey/Documents/20211008_133012.bag");
//cfg.enable_device_from_file("C:/Users/hedey/Documents/20211008_132819.bag");

template<typename PointT>
float average_distance(typename pcl::PointCloud<PointT>::Ptr inputCloud)
{
    float totalcount = inputCloud->width * inputCloud->height;
    //std::cout << "total count: " << totalcount << std::endl;//****
    //vector<float> EuclidianDistance = new float[totalcount];
    vector<float> EuclidianDistance(totalcount);

    pcl::KdTreeFLANN<PointT> kdtree;

    //kdtree.setInputCloud(inputCloud);
    kdtree.setInputCloud(inputCloud);

    int K = 2; //first will be the distance with point it self and second will the nearest point that's why "2"

    std::vector<int> pointIdxNKNSearch(K);
    std::vector<float> pointNKNSquaredDistance(K);

    for (int i = 0; i < totalcount; ++i)
    {
        /*
        std::cout << "\nK nearest neighbor search at (" << inputCloud->points[i].x
            << " " << inputCloud->points[i].y
            << " " << inputCloud->points[i].z
            << ") with K=" << K << std::endl;
        */

        if (kdtree.nearestKSearch(inputCloud->points[i], K, pointIdxNKNSearch, pointNKNSquaredDistance) > 0)
        {
            for (size_t j = 0; j < pointIdxNKNSearch.size(); ++j)
            {
                //saving all the distance in Vector
                EuclidianDistance[i] = pointNKNSquaredDistance[j];
                //std::cout << "EuclidianDistance " << i << " : " << EuclidianDistance[i] << std::endl;//****

            }
        }
    }

    float totalDistance, meanDistance;
    totalDistance = 0;
    for (int i = 0; i < totalcount; i++)
    {
        //accumulating all distances
        totalDistance = totalDistance + EuclidianDistance[i];
        //std::cout << "totalDistance: " << totalDistance << std::endl;//****
    }

    
    //calculating the mean distance
    meanDistance = totalDistance / totalcount;

    //freeing the allocated memory      
    //delete[] EuclidianDistance;
    std::cout << "Mean Distance is: " << meanDistance << std::endl;
    return meanDistance;
}

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
    //int distance = 5;
    int distance = 0;
    //int distance = 1;

    switch (setAngle)
    {
    case XY: viewer->setCameraPosition(distance, distance, distance, -1, 0, 1); break;
        //case TopDown: viewer->setCameraPosition(0, -distance, 0, 0, -1, 1); break;
    case TopDown: viewer->setCameraPosition(0, -distance, 0, -1, -1, 1); break;
        //case TopDown: viewer->setCameraPosition(-distance, -distance, -distance, 0, -1, 1); break;
    case Side: viewer->setCameraPosition(0, -distance, 0, 0, 0, 1); break;
    //case FPS: viewer->setCameraPosition(0, 0, -10, 0, -1, 0); break;
    case FPS: viewer->setCameraPosition(0, 0, -distance, 0, -1, 0); break;
    //case FPS: viewer->setCameraPosition(0, 0, -distance, 0, 0, -1); break;
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
        r = float(std::min(255, (int)(255 * (2 - abs((float(dp_value) - float(minimum)) / float(minimum)))))) / 255.0;
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
        b = float(std::min(255, (int)(255 * (2 - abs((float(dp_value) - float(maximum)) / float(maximum)))))) / 255.0;
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

template<typename PointT>
void fit_plane(PointT p1, PointT p2, PointT p3, float& a, float& b, float& c, float& d)
{
    float x1, x2, x3, y1, y2, y3, z1, z2, z3;
    x1 = p1.x, y1 = p1.y, z1 = p1.z;
    x2 = p2.x, y2 = p2.y, z2 = p2.z;
    x3 = p3.x, y3 = p3.y, z3 = p3.z;
    a = (y2 - y1) * (z3 - z1) - (z2 - z1) * (y3 - y1);
    //float b = x2 - x1;
    b = (z2 - z1) * (x3 - x1) - (z3 - z1) * (x2 - x1);
    //float c = x1 * y2 - x2 * y1;
    c = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1);
    d = -1 * (a * x1 + b * y1 + c * z1);

}


int main(int argc, char* argv[]) try
{
    
    string yoloBasePath = argv[1];
    string yoloClassesFile = yoloBasePath + "Proj_obj-13.names";
    string yoloModelConfiguration = yoloBasePath + "yolov3_proj.cfg";
    string yoloModelWeights = yoloBasePath + argv[2];
    
    if (argc > 0)
    {
        for (int i = 0; i < argc; i++)
        {
            std::cout << argv[i] << std::endl;
        }
    }


    for (const auto& entry : fs::directory_iterator(test_input_path))
        std::cout << entry.path() << std::endl;
    // load class names from file
    vector<string> classes;
    ifstream ifs(yoloClassesFile.c_str());
    string line;
    while (getline(ifs, line)) classes.push_back(line);

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
    cv::namedWindow("RGB_out", cv::WINDOW_NORMAL);
    
    //namedWindow(color_frame, WINDOW_NORMAL);
    Cloud_Type minPt, maxPt;

    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
    pcl::PointCloud<Cloud_Type>::Ptr newCloud(new pcl::PointCloud<Cloud_Type>);
    CameraAngle setAngle = FPS; //XY, FPS, Side, TopDown
    initCamera(setAngle, viewer);

    boost::shared_ptr<pcl::visualization::PCLVisualizer> openCloud;

    int f = 0;
    int first_f = 0;
    
    
    //for (int k = 0; k < 100; k++)
    //for (int k = 0; k < 250; k++)
    //for (int k = 0; k < 1200; k++)
    //{
    //    data = pipe.wait_for_frames();
    //    f += 1;
    //}
    

    //while (waitKey(1) < 0 && getWindowProperty(window_name, WND_PROP_AUTOSIZE) >= 0)
    while (true)
        //while(!viewer->wasStopped()) && waitKey(1)< 0 && getWindowProperty(window_name, WND_PROP_AUTOSIZE) >= 0
    {

        // Define two align objects. One will be used to align
        // to depth viewport and the other to color.
        // Creating align object is an expensive operation
        // that should not be performed in the main loop
        
        for (int k = 0; k < 2; k++)
        {
            data = pipe.wait_for_frames();
            f += 1;
        }
        

        //rs2::frameset data = pipe.wait_for_frames(); // Wait for next set of frames from the camera
        data = pipe.wait_for_frames();
        f += 1;
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
        
        std::cout << "-----------------------------------------------------------------------" << std::endl;

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
        cv::Mat rgb_out = imageRGB.clone();

        // Update the window with new data
        imshow(depth_frame, image);
        //imshow(color_frame, imageRGB);

        float confThreshold = 0.2;//.9, 0.5
        float nmsThreshold = 0.4;//0.5
        bool bVis = true;
        std::vector<BoundingBox> bBoxes;

        //std::vector<int> compression_params;
        //compression_params.push_back(CV_IMWRITE_JPEG_QUALITY);
        //compression_params.push_back(100);

        //imwrite("C:/Users/hedey/source/repos/Road_Defects_Detector/RGB_images/"+ zero_padding(std::to_string(f), 5) +".jpg", imageRGB);
        imwrite(argv[3]+ zero_padding(std::to_string(f), 5) +".jpg", imageRGB);
        //detectObjects((dataBuffer.end() - 1)->cameraImg, (dataBuffer.end() - 1)->boundingBoxes, confThreshold, nmsThreshold,
        detectObjects(f,imageRGB, bBoxes, confThreshold, nmsThreshold,
            yoloBasePath, classes, yoloModelConfiguration, yoloModelWeights, bVis);

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
        if (first_f == 0)
        {
            std::cout << "Color Frame Width: " << w_rgb << ", " << "Color Frame Height: " << h_rgb << std::endl;
            std::cout << "Original Depth Frame Width: " << w_pc << ", " << "Original Depth Frame Height: " << h_pc << std::endl;
            std::cout << "Aligned Depth Frame Width: " << w << ", " << "Aligned Depth Frame Height: " << h << std::endl;
            first_f = 1;
        }
        pcl::getMinMax3D(*cloud, minPt, maxPt);
        //pcl::getMinMax3D(*newCloud, minPt, maxPt);
        std::cout << "Max x: " << maxPt.x << std::endl;
        std::cout << "Max y: " << maxPt.y << std::endl;
        std::cout << "Max z: " << maxPt.z << std::endl;
        std::cout << "Min x: " << minPt.x << std::endl;
        std::cout << "Min y: " << minPt.y << std::endl;
        std::cout << "Min z: " << minPt.z << std::endl;

        // Clear viewer
        viewer->removeAllPointClouds();
        viewer->removeAllShapes();
        // Load pcd and run obstacle detection process
        
        renderPointCloud(viewer, cloud, "bagCloud");
        
        //renderPointCloud(viewer, newCloud, "bagCloud");
        viewer->spinOnce(); // Allow user to rotate point cloud and view it
        if (bBoxes.size() > 0)
        {
            for (auto bBox : bBoxes)
            {
                auto startTime_bb_conversion_projection = std::chrono::steady_clock::now();
                float Point3d_tl[3], Point3d_tr[3], Point3d_bl[3], Point3d_br[3], Point3d_cc[3];
                float Point3d_tl_depth[3], Point3d_tr_depth[3], Point3d_bl_depth[3], Point3d_br_depth[3], Point3d_cc_depth[3];
                int x_tl = std::max(bBox.roi.x, 0) , x_tr = std::min(bBox.roi.x + bBox.roi.width, w_rgb-1), x_bl = std::max(bBox.roi.x, 0), x_br = std::min(bBox.roi.x + bBox.roi.width, w_rgb-1);
                //int x_tl = std::max(bBox.roi.x, 10), x_tr = std::min(bBox.roi.x + bBox.roi.width, w_rgb - 10), x_bl = std::max(bBox.roi.x, 10), x_br = std::min(bBox.roi.x + bBox.roi.width, w_rgb - 10);
                int y_tl = std::max(bBox.roi.y, 0), y_tr = std::max(bBox.roi.y, 0), y_bl = std::min(bBox.roi.y + bBox.roi.height, h_rgb-1), y_br = std::min(bBox.roi.y + bBox.roi.height, h_rgb-1);
                //int y_tl = std::max(bBox.roi.y, 10), y_tr = std::max(bBox.roi.y, 10), y_bl = std::min(bBox.roi.y + bBox.roi.height, h_rgb - 10), y_br = std::min(bBox.roi.y + bBox.roi.height, h_rgb - 10);
                int x_cc = int(x_tl + (x_br - x_tl) / 2);
                int y_cc = int(y_tl + (y_br - y_tl) / 2);
                std::cout <<"tl: "<< "(" << x_tl << ", " << y_tl << ")" << ", br: " << "(" << x_br << ", " << y_br << ")" << ", cc: " << "(" << x_cc << ", " << y_cc << ")" << std::endl;
                
                float pixel_tl[2] = { float(x_tl),float(y_tl) };
                float pixel_tr[2] = { float(x_tr),float(y_tr) };
                float pixel_bl[2] = { float(x_bl),float(y_bl) };
                float pixel_br[2] = { float(x_br),float(y_br) };
                float pixel_cc[2] = { float(x_cc),float(y_cc) };

                float pixel_distance_in_meters_tl = depth.get_distance(x_tl, y_tl);
                float pixel_distance_in_meters_tr = depth.get_distance(x_tr, y_tr);
                float pixel_distance_in_meters_bl = depth.get_distance(x_bl, y_bl);
                float pixel_distance_in_meters_br = depth.get_distance(x_br, y_br);
                float pixel_distance_in_meters_cc = depth.get_distance(x_cc, y_cc);
                std::cout << "*** Distance to center of Bbox is: " << pixel_distance_in_meters_cc << " m. ***" << std::endl;

                rs2_deproject_pixel_to_point(Point3d_tl, &color_intrin, pixel_tl, pixel_distance_in_meters_tl);
                rs2_transform_point_to_point(Point3d_tl_depth, &color_extrin_to_depth, Point3d_tl);
                std::cout << "tl_depth: " << Point3d_tl_depth[0] << ", " << Point3d_tl_depth[1] << ", " << Point3d_tl_depth[2] << ", " << std::endl;
                rs2_deproject_pixel_to_point(Point3d_tr, &color_intrin, pixel_tr, pixel_distance_in_meters_tr);
                rs2_transform_point_to_point(Point3d_tr_depth, &color_extrin_to_depth, Point3d_tr);
                std::cout << "tr_depth: " << Point3d_tr_depth[0] << ", " << Point3d_tr_depth[1] << ", " << Point3d_tr_depth[2] << ", " << std::endl;
                rs2_deproject_pixel_to_point(Point3d_bl, &color_intrin, pixel_bl, pixel_distance_in_meters_bl);
                rs2_transform_point_to_point(Point3d_bl_depth, &color_extrin_to_depth, Point3d_bl);
                std::cout << "bl_depth: " << Point3d_bl_depth[0] << ", " << Point3d_bl_depth[1] << ", " << Point3d_bl_depth[2] << ", " << std::endl;
                rs2_deproject_pixel_to_point(Point3d_br, &color_intrin, pixel_br, pixel_distance_in_meters_br);
                rs2_transform_point_to_point(Point3d_br_depth, &color_extrin_to_depth, Point3d_br);
                std::cout << "br_depth: " << Point3d_br_depth[0] << ", " << Point3d_br_depth[1] << ", " << Point3d_br_depth[2] << ", " << std::endl;
                rs2_deproject_pixel_to_point(Point3d_cc, &color_intrin, pixel_cc, pixel_distance_in_meters_cc);
                rs2_transform_point_to_point(Point3d_cc_depth, &color_extrin_to_depth, Point3d_cc);
                std::cout << "cc_depth: " << Point3d_cc_depth[0] << ", " << Point3d_cc_depth[1] << ", " << Point3d_cc_depth[2] << ", " << std::endl;

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
                float projected_xmin, projected_ymin, projected_zmin, projected_xmax, projected_ymax, projected_zmax;
                if (classes[bBox.classID] == "Potholes")
                {
                    projected_xmin = std::max(std::min(Point3d_bl_depth[0], Point3d_tl_depth[0]), minPt.x);
                    projected_ymin = std::max(std::min(Point3d_tl_depth[1], Point3d_tr_depth[1]), minPt.y);
                    projected_zmin = std::max(std::min(Point3d_bl_depth[2], Point3d_br_depth[2]), minPt.z);
                    projected_xmax = std::min(std::max(Point3d_tr_depth[0], Point3d_br_depth[0]), maxPt.x);
                    projected_ymax = std::min(std::max(Point3d_bl_depth[1], Point3d_br_depth[1]), maxPt.y);
                    projected_zmax = std::min(std::max(Point3d_tl_depth[2], Point3d_tr_depth[2]), maxPt.z);
                }
                else
                {
                    projected_xmin = std::max(std::min(Point3d_bl_depth[0], Point3d_tl_depth[0]), minPt.x);
                    projected_ymin = std::max(std::max(Point3d_tl_depth[1], Point3d_tr_depth[1]), minPt.y);
                    projected_zmin = std::max(std::min(Point3d_bl_depth[2], Point3d_br_depth[2]), minPt.z);
                    projected_xmax = std::min(std::max(Point3d_tr_depth[0], Point3d_br_depth[0]), maxPt.x);
                    projected_ymax = std::min(std::min(Point3d_bl_depth[1], Point3d_br_depth[1]), maxPt.y);
                    projected_zmax = std::min(std::min(Point3d_tl_depth[2], Point3d_tr_depth[2]), maxPt.z);
                }
                float x_dist = projected_xmax - projected_xmin;
                float y_dist = projected_ymax - projected_ymin;
                float z_dist = projected_zmax - projected_zmin;
                float ROImin_x = Point3d_cc_depth[0] - x_dist / 2;
                float ROImin_y = Point3d_cc_depth[1] - y_dist / 2;
                //float ROImin_z = Point3d_cc_depth[2] - z_dist / 2;
                float ROImin_z = projected_zmin;
                float ROImax_x = Point3d_cc_depth[0] + x_dist / 2;
                float ROImax_y = Point3d_cc_depth[1] + y_dist / 2;
                //float ROImax_z = Point3d_cc_depth[2] + z_dist / 2;
                float ROImax_z = projected_zmax;

                float ROI_xmin = std::max(std::min(ROImin_x, ROImax_x), minPt.x);
                float ROI_ymin = std::max(std::min(ROImin_y, ROImax_y), minPt.y);
                float ROI_zmin = projected_zmin;
                float ROI_xmax = std::min(std::max(ROImin_x, ROImax_x), maxPt.x);
                float ROI_ymax = std::min(std::max(ROImin_y, ROImax_y), maxPt.y);
                float ROI_zmax = projected_zmax;

                //========================================
                // Filter PointCloud (PassThrough Method)
                //========================================
                /*
                pcl::PassThrough<Cloud_Type> Cloud_Filter; // Create the filtering object
                Cloud_Filter.setInputCloud(cloud);           // Input generated cloud to filter
                Cloud_Filter.setFilterFieldName("z");        // Set field name to Z-coordinate
                Cloud_Filter.setFilterLimits(ROI_zmin, ROI_zmax);      // Set accepted interval values
                Cloud_Filter.filter(*newCloud);              // Filtered Cloud Outputted
                */

                
                pcl::PointCloud<Cloud_Type>::Ptr filterCloud = pointProcessorI->FilterCloud(cloud, 0.1f, Eigen::Vector4f(ROI_xmin, ROI_ymin, ROI_zmin, 1), Eigen::Vector4f(ROI_xmax, ROI_ymax, ROI_zmax, 1)); //Best
                float avg_dist = average_distance<Cloud_Type>(cloud);
                /*
                pcl::PointCloud<Cloud_Type>::Ptr filterCloud;
                pcl::PointCloud<Cloud_Type>::Ptr filterCloud = pointProcessorI->FilterCloud(newCloud, 0.1f, Eigen::Vector4f(ROI_xmin, ROI_ymin, ROI_zmin, 1), Eigen::Vector4f(ROI_xmax, ROI_ymax, ROI_zmax, 1)); //Best
                */
                //filterCloud = pointProcessorI->FilterCloud(newCloud, 0.1f, Eigen::Vector4f(ROI_xmin, ROI_ymin, ROI_zmin, 1), Eigen::Vector4f(ROI_xmax, ROI_ymax, ROI_zmax, 1)); //Best
                std::cout << "ROI min: " << ROI_xmin << ", " << ROI_ymin << ", " << ROI_zmin << std::endl;
                std::cout << "ROI max: " << ROI_xmax << ", " << ROI_ymax << ", " << ROI_zmax << std::endl;
                
                pcl::PointCloud<Cloud_Type>::Ptr ROI_min_max(new pcl::PointCloud<Cloud_Type>);
                Cloud_Type ROImin, ROImax;
                //ROImin.x = ROImin_x;
                ROImin.x = ROI_xmin;
                //ROImin.y = ROImin_y;
                ROImin.y = ROI_ymin;
                //ROImin.z = ROImin_z;
                ROImin.z = ROI_zmin;
                //ROImax.x = ROImax_x;
                ROImax.x = ROI_xmax;
                //ROImax.y = ROImax_y;
                ROImax.y = ROI_ymax;
                //ROImax.z = projected_zmax;
                ROImax.z = ROI_zmax;
                ROI_min_max->points.push_back(ROImin);
                ROI_min_max->points.push_back(ROImax);
                
                //renderPointCloud(viewer, ROI_min_max, "ROIminmax" + zero_padding(std::to_string(bBox.boxID), 3), Color(0, 1, 1), 10);

                if (filterCloud->points.size() > 50)
                {
                    renderPointCloud(viewer, filterCloud, "filterCloud" + zero_padding(std::to_string(bBox.boxID), 3), Color(1, 0, 1));
                    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
                    std::pair<typename pcl::PointCloud<Cloud_Type>::Ptr, typename pcl::PointCloud<Cloud_Type>::Ptr> segmentCloud;
                    //std::pair<typename pcl::PointCloud<RGB_Cloud>::Ptr, typename pcl::PointCloud<RGB_Cloud>::Ptr> segmentCloud = pointProcessorI->SegmentPlane(cloud, 3000, 0.01, coefficients); // Possible pavement defects //.01
                    ////if (classes[bBox.classID] == "Potholes")
                    ////{
                        //std::pair<typename pcl::PointCloud<Cloud_Type>::Ptr, typename pcl::PointCloud<Cloud_Type>::Ptr> segmentCloud = pointProcessorI->SegmentPlane(filterCloud, 5000, 0.01, coefficients); // Possible pavement defects //.01
                        /*
                        if(pixel_distance_in_meters_cc>2 && pixel_distance_in_meters_cc<5)
                            segmentCloud = pointProcessorI->SegmentPlane(filterCloud, 5000, 0.01, coefficients); // Possible pavement defects //.01
                        else if(pixel_distance_in_meters_cc<=2)
                            segmentCloud = pointProcessorI->SegmentPlane(filterCloud, 5000, 0.005, coefficients);
                        else
                            segmentCloud = pointProcessorI->SegmentPlane(filterCloud, 5000, 0.015, coefficients);
                        */

                        //segmentCloud = pointProcessorI->SegmentPlane(filterCloud, 3000, 0.03, coefficients); // Possible pavement defects //.01
                        //segmentCloud = pointProcessorI->SegmentPlane(filterCloud, 3000, 26.557* avg_dist+0.0095, coefficients); // Possible pavement defects //.01
                        segmentCloud = pointProcessorI->SegmentPlane(filterCloud, 5000, pixel_distance_in_meters_cc/300, coefficients); // Possible pavement defects //.01
                    ////}
                    ////else
                    ////{
                        //std::pair<typename pcl::PointCloud<Cloud_Type>::Ptr, typename pcl::PointCloud<Cloud_Type>::Ptr> segmentCloud = pointProcessorI->SegmentPlane(filterCloud, 5000, 0.005, coefficients); // Possible pavement defects //.01
                        ////segmentCloud = pointProcessorI->SegmentPlane(filterCloud, 2000, 0.005, coefficients); // Possible pavement defects //.01
                    ////}
                    std::cerr << "Model coefficients: " << coefficients->values[0] << " "
                        << coefficients->values[1] << " "
                        << coefficients->values[2] << " "
                        << coefficients->values[3] << std::endl;
                    //std::pair<typename pcl::PointCloud<PointT>::Ptr, typename pcl::PointCloud<PointT>::Ptr> segmentCloud = pointProcessorI->SegmentPlane(filterCloud, 3000, 0.05); // Possible pavement defects
                    //renderPointCloud(viewer, segmentCloud.first, "obstCloud", Color(1, 0, 1));
                    //renderPointCloud(viewer, segmentCloud.second, "planeCloud2" + zero_padding(std::to_string(bBox.boxID), 3), Color(1, 1, 0));
                    
                    renderPointCloud(viewer, segmentCloud.second, "planeCloud2" + zero_padding(std::to_string(bBox.boxID), 3), Color(0, 1, 0));
                    
                    std::vector<pcl::PointCloud<Cloud_Type>::Ptr> cloudClusters;
                    if (segmentCloud.first->points.size() > 0)
                    {
                        //std::vector<pcl::PointCloud<Cloud_Type>::Ptr> cloudClusters = pointProcessorI->Clustering(segmentCloud.first, 0.01, 1000, 50000);
                        //std::vector<pcl::PointCloud<Cloud_Type>::Ptr> cloudClusters = pointProcessorI->Clustering(segmentCloud.first, 0.02, 3000, 50000);
                        if (classes[bBox.classID] == "Potholes")
                        {
                            //std::vector<pcl::PointCloud<Cloud_Type>::Ptr> cloudClusters = pointProcessorI->Clustering(segmentCloud.first, 0.02, 1000, 50000);
                            //cloudClusters = pointProcessorI->Clustering(segmentCloud.first, 0.02, 3500, 50000);
                            //cloudClusters = pointProcessorI->Clustering(segmentCloud.first, 100*avg_dist, 100, 50000);
                            //cloudClusters = pointProcessorI->Clustering(segmentCloud.first, 2*avg_dist, 500, 50000);
                            //cloudClusters = pointProcessorI->Clustering(segmentCloud.first, 1.5*pixel_distance_in_meters_cc * avg_dist, 500, 50000);//best for Eslam's bag file
                            //cloudClusters = pointProcessorI->Clustering(segmentCloud.first, pixel_distance_in_meters_cc* pixel_distance_in_meters_cc * pixel_distance_in_meters_cc * avg_dist, 500, 50000);
                            //cloudClusters = pointProcessorI->Clustering(segmentCloud.first, 15 * pixel_distance_in_meters_cc * avg_dist, 500, 50000);
                            //cloudClusters = pointProcessorI->Clustering(segmentCloud.first, 0.01 * pixel_distance_in_meters_cc , 500, 50000);//best on 1/11/2022
                            cloudClusters = pointProcessorI->Clustering(segmentCloud.first, 0.01 * pixel_distance_in_meters_cc, 500, 50000);
                            /*
                            }
                            else
                            {
                                cloudClusters = pointProcessorI->Clustering(segmentCloud.first, 0.01, 200, 2000);
                            }
                            */
                            int clusterId = 0;
                            //std::vector<Color> colors = { Color(1,0,0), Color(1,1,0), Color(0,0,1) };
                            int i = 0;
                            for (pcl::PointCloud<Cloud_Type>::Ptr cluster : cloudClusters)
                            {
                                /*
                                if (i == 0)
                                {
                                    viewer->removeAllPointClouds();
                                    viewer->removeAllShapes();
                                    // Load pcd and run obstacle detection process
                                    //renderPointCloud(viewer, filterCloud, "filterCloud" + zero_padding(std::to_string(bBox.boxID), 3), Color(1, 0, 1));
                                    //renderPointCloud(viewer, segmentCloud.second, "planeCloud2" + zero_padding(std::to_string(bBox.boxID), 3), Color(0, 1, 0));
                                    viewer->spinOnce(); // Allow user to rotate point cloud and view it
                                    i += 1;
                                }
                                */
                                std::cout << "--------------------------------------" << std::endl;
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
                                else if (min_depth != 10000 && max_depth != -10000)
                                {
                                    std::cout << "Cluster's depth ranges from: " << min_depth << " to: " << max_depth << std::endl;//<< ", Max Depth is: " << abs(floor(min_depth * 1000.0 + .5)) << " mm." 
                                    //std::cout << "Defect's height ranges from: " << min_depth << " to: " << max_depth << ", Max Height is: " << abs(floor(max_depth * 1000.0 + .5)) << " mm." << std::endl;
                                }
                                else
                                    std::cout << "Cluster out of appropriate range: " << min_depth << " to: " << max_depth << std::endl;

                                pcl::PointCloud<Cloud_Type>::Ptr min_depth_Cloud(new pcl::PointCloud<Cloud_Type>);
                                pcl::PointCloud<Cloud_Type>::Ptr max_depth_Cloud(new pcl::PointCloud<Cloud_Type>);
                                int rounded_max = floor(max_depth * 1000.0 + .5);
                                int rounded_min = floor(min_depth * 1000.0 + .5);
                                int max_level = 150;
                                int min_level = -110;
                                int j = 0;
                                std::vector<int> depths;
                                //int capacity = 0;
                                int pos_capacity = 0;
                                int neg_capacity = 0;
                                for (auto it = depth_estimation.begin(); it != depth_estimation.end(); ++it)
                                {
                                    pcl::PointCloud<Cloud_Type>::Ptr depth_contour_Cloud(new pcl::PointCloud<Cloud_Type>);
                                    if (min_depth != 10000 && max_depth != -10000)
                                    {
                                        std::tuple<float, float, float> RGB_Color;
                                        //std::cout << int(it->first) << std::endl;
                                        RGB_Color = RGB_Heatmap(min_level, max_level, int(it->first));
                                        //if ((it->first > 0 && it->first == rounded_min) || (it->first < 0 && it->first == rounded_max))
                                        if ((it->first > 0 && it->first == rounded_min) || (it->first < 0 && it->first == rounded_max))
                                        {
                                            for (Cloud_Type point : it->second)
                                                min_depth_Cloud->points.push_back(point);

                                            renderPointCloud(viewer, min_depth_Cloud, "minDepth" + zero_padding(std::to_string(bBox.boxID), 3) + zero_padding(std::to_string(clusterId), 4) + zero_padding(std::to_string(j), 4), Color(get<0>(RGB_Color), get<1>(RGB_Color), get<2>(RGB_Color)));
                                            //std::cout << "Size of nearest contour to plane is: " << min_depth_Cloud->points.size() << std::endl;
                                            //std::cout << "Colors: " << get<0>(RGB_Color) << ", " << get<1>(RGB_Color) << ", " << get<2>(RGB_Color) << std::endl;
                                        }
                                        else if ((it->first > 0 && it->first == rounded_max) || (it->first < 0 && it->first == rounded_min))
                                        {
                                            for (Cloud_Type point : it->second)
                                                max_depth_Cloud->points.push_back(point);

                                            renderPointCloud(viewer, max_depth_Cloud, "maxDepth" + zero_padding(std::to_string(bBox.boxID), 3) + zero_padding(std::to_string(clusterId), 4) + zero_padding(std::to_string(j), 4), Color(get<0>(RGB_Color), get<1>(RGB_Color), get<2>(RGB_Color)));
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
                                        //if (min_depth < 0 && max_depth < 0)
                                        if (it->first > 0)
                                            pos_capacity += (it->first) * depth_estimation[it->first].size();
                                        else if (it->first < 0)
                                            neg_capacity += (it->first) * depth_estimation[it->first].size();
                                    }
                                    j++;
                                }
                                //std::cout << "Pothole's total capacity in mm is: " << capacity <<"." << std::endl;
                                std::cout << "Defect's total positive capacity in mm is: " << pos_capacity << "." << std::endl;
                                std::cout << "Defect's total negative capacity in mm is: " << neg_capacity << "." << std::endl;


                                pcl::PointCloud<Cloud_Type>::Ptr cloud_projected(new pcl::PointCloud<Cloud_Type>);
                                // Project the model inliers
                                pcl::ProjectInliers<Cloud_Type> proj;
                                proj.setModelType(pcl::SACMODEL_PLANE);
                                // proj.setIndices (inliers);
                                proj.setInputCloud(cluster);
                                proj.setModelCoefficients(coefficients);
                                proj.filter(*cloud_projected);
                                std::cerr << "Projected cluster has: "
                                    << cloud_projected->size() << " points." << std::endl;

                                // Create a Concave Hull representation of the projected inliers
                                pcl::PointCloud<Cloud_Type>::Ptr cloud_hull(new pcl::PointCloud<Cloud_Type>);
                                pcl::ConcaveHull<Cloud_Type> chull;
                                chull.setInputCloud(cloud_projected);
                                chull.setAlpha(0.1);
                                chull.reconstruct(*cloud_hull);

                                std::cerr << "Concave hull has: " << cloud_hull->size()
                                    << " points." << std::endl;
                                renderPointCloud(viewer, cloud_hull, "cloud_hull" + zero_padding(std::to_string(bBox.boxID), 3) + zero_padding(std::to_string(clusterId), 4), Color(0, 0, 1));
                                float concave_hull_area = calculatePolygonArea(*cloud_hull);
                                std::cout << "Area within concave hull is: " << concave_hull_area << std::endl;
                                float avg_area_per_point = concave_hull_area / cloud_projected->size();
                                //float volume = avg_area_per_point * capacity/1000;
                                //std::cout << "Volume is: " << volume << " (" << 1000 * volume << " litres.)" << std::endl;
                                float pos_volume = avg_area_per_point * pos_capacity / 1000;
                                float neg_volume = avg_area_per_point * neg_capacity / 1000;
                                std::cout << "Positive volume is: " << pos_volume << " (" << 1000 * pos_volume << " litres)." << std::endl;
                                std::cout << "Negative volume is: " << neg_volume << " (" << 1000 * neg_volume << " litres)." << std::endl;

                                //std::vector<std::array<float[2]>> rbg_hull;
                                for (Cloud_Type p_hull : cloud_hull->points)
                                {
                                    float pixel[2];
                                    float point[3] = { p_hull.x, p_hull.y, p_hull.z };
                                    //std::cout << point[0] << ", " << point[1] << ", " << point[2] <<" : "<< p_hull.x<<", "<< p_hull.y<<", "<< p_hull.z <<std::endl;
                                    float point_color[3];
                                    rs2_transform_point_to_point(point_color, &depth_extrin_to_color, point);
                                    rs2_project_point_to_pixel(pixel, &color_intrin, point_color);
                                    //rs2_project_point_to_pixel(pixel, &color_intrin, point);
                                    //rbg_hull.push_back(pixel);
                                    //cv::circle(imageRGB, Point(pixel[0], pixel[1]), 0, Scalar(255, 0, 0),-1,8,0);
                                    cv::circle(rgb_out, Point(pixel[0], pixel[1]), 0, Scalar(255, 0, 0), 6, 8, 0);
                                    //std::cout << pixel[0] << ", " << pixel[1] << std::endl;
                                    //std::cout << "tl_depth: " << Point3d_tl_depth[0] << ", " << Point3d_tl_depth[1] << ", " << Point3d_tl_depth[2] << ", " << std::endl;

                                }
                                for (Cloud_Type p_max_depth : max_depth_Cloud->points)
                                {
                                    float pixel[2];
                                    float point[3] = { p_max_depth.x, p_max_depth.y, p_max_depth.z };
                                    //std::cout << point[0] << ", " << point[1] << ", " << point[2] <<" : "<< p_hull.x<<", "<< p_hull.y<<", "<< p_hull.z <<std::endl;
                                    float point_color[3];
                                    rs2_transform_point_to_point(point_color, &depth_extrin_to_color, point);
                                    rs2_project_point_to_pixel(pixel, &color_intrin, point_color);
                                    //rs2_project_point_to_pixel(pixel, &color_intrin, point);
                                    //rbg_hull.push_back(pixel);
                                    //cv::circle(imageRGB, Point(pixel[0], pixel[1]), 0, Scalar(255, 0, 0),-1,8,0);
                                    cv::circle(rgb_out, Point(pixel[0], pixel[1]), 0, Scalar(0, 0, 255), 6, 8, 0);
                                    //std::cout << pixel[0] << ", " << pixel[1] << std::endl;
                                    //std::cout << "tl_depth: " << Point3d_tl_depth[0] << ", " << Point3d_tl_depth[1] << ", " << Point3d_tl_depth[2] << ", " << std::endl;

                                }
                                cv::imshow("RGB_out", rgb_out);
                                imwrite("C:/Users/hedey/source/repos/Road_Defects_Detector/RGB_images/" + zero_padding(std::to_string(f), 5) + "_bkprj" + ".jpg", rgb_out);

                                //}
                                //if(render_box)
                                //{
                                //Box box = pointProcessor.BoundingBox(cluster);
                                //renderBox(viewer, box, clusterId);
                                //}
                                // Compute principal directions
                                Eigen::Vector4f pcaCentroid;
                                //pcl::compute3DCentroid(*cloud_projected, pcaCentroid);
                                pcl::compute3DCentroid(*cluster, pcaCentroid);
                                Eigen::Matrix3f covariance;
                                //computeCovarianceMatrixNormalized(*cloud_projected, pcaCentroid, covariance);
                                computeCovarianceMatrixNormalized(*cluster, pcaCentroid, covariance);
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
                                //pcl::transformPointCloud(*cloud_projected, *cloudPointsProjected, projectionTransform);
                                pcl::transformPointCloud(*cluster, *cloudPointsProjected, projectionTransform);
                                // Get the minimum and maximum points of the transformed cloud.
                                Cloud_Type minPoint, maxPoint;
                                pcl::getMinMax3D(*cloudPointsProjected, minPoint, maxPoint);
                                const Eigen::Vector3f meanDiagonal = 0.5f * (maxPoint.getVector3fMap() + minPoint.getVector3fMap());

                                // Final transform
                                const Eigen::Quaternionf bboxQuaternion(eigenVectorsPCA); //Quaternions are a way to do rotations https://www.youtube.com/watch?v=mHVwd8gYLnI
                                const Eigen::Vector3f bboxTransform = eigenVectorsPCA * meanDiagonal + pcaCentroid.head<3>();
                                Eigen::Vector3f x_axis = eigenVectorsPCA.col(0);
                                //std::cout << x_axis[0] << ", " << x_axis[1] << ", " << x_axis[2] << std::endl;
                                Eigen::Vector3f y_axis = eigenVectorsPCA.col(1);
                                //viewer->addCube(bboxTransform, bboxQuaternion, maxPoint.x - minPoint.x, maxPoint.y - minPoint.y, maxPoint.z - minPoint.z, "bbox" + zero_padding(std::to_string(bBox.boxID), 3) + zero_padding(std::to_string(clusterId), 4));

                                Cloud_Type minPoint_z, maxPoint_z;
                                pcl::PointCloud<Cloud_Type>::Ptr pca_end_points_z(new pcl::PointCloud<Cloud_Type>);
                                pcl::PointCloud<Cloud_Type>::Ptr pca_end_points_z_out(new pcl::PointCloud<Cloud_Type>);
                                //minPoint_z.x = 0.0, minPoint_z.y = 0.0, minPoint_z.z = minPoint.z;
                                minPoint_z.x = maxPoint.x, minPoint_z.y = 0.0, minPoint_z.z = minPoint.z;
                                //maxPoint_z.x = 0.0, maxPoint_z.y = 0.0, maxPoint_z.z = maxPoint.z;
                                maxPoint_z.x = maxPoint.x, maxPoint_z.y = 0.0, maxPoint_z.z = maxPoint.z;
                                pca_end_points_z->points.push_back(minPoint_z);
                                pca_end_points_z->points.push_back(maxPoint_z);
                                pcl::transformPointCloud(*pca_end_points_z, *pca_end_points_z_out, bboxTransform, bboxQuaternion);
                                renderPointCloud(viewer, pca_end_points_z_out, "pca_end_points_z" + zero_padding(std::to_string(bBox.boxID), 3) + zero_padding(std::to_string(clusterId), 4), Color(0, 0, 1), 10);
                                float distance_z = pcl::euclideanDistance(pca_end_points_z_out->points[0], pca_end_points_z_out->points[1]);
                                std::cout << "Blue Diameter (z) equals: " << distance_z << " m." << std::endl;

                                Cloud_Type axis_pt, origin_pt;
                                pcl::PointCloud<Cloud_Type>::Ptr pca_prm_axs_pts(new pcl::PointCloud<Cloud_Type>);
                                pcl::PointCloud<Cloud_Type>::Ptr pca_prm_axs_pts_out(new pcl::PointCloud<Cloud_Type>);
                                origin_pt.x = 0, origin_pt.y = 0, origin_pt.z = 0;
                                pca_prm_axs_pts->points.push_back(origin_pt);
                                for (int i = 0; i < 3; i++)
                                {
                                    axis_pt.x = eigenVectorsPCA.col(i)[0], axis_pt.y = eigenVectorsPCA.col(i)[1], axis_pt.z = eigenVectorsPCA.col(i)[2];
                                    pca_prm_axs_pts->points.push_back(axis_pt);
                                }
                                pcl::transformPointCloud(*pca_prm_axs_pts, *pca_prm_axs_pts_out, bboxTransform, bboxQuaternion);
                                //renderPointCloud(viewer, pca_prm_axs_pts_out, "pca_prm_axs_pts_out" + zero_padding(std::to_string(bBox.boxID), 3) + zero_padding(std::to_string(clusterId), 4), Color(0, 1, 1), 15);

                                Cloud_Type minPoint_y, maxPoint_y;
                                pcl::PointCloud<Cloud_Type>::Ptr pca_end_points_y(new pcl::PointCloud<Cloud_Type>);
                                pcl::PointCloud<Cloud_Type>::Ptr pca_end_points_y_out(new pcl::PointCloud<Cloud_Type>);
                                //minPoint_y.x = 0.0, minPoint_y.y = minPoint.y, minPoint_y.z = 0.0;
                                minPoint_y.x = maxPoint.x, minPoint_y.y = minPoint.y, minPoint_y.z = 0.0;
                                //maxPoint_y.x = 0.0, maxPoint_y.y = maxPoint.y, maxPoint_y.z = 0.0;
                                maxPoint_y.x = maxPoint.x, maxPoint_y.y = maxPoint.y, maxPoint_y.z = 0.0;
                                pca_end_points_y->points.push_back(minPoint_y);
                                pca_end_points_y->points.push_back(maxPoint_y);
                                pcl::transformPointCloud(*pca_end_points_y, *pca_end_points_y_out, bboxTransform, bboxQuaternion);
                                renderPointCloud(viewer, pca_end_points_y_out, "pca_end_points_y" + zero_padding(std::to_string(bBox.boxID), 3) + zero_padding(std::to_string(clusterId), 4), Color(1, 0, 0), 10);
                                float distance_y = pcl::euclideanDistance(pca_end_points_y_out->points[0], pca_end_points_y_out->points[1]);
                                std::cout << "Red Diameter (y) equals: " << distance_y << " m." << std::endl;

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
                                //renderPointCloud(viewer, pca_end_points_x_out, "pca_end_points_x" + zero_padding(std::to_string(bBox.boxID), 3) + zero_padding(std::to_string(clusterId), 4), Color(1, 1, 1), 10);
                                float distance_x = pcl::euclideanDistance(pca_end_points_x_out->points[0], pca_end_points_x_out->points[1]);
                                std::cout << "White Diameter (x) equals: " << distance_x << " m." << std::endl;


                                float a1, b1, c1, d1, a2, b2, c2, d2;
                                fit_plane(pca_prm_axs_pts_out->points[0], pca_prm_axs_pts_out->points[1], pca_prm_axs_pts_out->points[3], a1, b1, c1, d1);
                                fit_plane(pca_prm_axs_pts_out->points[0], pca_prm_axs_pts_out->points[2], pca_prm_axs_pts_out->points[3], a2, b2, c2, d2);
                                pcl::PointCloud<Cloud_Type>::Ptr cluster_pts_prmry_pca_axis_1(new pcl::PointCloud<Cloud_Type>);
                                pcl::PointCloud<Cloud_Type>::Ptr cluster_pts_prmry_pca_axis_2(new pcl::PointCloud<Cloud_Type>);
                                for (Cloud_Type p : cluster->points)
                                {
                                    //float dist = fabs(a * p.x + b * p.y + c * p.z + d) / sqrt(a * a + b * b + c * c);
                                    //if (dist<=.0005)
                                    if (abs(a1 * p.x + b1 * p.y + c1 * p.z + d1) <= 1e-3)
                                    {
                                        cluster_pts_prmry_pca_axis_1->points.push_back(p);
                                    }
                                    if (abs(a2 * p.x + b2 * p.y + c2 * p.z + d2) <= 1e-3)
                                    {
                                        cluster_pts_prmry_pca_axis_2->points.push_back(p);
                                    }
                                }
                                renderPointCloud(viewer, cluster_pts_prmry_pca_axis_1, "cluster_pts_prmry_pca_axis_1" + zero_padding(std::to_string(bBox.boxID), 3) + zero_padding(std::to_string(clusterId), 4), Color(0, 0, 1), 4);
                                std::cout << "cluster_pts_prmry_pca_axis_1: " << cluster_pts_prmry_pca_axis_1->points.size() << "." << std::endl;
                                renderPointCloud(viewer, cluster_pts_prmry_pca_axis_2, "cluster_pts_prmry_pca_axis_2" + zero_padding(std::to_string(bBox.boxID), 3) + zero_padding(std::to_string(clusterId), 4), Color(0, 0, 1), 4);
                                std::cout << "cluster_pts_prmry_pca_axis_2: " << cluster_pts_prmry_pca_axis_2->points.size() << "." << std::endl;


                                if (classes[bBox.classID] == "Potholes" && rounded_min < 0)
                                {
                                    std::cout << "**************************************" << std::endl;
                                    std::cout << "Pothole attributes:" << std::endl;
                                    std::cout << "-------------------" << std::endl;
                                    //std::cout << "Distance to center of Bbox is: " << pixel_distance_in_meters_cc << " m." << std::endl;
                                    std::cout << "Maximum Pothole's Depth is: " << abs(rounded_min) << " mm." << std::endl;
                                    float avg_diameter_mm = abs(floor(0.5 * (distance_z + distance_y) * 1000.0 + .5));
                                    std::cout << "Average Pothole's Diameter is: " << avg_diameter_mm << " mm." << std::endl;
                                    std::cout << "Pothole's Area is: " << concave_hull_area << " m2." << std::endl;
                                    float pothole_count = 0.0;
                                    string svrty;
                                    if ((abs(rounded_min) > 13) && (abs(rounded_min) <= 25))
                                    {
                                        if ((avg_diameter_mm > 100) && (avg_diameter_mm <= 200))
                                        {
                                            pothole_count += 1;
                                            svrty = "L";
                                        }
                                        else if ((avg_diameter_mm > 200) && (avg_diameter_mm <= 450))
                                        {
                                            pothole_count += 1;
                                            svrty = "L";
                                        }
                                        else if ((avg_diameter_mm > 450) && (avg_diameter_mm <= 750))
                                        {
                                            pothole_count += 1;
                                            svrty = "M";
                                        }
                                        else
                                        {
                                            pothole_count += concave_hull_area / 0.5;
                                            svrty = "M";
                                        }
                                    }
                                    else if ((abs(rounded_min) > 25) && (abs(rounded_min) <= 50))
                                    {
                                        if ((avg_diameter_mm > 100) && (avg_diameter_mm <= 200))
                                        {
                                            pothole_count += 1;
                                            svrty = "L";
                                        }
                                        else if ((avg_diameter_mm > 200) && (avg_diameter_mm <= 450))
                                        {
                                            pothole_count += 1;
                                            svrty = "M";
                                        }
                                        else if ((avg_diameter_mm > 450) && (avg_diameter_mm <= 750))
                                        {
                                            pothole_count += 1;
                                            svrty = "H";
                                        }
                                        else
                                        {
                                            pothole_count += concave_hull_area / 0.5;
                                            svrty = "H";
                                        }
                                    }
                                    else
                                    {
                                        if ((avg_diameter_mm > 100) && (avg_diameter_mm <= 200))
                                        {
                                            pothole_count += 1;
                                            svrty = "M";
                                        }
                                        else if ((avg_diameter_mm > 200) && (avg_diameter_mm <= 450))
                                        {
                                            pothole_count += 1;
                                            svrty = "M";
                                        }
                                        else if ((avg_diameter_mm > 450) && (avg_diameter_mm <= 750))
                                        {
                                            pothole_count += 1;
                                            svrty = "H";
                                        }
                                        else
                                        {
                                            pothole_count += concave_hull_area / 0.5;
                                            svrty = "H";
                                        }
                                    }
                                    std::cout << "Pothole's severity is: " << svrty << std::endl;
                                    std::cout << "Pothole's equivalent count is: " << pothole_count << " (" << floor(pothole_count + .5) << " potholes)." << std::endl;
                                    std::cout << "Pothole's volume (under plane only) is: " << abs(neg_volume) << " m3 (" << 1000 * abs(neg_volume) << " litres)." << std::endl;
                                    std::cout << "**************************************" << std::endl;
                                }
                                ++clusterId;
                                viewer->spinOnce(); // Allow user to rotate point cloud and view it

                            }
                        }//
                    }
                }
            }
        }
        auto endTime = std::chrono::steady_clock::now();
        auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime_pc_processing);
        std::cout << "Point cloud processing took " << elapsedTime.count() << " milliseconds" << std::endl;
        std::cout << "-----------------------------------------------------------------------" << std::endl;

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