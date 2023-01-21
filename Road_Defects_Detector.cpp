// Road_Defects_Detector.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#define CV_IMWRITE_JPEG_QUALITY 1

#include <iostream>
//#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API
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

#include <ctime>
//#include "date/date.h"

#include <opencv2/tracking.hpp>

#include "randutils.hpp"

#include <opencv2/core_detect.hpp>

#include <typeinfo>

#include <numeric>
#include <cmath>


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
*/
#include <pcl/io/pcd_io.h>

#include <pcl/io/io.h>

#include <typeinfo>
#include <pcl/filters/passthrough.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/visualization/cloud_viewer.h>

#include "render.cpp"
#include "processPointClouds.cpp"
#include "objectDetection2D.cpp"
//#include "detector.cpp"
//#include "detector.h"
#include "dataStructures.h"

#include <pcl/common/pca.h>
#include <pcl/common/transforms.h>
#include <pcl/common/distances.h>

#include <filesystem>
#include <pcl/kdtree/kdtree_flann.h>

#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API
// 3rd party header for writing png files
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

//#include <C:/Program Files (x86)/jsoncpp/include/json/writer.h>

//#include <torch/script.h>
//#include <torch/torch.h>
#include <experimental/filesystem>

//namespace fs = std::filesystem;
namespace fs = std::experimental::filesystem;

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

//string bag_path = "C:/Users/hedey/Documents/20211008_130710.bag";//important: 2 good potholes

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
/*
std::chrono::system_clock::time_point
to_chrono_time_point(double d)
{
    using namespace std::chrono;
    using namespace date;
    using ddays = duration<double, days::period>;
    return sys_days{ December / 30 / 1899 } + round<system_clock::duration>(ddays{ d });
}
*/

float mat_mean(cv::Mat img)
{
    vector<int> elem;
    int index;
    float mean_v = 0;
    int c = 0;
    for (int i = 0; i <= img.rows; i++)
    {
        for (int j = 0; j < img.cols; j++)
        {
            int pix = img.at<uchar>(i, j);
            mean_v += pix;
            elem.push_back(pix);
            c+=1;
        }
    }
    std::sort(elem.begin(), elem.end());
    if (elem.size() % 2 == 0)
        index = elem.size() / 2 - 1;
    else
        index = (elem.size() - 1) / 2;
    std::cout << "c: " << c << ", size: " <<img.rows<<", "<<img.cols << ", "<< (img.rows) * (img.cols) << ", " <<elem.size()<< std::endl;
    std::cout << "mean: " << mean_v / elem.size() << ", median: " << elem[index]<<", at index: "<<index << std::endl;
    std::cout << "25%: " << elem[int(elem.size()*0.25)]<<", at: "<< int(elem.size() * 0.25) << ", 75%: " << elem[int(elem.size() * 0.75)] << ", at: " << int(elem.size() * 0.75) <<", 90%: " << elem[int(elem.size() * 0.9)] << ", at: " << int(elem.size() * 0.9) << std::endl;
    return mean_v/((img.rows)*(img.cols));
}

int mat_percentile(cv::Mat img, float perc)
{
    vector<int> elem;
    int index;
    for (int i = 0; i <= img.rows; i++)
    {
        for (int j = 0; j < img.cols; j++)
        {
            int pix = img.at<uchar>(i, j);
            elem.push_back(pix);
        }
    }
    std::sort(elem.begin(), elem.end());
    index = perc * elem.size();
    return elem[index];
}


std::string get_image_type(const cv::Mat& img, bool more_info = true)
{
    std::string r;
    int type = img.type();
    uchar depth = type & CV_MAT_DEPTH_MASK;
    uchar chans = 1 + (type >> CV_CN_SHIFT);

    switch (depth) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
    }

    r += "C";
    r += (chans + '0');

    if (more_info)
        std::cout << "depth: " << img.depth() << " channels: " << img.channels() << std::endl;

    return r;
}

void show_image(cv::Mat& img, std::string title)
{
    std::string image_type = get_image_type(img);
    cv::namedWindow(title + " type:" + image_type, cv::WINDOW_NORMAL); // Create a window for display.
    cv::imshow(title, img);
    cv::waitKey(0);
}
/*
auto transpose(torch::Tensor tensor, c10::IntArrayRef dims = { 0, 3, 1, 2 })
{
    //std::cout << "############### transpose ############" << std::endl;
    //std::cout << "shape before : " << tensor.sizes() << std::endl;
    tensor = tensor.permute(dims);
    //std::cout << "shape after : " << tensor.sizes() << std::endl;
    //std::cout << "######################################" << std::endl;
    return tensor;
}

auto ToTensor(cv::Mat img, bool show_output = false, bool unsqueeze = false, int unsqueeze_dim = 0)
{
    //std::cout << "image shape: " << img.size() << std::endl;
    torch::Tensor tensor_image = torch::from_blob(img.data, { img.rows, img.cols, 3 }, torch::kByte);

    if (unsqueeze)
    {
        tensor_image.unsqueeze_(unsqueeze_dim);
        //std::cout << "tensors new shape: " << tensor_image.sizes() << std::endl;
    }

    if (show_output)
    {
        std::cout << tensor_image.slice(2, 0, 1) << std::endl;
    }
    //std::cout << "tensor shape: " << tensor_image.sizes() << std::endl;
    return tensor_image;
}

auto ToInput(torch::Tensor tensor_image)
{
    // Create a vector of inputs.
    return std::vector<torch::jit::IValue>{tensor_image};
}

auto ToCvImage(torch::Tensor tensor)
{
    //int width = tensor.sizes()[0];
    //int height = tensor.sizes()[1];
    int width = 448;
    int height = 448;
    try
    {
        cv::Mat output_mat(cv::Size{ height, width }, CV_8UC3, tensor.data_ptr<uchar>());

        show_image(output_mat, "converted image from tensor");
        return output_mat.clone();
    }
    catch (const c10::Error& e)
    {
        std::cout << "an error has occured : " << e.msg() << std::endl;
    }
    return cv::Mat(height, width, CV_8UC3);
}
*/

void metadata_to_csv(const rs2::frame& frm, const std::string& filename)
{
    std::ofstream csv;

    csv.open(filename);

    //    std::cout << "Writing metadata to " << filename << endl;
    csv << "Stream," << rs2_stream_to_string(frm.get_profile().stream_type()) << "\nMetadata Attribute,Value\n";

    // Record all the available metadata attributes
    for (size_t i = 0; i < RS2_FRAME_METADATA_COUNT; i++)
    {
        if (frm.supports_frame_metadata((rs2_frame_metadata_value)i))
        {
            csv << rs2_frame_metadata_to_string((rs2_frame_metadata_value)i) << ","
                << frm.get_frame_metadata((rs2_frame_metadata_value)i) << "\n";
        }
    }

    csv.close();
}
/*
void metadata_to_jsn(int counter, Json::Value &event, const rs2::frame& frm)
{
    //Json::Value vec(Json::arrayValue);
    //vec.append(Json::Value(1));
    //vec.append(Json::Value(2));
    //vec.append(Json::Value(3));
    std::string streamType = rs2_stream_to_string(frm.get_profile().stream_type());

    // Record all the available metadata attributes
    for (size_t i = 0; i < RS2_FRAME_METADATA_COUNT; i++)
    {
        if (frm.supports_frame_metadata((rs2_frame_metadata_value)i))
        {
            event[to_string(counter)][streamType][rs2_frame_metadata_to_string((rs2_frame_metadata_value)i)] = frm.get_frame_metadata((rs2_frame_metadata_value)i);
        }
    }
}
*/


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
    int distance = 0.5;
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

vector<string> trackerTypes = { "BOOSTING", "MIL", "KCF", "TLD", "MEDIANFLOW", "GOTURN", "MOSSE", "CSRT" };

// create tracker by name
cv::Ptr<cv::Tracker> createTrackerByName(string trackerType)
{
    cv::Ptr<cv::Tracker> tracker;
    if (trackerType == trackerTypes[0])
        tracker = cv::TrackerBoosting::create();
    else if (trackerType == trackerTypes[1])
        tracker = cv::TrackerMIL::create();
    else if (trackerType == trackerTypes[2])
        tracker = cv::TrackerKCF::create();
    else if (trackerType == trackerTypes[3])
        tracker = cv::TrackerTLD::create();
    else if (trackerType == trackerTypes[4])
        tracker = cv::TrackerMedianFlow::create();
    else if (trackerType == trackerTypes[5])
        tracker = cv::TrackerGOTURN::create();
    else if (trackerType == trackerTypes[6])
        tracker = cv::TrackerMOSSE::create();
    else if (trackerType == trackerTypes[7])
        tracker = cv::TrackerCSRT::create();
    else {
        cout << "Incorrect tracker name" << endl;
        cout << "Available trackers are: " << endl;
        for (vector<string>::iterator it = trackerTypes.begin(); it != trackerTypes.end(); ++it)
            std::cout << " " << *it << endl;
    }
    return tracker;
}

void getRandomColors(vector<cv::Scalar>& colors, int numColors)
{
    randutils::mt19937_rng rng;
    for (int i = 0; i < numColors; i++)
        colors.push_back(cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)));
}

cv::Scalar getRandomColor()
{
    randutils::mt19937_rng rng;
    return cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
}

double calculateIoU()
{

}

cv::Mat format_yolov5(const cv::Mat& source) {
    // put the image in a square big enough
    int col = source.cols;
    int row = source.rows;
    int _max = MAX(col, row);
    cv::Mat resized = cv::Mat::zeros(_max, _max, CV_8UC3);
    source.copyTo(resized(cv::Rect(0, 0, col, row)));

    // resize to 640x640, normalize to [0,1[ and swap Red and Blue channels
    cv::Mat result;
    cv::dnn::blobFromImage(source, result, 1. / 255., cv::Size(640, 640), cv::Scalar(), true, false);

    return result;
}

std::vector<std::string> LoadNames(const std::string& path) {
    // load class names
    std::vector<std::string> class_names;
    std::ifstream infile(path);
    if (infile.is_open()) {
        std::string line;
        while (getline(infile, line)) {
            class_names.emplace_back(line);
        }
        infile.close();
    }
    else {
        std::cerr << "Error loading the class names!\n";
    }

    return class_names;
}

void pothole_severity(int rounded_min, float avg_diameter_mm, float concave_hull_area)
{
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
}

int main(int argc, char* argv[]) try
{
    typedef pcl::PointXYZRGB Cloud_Type;
    pcl::PointCloud<Cloud_Type>::Ptr newCloud(new pcl::PointCloud<Cloud_Type>);
    ProcessPointClouds<Cloud_Type>* pointProcessorRGB = new ProcessPointClouds<Cloud_Type>();
    //newCloud = pointProcessorRGB->loadPly("C:/Users/hedey/OneDrive/Documents/COLMAP_projects/Challenge_video_1/dense/8/fused.ply");
    newCloud = pointProcessorRGB->loadPly("fused.ply");

    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
    CameraAngle setAngle = FPS; //XY, FPS, Side, TopDown
    initCamera(setAngle, viewer);


    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    std::pair<typename pcl::PointCloud<Cloud_Type>::Ptr, typename pcl::PointCloud<Cloud_Type>::Ptr> segmentCloud;
    segmentCloud = pointProcessorRGB->SegmentPlane(newCloud, 1000, .01, coefficients);

    std::cerr << "Plane model coefficients: " << coefficients->values[0] << " "
        << coefficients->values[1] << " "
        << coefficients->values[2] << " "
        << coefficients->values[3] << std::endl;

    renderPointCloud(viewer, newCloud, "fusedCloud");
    renderPointCloud(viewer, segmentCloud.second, "planeCloud", Color(0, 1, 0));
    std::vector<pcl::PointCloud<Cloud_Type>::Ptr> clustersCloud;

    clustersCloud = pointProcessorRGB->Clustering(segmentCloud.first, .01, 500, 500000);
    int clusterId = 0;

    int i = 0;
    for (pcl::PointCloud<Cloud_Type>::Ptr cluster : clustersCloud)
    {
        bool pothole=false;
        std::cout << "--------------------------------------" << std::endl;
        std::cout << "cluster size ";
        pointProcessorRGB->numPoints(cluster);

        cCluster cc;
        cc.clusterID = clusterId;

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
            pothole = true;
            std::cout << "Pothole's depth ranges from: " << min_depth << " to: " << max_depth << ", Max Depth is: " << abs(floor(min_depth * 1000.0 + .5)) << " mm." << std::endl;

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
        cc.minDepth = rounded_min;
        cc.maxDepth = rounded_max;
        int max_level = 950;//150
        int min_level = -110;//-110
        int j = 0;

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
                for (Cloud_Type point : it->second)
                {
                    LidarPoint p;
                    p.x = point.x, p.y = point.y, p.z = point.z;
                    p.r = get<0>(RGB_Color) * 255.0, p.g = get<1>(RGB_Color) * 255.0, p.b = get<2>(RGB_Color) * 255.0;
                    cc.lidarPoints.push_back(p);
                    if ((it->first > 0 && it->first == rounded_min) || (it->first < 0 && it->first == rounded_max))
                        min_depth_Cloud->points.push_back(point);
                    else if ((it->first > 0 && it->first == rounded_max) || (it->first < 0 && it->first == rounded_min))
                        max_depth_Cloud->points.push_back(point);
                    else
                        depth_contour_Cloud->points.push_back(point);
                }

                if (it->first > 0)
                    pos_capacity += (it->first) * depth_estimation[it->first].size();
                else if (it->first < 0)
                    neg_capacity += (it->first) * depth_estimation[it->first].size();
            }
            j++;
        }
        pcl::PointCloud<Cloud_Type>::Ptr depth_colored_Cloud(new pcl::PointCloud<Cloud_Type>);
        for (auto p : cc.lidarPoints)
        {
            Cloud_Type point;
            point.x = p.x, point.y = p.y, point.z = p.z;
            point.r = p.r, point.g = p.g, point.b = p.b;
            depth_colored_Cloud->points.push_back(point);
        }
        renderPointCloud(viewer, depth_colored_Cloud, "coloredDepth" + zero_padding(std::to_string(clusterId), 4) + zero_padding(std::to_string(j), 4));
        //cout << "Press [Q] in viewer to continue. " << endl;
        viewer->spinOnce();

        if (pothole==true)
        {
            pcl::PointCloud<Cloud_Type>::Ptr cloud_projected(new pcl::PointCloud<Cloud_Type>);
            cloud_projected = pointProcessorRGB->ProjectCloud(cluster, coefficients);
            //std::cerr << "Projected cluster has: " << cloud_projected->size() << " points." << std::endl;

            // Create a Concave Hull representation of the projected inliers
            pcl::PointCloud<Cloud_Type>::Ptr cloud_hull(new pcl::PointCloud<Cloud_Type>);
            pointProcessorRGB->ConcaveHullCloud(cloud_projected, cloud_hull);

            //std::cerr << "Concave hull has: " << cloud_hull->size()
            //    << " points." << std::endl;
            renderPointCloud(viewer, cloud_hull, "cloud_hull" + zero_padding(std::to_string(clusterId), 4), Color(0, 0, 1));
            float concave_hull_area = calculatePolygonArea(*cloud_hull);
            std::cout << "Area within concave hull is: " << concave_hull_area << std::endl;
            float avg_area_per_point = concave_hull_area / cloud_projected->size();
            //float volume = avg_area_per_point * capacity/1000;
            //std::cout << "Volume is: " << volume << " (" << 1000 * volume << " litres.)" << std::endl;
            float pos_volume = avg_area_per_point * pos_capacity / 1000;
            float neg_volume = avg_area_per_point * neg_capacity / 1000;
            //std::cout << "Positive volume is: " << pos_volume << " (" << 1000 * pos_volume << " litres)." << std::endl;
            //std::cout << "Negative volume is: " << neg_volume << " (" << 1000 * neg_volume << " litres)." << std::endl;
            std::cout << "**************************************" << std::endl;
            std::cout << "Pothole attributes:" << std::endl;
            std::cout << "-------------------" << std::endl;
            std::cout << "Maximum Pothole's Depth is: " << abs(rounded_min) << " mm." << std::endl;
            float avg_diameter_mm = 2*1000*sqrt(concave_hull_area/ M_PI); //abs(floor(0.5 * (distance_z + distance_y) * 1000.0 + .5))
            std::cout << "Average Pothole's Diameter is: " << avg_diameter_mm << " mm." << std::endl;
            std::cout << "Pothole's Area is: " << concave_hull_area << " m2." << std::endl;
            pothole_severity(rounded_min, avg_diameter_mm, concave_hull_area);
            std::cout << "Pothole's volume (under plane only) is: " << abs(neg_volume) << " m3 (" << 1000 * abs(neg_volume) << " litres)." << std::endl;
            std::cout << "**************************************" << std::endl;
        }

        clusterId++;
    }
    while (!viewer->wasStopped())
    {
        viewer->spinOnce();
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