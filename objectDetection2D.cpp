
#include <fstream>
#include <sstream>
#include <iostream>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "objectDetection2D.hpp"

#include <cmath>


using namespace std;

std::tuple<int, int, int> to_color(int indx, int base)
{
    /* return (b, r, g) tuple */
    int b, r, g, base2;
    base2 = base * base;
    b = 2 - indx / base2;
    r = 2 - (indx % base2) / base;
    g = 2 - (indx % base2) % base;
    return std::tuple<int, int, int>(b * 127, r * 127, g * 127);
}

void draw_yolo_bb(cv::Mat& img, int c, float x, float y, float w, float h, vector<string> classes, bool bVis, cv::Mat& visImg)
{
    //cv::Mat visImg = img.clone();

    int base = int(ceil(pow(classes.size(), 1.0 / 3.0)));
    std::tuple<int, int, int> RGB_Color;
    RGB_Color = to_color(c, base);

    //int left = (b.x - b.w / 2.) * im.w;
    //int right = (b.x + b.w / 2.) * im.w;
    //int top = (b.y - b.h / 2.) * im.h;
    //int bot = (b.y + b.h / 2.) * im.h;

    int left = (x - w / 2.) * visImg.size().width;
    int right = (x + w / 2.) * visImg.size().width;
    int top = (y - h / 2.) * visImg.size().height;
    int bot = (y + h / 2.) * visImg.size().height;

    if (left < 0) left = 0;
    if (right > visImg.size().width - 1) right = visImg.size().width - 1;
    if (top < 0) top = 0;
    if (bot > visImg.size().height - 1) bot = visImg.size().height - 1;
    std::cout << visImg.size().width << ", " << visImg.size().height << std::endl;
    std::cout << left << ", " << top << ", " << right << ", " << bot << std::endl;
    cv::rectangle(visImg, cv::Point(left, top), cv::Point(right, bot), cv::Scalar(get<1>(RGB_Color), get<2>(RGB_Color), get<0>(RGB_Color)), 2);
    //cv::rectangle(visImg, cv::Point(left, top), cv::Point(min(left + width, visImg.size().width), top + height), cv::Scalar(get<1>(RGB_Color), get<2>(RGB_Color), get<0>(RGB_Color)), 2);

    string label = classes[c];

    int baseLine;
    cv::Size labelSize = getTextSize(label, cv::FONT_ITALIC, 0.75, 2, &baseLine);
    top = max(top, labelSize.height);
    int label_left = max(int(left), 0);
    int label_top = max(int(top - round(2 * labelSize.height)), 0);
    int label_width = int(min(round(labelSize.width), double(visImg.size().width - left)));
    cv::Mat roi = visImg(cv::Rect(label_left, label_top, label_width, round(2.0 * labelSize.height)));

    //cv::Mat color(roi.size(), CV_8UC3, cv::Scalar(0, 255, 255));
    cv::Mat color(roi.size(), CV_8UC3, cv::Scalar(get<1>(RGB_Color), get<2>(RGB_Color), get<0>(RGB_Color)));
    //cv::Mat color(roi.size(), CV_8UC3, cv::Scalar(get<0>(RGB_Color), get<1>(RGB_Color), get<2>(RGB_Color)));
    //cv::Mat color(roi.size(), CV_8UC3, cv::Scalar(get<0>(RGB_Color), get<2>(RGB_Color), get<1>(RGB_Color)));
    double alpha = 0.6;
    cv::addWeighted(color, alpha, roi, 1.0 - alpha, 0.0, roi);


    //cv::putText(visImg, label, cv::Point(left, top), cv::FONT_HERSHEY_DUPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    //cv::putText(visImg, label, cv::Point(left, top), cv::FONT_HERSHEY_DUPLEX, 0.75, cv::Scalar(0, 0, 0), 0.75);
    cv::putText(visImg, label, cv::Point(left, top - round(0.35 * labelSize.height)), cv::FONT_ITALIC, 0.75, cv::Scalar(0, 0, 0), 2);


    string windowName = "Object annotation";
    //cv::namedWindow( windowName, 1 );
    cv::namedWindow(windowName, cv::WINDOW_NORMAL);
    cv::imshow(windowName, visImg);
    cv::waitKey(0); // wait for key to be pressed

}



// detects objects in an image using the YOLO library and a set of pre-trained objects from the COCO database;
// a set of 80 classes is listed in "coco.names" and pre-trained weights are stored in "yolov3.weights"
//void detectObjects(int f, cv::Mat& img, std::vector<BoundingBox>& bBoxes, float confThreshold, float nmsThreshold,
//    std::string basePath, std::string classesFile, std::string modelConfiguration, std::string modelWeights, bool bVis)
void detectObjects(int f, cv::Mat& img, std::string save_path, std::vector<BoundingBox>& bBoxes, float confThreshold, float nmsThreshold,
    std::string basePath, vector<string> classes, std::string modelConfiguration, std::string modelWeights, bool bVis, int s)
{
    // load class names from file
    /*
    vector<string> classes;
    ifstream ifs(classesFile.c_str());
    string line;

    while (getline(ifs, line)) classes.push_back(line);
    */
    int base = int(ceil(pow(classes.size(), 1.0 / 3.0)));

    // load neural network
    cv::dnn::Net net = cv::dnn::readNetFromDarknet(modelConfiguration, modelWeights);
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

    // generate 4D blob from input image
    cv::Mat blob;
    vector<cv::Mat> netOutput;
    double scalefactor = 1 / 255.0;
    //cv::Size size = cv::Size(416, 416);
    cv::Size size = cv::Size(s, s);
    cv::Scalar mean = cv::Scalar(0, 0, 0);
    bool swapRB = false;
    bool crop = false;
    cv::dnn::blobFromImage(img, blob, scalefactor, size, mean, swapRB, crop);

    // Get names of output layers
    vector<cv::String> names;
    vector<int> outLayers = net.getUnconnectedOutLayers(); // get  indices of  output layers, i.e.  layers with unconnected outputs
    vector<cv::String> layersNames = net.getLayerNames(); // get  names of all layers in the network

    names.resize(outLayers.size());
    for (size_t i = 0; i < outLayers.size(); ++i) // Get the names of the output layers in names
        names[i] = layersNames[outLayers[i] - 1];

    // invoke forward propagation through network
    net.setInput(blob);
    net.forward(netOutput, names);

    // Scan through all bounding boxes and keep only the ones with high confidence
    vector<int> classIds; vector<float> confidences; vector<cv::Rect> boxes;
    for (size_t i = 0; i < netOutput.size(); ++i)
    {
        float* data = (float*)netOutput[i].data;
        for (int j = 0; j < netOutput[i].rows; ++j, data += netOutput[i].cols)
        {
            cv::Mat scores = netOutput[i].row(j).colRange(5, netOutput[i].cols);
            cv::Point classId;
            double confidence;

            // Get the value and location of the maximum score
            cv::minMaxLoc(scores, 0, &confidence, 0, &classId);
            if (confidence > confThreshold)
            {
                cv::Rect box; int cx, cy;
                cx = (int)(data[0] * img.cols);
                cy = (int)(data[1] * img.rows);
                box.width = (int)(data[2] * img.cols);
                box.height = (int)(data[3] * img.rows);
                box.x = cx - box.width / 2; // left
                box.y = cy - box.height / 2; // top

                boxes.push_back(box);
                classIds.push_back(classId.x);
                confidences.push_back((float)confidence);
            }
        }
    }

    // perform non-maxima suppression
    vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
    for (auto it = indices.begin(); it != indices.end(); ++it) {

        BoundingBox bBox;
        bBox.roi = boxes[*it];
        bBox.classID = classIds[*it];
        bBox.confidence = confidences[*it];
        bBox.boxID = (int)bBoxes.size(); // zero-based unique identifier for this bounding box

        bBoxes.push_back(bBox);
    }
    //std::string output_path = "C:/Users/hedey/source/repos/Road_Defects_Detector/Test_output/";
    std::string output_path = save_path;


    cv::Mat visImg = img.clone();
    for (auto it = bBoxes.begin(); it != bBoxes.end(); ++it) {

        // Draw rectangle displaying the bounding box
        int top, left, width, height;
        top = (*it).roi.y;
        left = (*it).roi.x;
        width = (*it).roi.width;
        height = (*it).roi.height;
        std::tuple<int, int, int> RGB_Color;
        RGB_Color = to_color((*it).classID, base);
        //cv::rectangle(visImg, cv::Point(left, top), cv::Point(left+width, top+height),cv::Scalar(0, 255, 255), 1);
        //std::cout << left << ", " << top << ", " << left + width << ", " << top + height << endl;
        cv::rectangle(visImg, cv::Point(left, top), cv::Point(min(left + width, visImg.size().width), top + height), cv::Scalar(get<1>(RGB_Color), get<2>(RGB_Color), get<0>(RGB_Color)), 2);
        //cv::rectangle(visImg, cv::Point(left, top), cv::Point(left + width, top + height), cv::Scalar(get<0>(RGB_Color), get<1>(RGB_Color), get<2>(RGB_Color)), 2);
        //cv::rectangle(visImg, cv::Point(left, top), cv::Point(left + width, top + height), cv::Scalar(get<0>(RGB_Color), get<2>(RGB_Color), get<1>(RGB_Color)), 2);

        string label = cv::format("%.2f", (*it).confidence);
        label = classes[((*it).classID)] + ": " + label;
        //label = classes[((*it).classID)];

        // Display label at the top of the bounding box
        int baseLine;
        //cv::Size labelSize = getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.75, 1, &baseLine);
        //cv::Size labelSize = getTextSize(label, cv::FONT_HERSHEY_DUPLEX, 0.5, 1, &baseLine);
        //cv::Size labelSize = getTextSize(label, cv::FONT_HERSHEY_DUPLEX, 0.75, .75, &baseLine);
        cv::Size labelSize = getTextSize(label, cv::FONT_ITALIC, 0.75, 2, &baseLine);
        top = max(top, labelSize.height);
        //rectangle(visImg, cv::Point(left, top - round(1.5*labelSize.height)), cv::Point(left + round(1.5*labelSize.width), top + baseLine), cv::Scalar(255, 255, 255), cv::FILLED);
        //rectangle(visImg, cv::Point(left, top - round(1.5 * labelSize.height)), cv::Point(left + round(1.5 * labelSize.width), top + baseLine), cv::Scalar(0, 255, 255), cv::FILLED);
            //cv::putText(visImg, label, cv::Point(left, top), cv::FONT_ITALIC, 0.75, cv::Scalar(0,0,0),1);


        //cv::Mat image = cv::imread("IMG_2083s.png");
        //cv::Mat roi = visImg(cv::Rect(left, top - round(1.5 * labelSize.height), left + round(1.5 * labelSize.width), top + baseLine));
        //cv::Mat roi = visImg(cv::Rect(left, top - round(1.5 * labelSize.height), round(1.5 * labelSize.width), baseLine));

        //cv::Mat roi = visImg(cv::Rect(left, top - round(1.5 * labelSize.height), round(1.5 * labelSize.width), round(1.5 * labelSize.height)));
        //cv::Mat roi = visImg(cv::Rect(left, int(min(top - round(2.0 * labelSize.height),0.0)), int(min(round(1.5 * labelSize.width), double(visImg.size().width-left))), int(max(round(2.0 * labelSize.height), double(visImg.size().height - top)))));
        //cv::Mat roi = visImg(cv::Rect(left, top-round(2.0 * labelSize.height), int(min(round(1.5 * labelSize.width), double(visImg.size().width - left))), round(2.0 * labelSize.height)));
        //std::cout << left<<", "<< top - round(2.0 * labelSize.height) << ", " << int(min(round(labelSize.width), double(visImg.size().width - left))) << ", " << round(2.0 * labelSize.height) << endl;
        int label_left = max(int(left), 0);
        int label_top = max(int(top - round(2 * labelSize.height)), 0);
        int label_width = int(min(round(labelSize.width), double(visImg.size().width - left)));
        //cv::Mat roi = visImg(cv::Rect(left, top - round(2.0 * labelSize.height), int(min(round(labelSize.width), double(visImg.size().width - left))), round(2.0 * labelSize.height)));
        cv::Mat roi = visImg(cv::Rect(label_left, label_top, label_width, round(2.0 * labelSize.height)));

        //cv::Mat color(roi.size(), CV_8UC3, cv::Scalar(0, 255, 255));
        cv::Mat color(roi.size(), CV_8UC3, cv::Scalar(get<1>(RGB_Color), get<2>(RGB_Color), get<0>(RGB_Color)));
        //cv::Mat color(roi.size(), CV_8UC3, cv::Scalar(get<0>(RGB_Color), get<1>(RGB_Color), get<2>(RGB_Color)));
        //cv::Mat color(roi.size(), CV_8UC3, cv::Scalar(get<0>(RGB_Color), get<2>(RGB_Color), get<1>(RGB_Color)));
        double alpha = 0.6;
        cv::addWeighted(color, alpha, roi, 1.0 - alpha, 0.0, roi);


        //cv::putText(visImg, label, cv::Point(left, top), cv::FONT_HERSHEY_DUPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
        //cv::putText(visImg, label, cv::Point(left, top), cv::FONT_HERSHEY_DUPLEX, 0.75, cv::Scalar(0, 0, 0), 0.75);
        cv::putText(visImg, label, cv::Point(left, top - round(0.35 * labelSize.height)), cv::FONT_ITALIC, 0.75, cv::Scalar(0, 0, 0), 2);
    }

    //imwrite(output_path + zero_padding(std::to_string(f), 5) + ".png", visImg);
    if (bBoxes.size() > 0) imwrite(output_path + std::to_string(f) + ".jpg", visImg);

    // show results
    if (bVis)
    {

        string windowName = "Object classification";
        //cv::namedWindow( windowName, 1 );
        cv::namedWindow(windowName, cv::WINDOW_NORMAL);
        cv::imshow(windowName, visImg);
        //cv::waitKey(0); // wait for key to be pressed

        //cv::destroyWindow(windowName);
    }
}

