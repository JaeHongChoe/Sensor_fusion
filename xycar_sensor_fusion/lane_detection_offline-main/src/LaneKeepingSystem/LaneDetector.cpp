// Copyright (C) 2023 Grepp CO.
// All rights reserved.

/**
 * @file LaneDetector.cpp
 * @author Jeongmin Kim
 * @author Jeongbin Yim
 * @brief lane detector class source file
 * @version 2.1
 * @date 2023-10-13
 * 
 * 
 * our msg info
 * bounding_boxes[]
  bounding_boxes[0]: 
    probability: 0.84668
    xmin: 220
    ymin: 108
    xmax: 616
    ymax: 456
    id: 3
 */

#include <numeric>
#include "LaneKeepingSystem/LaneDetector.hpp"

namespace Xycar {

template <typename PREC>
void LaneDetector<PREC>::setConfiguration(const YAML::Node& config)
{
    mImageWidth = config["IMAGE"]["WIDTH"].as<int32_t>();
    mImageHeight = config["IMAGE"]["HEIGHT"].as<int32_t>();
    low_threshold = config["CANNY"]["LOW_THRESHOLD"].as<int32_t>();
    high_threshold = config["CANNY"]["HIGH_THRESHOLD"].as<int32_t>();    
    min_pixel = config["HOUGH"]["MIN_PIXEL"].as<int32_t>();
    min_line = config["HOUGH"]["MIN_LINE"].as<int32_t>();    
    max_gap = config["HOUGH"]["MAX_GAP"].as<int32_t>();
    yOffset = config["IMAGE"]["Y_OFFSET"].as<int32_t>();
    yGap = config["IMAGE"]["Y_GAP"].as<int32_t>(); 
    yGain = config["IMAGE"]["Y_GAIN"].as<double>();    
    mDebugging = config["DEBUG"].as<bool>();
}

template <typename PREC>
std::pair<cv::Point, cv::Point> LaneDetector<PREC>::getLinePos(std::vector<int>& left_x_at_Y_offset, std::vector<int>& right_x_at_Y_offset)
{
    cv::Point lpos, rpos;

    if (!left_x_at_Y_offset.empty() && left_x_at_Y_offset.size() <6) {
        int lposl_x = *min_element(left_x_at_Y_offset.begin(), left_x_at_Y_offset.end());
        int lposr_x = *max_element(left_x_at_Y_offset.begin(), left_x_at_Y_offset.end());
        if (lposr_x-lposl_x > 50 && left_x_at_Y_offset.size() !=1) lposl_x = lposr_x-50;
        lpos = cv::Point((lposl_x+lposr_x)/2,tempYOffset);
        
        prev_lpos = lpos;
        leftC = 0;

    }else{
        lpos = prev_lpos;
        leftC +=1;
    }
    if (!right_x_at_Y_offset.empty() && right_x_at_Y_offset.size() <6) {
        int rposl_x = *min_element(right_x_at_Y_offset.begin(), right_x_at_Y_offset.end());
        int rposr_x = *max_element(right_x_at_Y_offset.begin(), right_x_at_Y_offset.end());
        if (rposr_x-rposl_x > 50 && right_x_at_Y_offset.size() !=1) rposr_x = rposl_x+50;
        rpos = cv::Point((rposl_x+rposr_x)/2, tempYOffset);
        prev_rpos = rpos;
        rightC =0;
    }else{
        rightC +=1;
        rpos = prev_rpos;
    }


    //when points are too close
    if (abs(rpos.x - lpos.x) <300){//300 100
        if(rightC != 0){
            rpos.x = 640;
            // rpos.x = lpos.x +300;
            prev_rpos = rpos;
        }
        if(leftC != 0) {
            lpos.x = 0;
            // lpos.x = rpos.x -300;
            prev_lpos = lpos;
        }
    }

    if(lpos.x >240){
        rpos.x += 50;
        prev_rpos.x += 1;
        rightC = 0;
    }
    if(rpos.x < 400){
        lpos.x -=50;
        prev_lpos.x -=1;
        leftC =0;
    }

    return std::make_pair(lpos, rpos);
}

//estimate angle
template <typename PREC>
std::pair<float, float> LaneDetector<PREC>::getLiDARAngle(cv::Point& lpos, cv::Point& rpos)
{
    //lpos angle
    cv::Mat input_lpos = (cv::Mat_<double>(3,1) << lpos.x, tempYOffset, 1);
    cv::Mat homo_lpos = mHomoMat * input_lpos;
    cv::Point2f out_lpos(
        static_cast<float>(homo_lpos.at<double>(0) / homo_lpos.at<double>(2)),
        static_cast<float>(homo_lpos.at<double>(1) / homo_lpos.at<double>(2))
    );

    float vector_lx = 270.0 - out_lpos.x;
    float vector_ly = 540.0 - out_lpos.y;

    float angle_rad_l = std::atan2(vector_ly, vector_lx);
    float angle_deg_l = angle_rad_l * (180.0 / M_PI);

    //rpos angle
    cv::Mat input_rpos = (cv::Mat_<double>(3,1) << rpos.x, tempYOffset, 1);
    cv::Mat homo_rpos = mHomoMat * input_rpos;
    cv::Point2f out_rpos(
        static_cast<float>(homo_rpos.at<double>(0) / homo_rpos.at<double>(2)),
        static_cast<float>(homo_rpos.at<double>(1) / homo_rpos.at<double>(2))
    );

    float vector_rx = 270.0 - out_rpos.x;
    float vector_ry = 540.0 - out_rpos.y;

    float angle_rad_r = std::atan2(vector_ry, vector_rx);
    float angle_deg_r = angle_rad_r * (180.0 / M_PI);

    return std::make_pair(angle_deg_l, angle_deg_r);
}

template <typename PREC>
std::pair<std::vector<int>, std::vector<int>> LaneDetector<PREC>::divideLeftRight(std::vector<cv::Vec4f>& lines)
{
    std::vector<int> left_x_at_Y_offset;
    std::vector<int> right_x_at_Y_offset;   
    double slope;    
    
  
    for (cv::Vec4f line_ : lines) {
        cv::Point pt1(line_[0], line_[1]);
        cv::Point pt2(line_[2], line_[3]);
        slope = (double)(pt2.y - pt1.y) / (pt2.x - pt1.x + 0.0001);
        double leng = sqrt(pow(line_[2] - line_[0],2) + pow(line_[3] - line_[1],2));
        if (abs(slope)>0 && abs(slope) < 10) { 
            int x_at_Y_offset;
            if (pt1.x != pt2.x) {
                x_at_Y_offset = (tempYOffset - pt1.y) / slope + pt1.x;
            } else {
                x_at_Y_offset = pt1.x; 
            }

            if (slope < 0 && x_at_Y_offset <280) {//250
                left_x_at_Y_offset.push_back(x_at_Y_offset);
            } else if (slope > 0 && x_at_Y_offset >360) {//380
                right_x_at_Y_offset.push_back(x_at_Y_offset);
            } 
        }
        if(abs(slope) < 0.02 && leng > 150){
                std::cout << "center line ----------" <<std::endl;
                stopline_flag = true;
        }    
    } 
    return std::make_pair(left_x_at_Y_offset, right_x_at_Y_offset);
}

template <typename PREC>
int LaneDetector<PREC>::processImage(cv::Mat& frame,yolov7::BoundingBoxes yolo)
{
    cv::Mat img_gray, img_histo, img_blur, img_edge, roi, thresframe;
    std::vector<cv::Vec4f> lines;
    cv::Mat output;
    cv::Mat mask = cv::Mat::zeros(frame.size(), CV_8UC1);
    
    std::vector<cv::Point> square;
    square.push_back(cv::Point(0, tempYOffset+yGap));
    square.push_back(cv::Point(0, tempYOffset-yGap)); 
    square.push_back(cv::Point(frame.cols, tempYOffset-yGap)); 
    square.push_back(cv::Point(frame.cols,tempYOffset+yGap)); 

    

    cv::fillConvexPoly(mask, &square[0], 4, cv::Scalar(255));

    cv::cvtColor(frame, img_gray, cv::COLOR_BGR2GRAY);	    
    cv::GaussianBlur(img_gray, img_blur, cv::Size(), 2.0);
    cv::Canny(img_blur, img_edge, low_threshold, high_threshold);
    // for(const auto& bbox : yolo.bounding_boxes) {
    //     cv::rectangle(img_edge, cv::Point(bbox.xmin, bbox.ymin), cv::Point(bbox.xmax, bbox.ymax), cv::Scalar(0, 255, 0), -1);
    // }
    cv::bitwise_and(img_edge, mask, output, mask= mask);    
    cv::threshold(output,thresframe,190,255,cv::THRESH_BINARY);
    cv::HoughLinesP(thresframe, lines, 1, CV_PI / 180, min_pixel, min_line, max_gap);
    
    
    std::vector<int> leftLines, rightLines;
    std::tie(leftLines, rightLines) = divideLeftRight(lines);
    cv::Point lpos, rpos;
    std::tie(lpos,rpos) = getLinePos(leftLines,rightLines);
   
    if (mDebugging)
    {
        // draw parts
        mask.copyTo(mDebugROI);
        frame.copyTo(mDebugFrame);
        drawLines(lines);
        drawRectangle(lpos.x, rpos.x);
    }
    return (double)((lpos.x + rpos.x)/2)+30;
}

// template <typename PREC>
// void LaneDetector<PREC>::depth(bbox, frame)
// {
//     cv::Mat homo_mat = (cv::Mat_<float>(3, 3) <<
//         -2.10492618e-01, -1.25468176e+00, 3.33525242e+02,
//         -2.32671202e-02, -2.95749888e+00, 7.41289630e+02,
//         -4.43564281e-05, -4.64696947e-03, 1.00000000e+00
//     );

    
// }

template <typename PREC>
void LaneDetector<PREC>::drawLines(std::vector<cv::Vec4f>& lines)
{
    for (auto& line : lines) {
        int x1, y1, x2, y2;
        std::tie(x1, y1, x2, y2) = std::make_tuple(line[0], line[1], line[2], line[3]);        
        cv::line(mDebugFrame, cv::Point(x1, y1), cv::Point(x2, y2), kRed, 2);
    }
}

template <typename PREC>
void LaneDetector<PREC>::drawRectangle(int leftPositionX, int rightPositionX)
{
    int center = (leftPositionX + rightPositionX) / 2;    
    cv::rectangle(mDebugFrame, cv::Point(leftPositionX - 5, tempYOffset-5), cv::Point(leftPositionX + 5, tempYOffset+5), kGreen, 2, cv::LINE_AA);
    putText(mDebugFrame, cv::format("(%d, %d,leftPositionX)", leftPositionX, tempYOffset), cv::Point(leftPositionX, tempYOffset-20), cv::FONT_HERSHEY_SIMPLEX, 0.5, kGreen, 1, cv::LINE_AA);
    cv::rectangle(mDebugFrame, cv::Point(rightPositionX - 5, tempYOffset-5), cv::Point(rightPositionX + 5,tempYOffset+5), kGreen, 2, cv::LINE_AA);
    putText(mDebugFrame, cv::format("(%d, %d,rightPositionX)", rightPositionX, tempYOffset), cv::Point(rightPositionX, tempYOffset-20), cv::FONT_HERSHEY_SIMPLEX, 0.5, kGreen, 1, cv::LINE_AA);
    cv::rectangle(mDebugFrame, cv::Point(center - 5, tempYOffset-5), cv::Point(center + 5, tempYOffset+5), kRed, 2, cv::LINE_AA);
    putText(mDebugFrame, cv::format("(%d, %d,lane_center)",center, tempYOffset), cv::Point(center, tempYOffset-20), cv::FONT_HERSHEY_SIMPLEX, 0.5, kRed, 1, cv::LINE_AA);
    cv::rectangle(mDebugFrame, cv::Point(mImageWidth/2 - 5, tempYOffset-5), cv::Point(mImageWidth/2 + 5, tempYOffset+5), kBlue, 2, cv::LINE_AA);
}
template class LaneDetector<float>;
template class LaneDetector<double>;
} // namespace Xycar
