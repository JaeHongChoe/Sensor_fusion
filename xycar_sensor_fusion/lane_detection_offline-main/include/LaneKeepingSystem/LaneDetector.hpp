#ifndef LANE_DETECTOR_HPP_
#define LANE_DETECTOR_HPP_

#include <cmath>
#include "opencv2/opencv.hpp"
#include <yaml-cpp/yaml.h>
#include <ctime>

#include <yolov7/BoundingBoxes.h>
/// create your lane detecter
/// Class naming.. it's up to you.
namespace Xycar {

enum class Direction : uint8_t
{
    LEFT = 0,  ///< Line direction LEFT
    RIGHT = 1, ///< Line direction RIGHT
};

template <typename PREC>
class LaneDetector final
{
public:
    using Ptr = LaneDetector*; /// < Pointer type of the class(it's up to u)

    static inline const cv::Scalar kRed = {0, 0, 255}; /// Scalar values of Red
    static inline const cv::Scalar kGreen = {0, 255, 0}; /// Scalar values of Green
    static inline const cv::Scalar kBlue = {255, 0, 0}; /// Scalar values of Blue
    static inline const cv::Scalar kBlack = {0, 0, 0}; /// Scalar values of Blue
    static inline const cv::Scalar kWhite = {255, 255, 255}; /// Scalar values of Blue

    LaneDetector(const YAML::Node& config) {
        setConfiguration(config);
        prev_lpos=cv::Point(0,yOffset);
        prev_rpos=cv::Point(mImageWidth, yOffset);
        leftC=0;
        rightC=0;
    }
    std::pair<float, float> getLiDARAngle(cv::Point& lpos, cv::Point& rpos);

    int processImage(cv::Mat& frame,yolov7::BoundingBoxes yolo);
    void drawRectangle(int32_t leftPositionX, int32_t rightPositionX);

    const cv::Mat& getDebugFrame() const { return mDebugFrame;};
    const cv::Mat& getDebugROI() const {return mDebugROI;};
    int32_t getWidth(){return mImageWidth;};
    void setYOffset(double speed){tempYOffset=yOffset-speed*yGain;}

    bool stopline_flag = false;

private:
    int32_t mImageWidth;
    int32_t mImageHeight;
    int32_t low_threshold;
    int32_t high_threshold;
    int32_t min_pixel;
    int32_t min_line;
    int32_t max_gap;
    int32_t yOffset; 
    int32_t yGap;    
    int32_t tempYOffset;  
    double yGain;
    bool mDebugging;
    cv::Mat mDebugFrame;
    cv::Mat mDebugROI;
    cv::Point prev_lpos;
    cv::Point prev_rpos;
    int leftC;
    int rightC;

    std::pair<std::vector<int>, std::vector<int>> divideLeftRight(std::vector<cv::Vec4f>& lines);
    std::pair<cv::Point, cv::Point> getLinePos(std::vector<int>& left_x_at_Y_offset, std::vector<int>& right_x_at_Y_offset);    
    void drawLines(std::vector<cv::Vec4f>& lines);    
    void setConfiguration(const YAML::Node& config);

    cv::Mat mHomoMat = (cv::Mat_<double>(3,3) <<
        -1.36650485e-01, -1.13733437e+00, 3.12113770e+02,
        6.53381187e-04, -2.45035619e+00, 6.27102327e+02,
        3.93808370e-06, -4.24034141e-03, 1.00000000e+00
    );

};
}

#endif // LANE_DETECTOR_HPP_