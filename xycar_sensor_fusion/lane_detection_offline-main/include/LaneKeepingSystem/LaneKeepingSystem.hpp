#ifndef LANE_KEEPING_SYSTEM_HPP_
#define LANE_KEEPING_SYSTEM_HPP_

#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <xycar_msgs/xycar_motor.h>
#include <sensor_msgs/LaserScan.h>
#include <yaml-cpp/yaml.h>
#include <yolov7/BoundingBoxes.h>
#include "LaneKeepingSystem/LaneDetector.hpp"
#include "LaneKeepingSystem/MovingAverageFilter.hpp"
#include "LaneKeepingSystem/PIDController.hpp"

namespace Xycar {
/**
 * @brief Lane Keeping System for searching and keeping Hough lines using Hough, Moving average and PID control
 *
 * @tparam Precision of data
 */
template <typename PREC>
class LaneKeepingSystem
{
public:
    using Ptr = LaneKeepingSystem*;                                     ///< Pointer type of this class
    using ControllerPtr = typename PIDController<PREC>::Ptr;            ///< Pointer type of PIDController
    using FilterPtr = typename MovingAverageFilter<PREC>::Ptr;          ///< Pointer type of MovingAverageFilter
    using DetectorPtr = typename LaneDetector<PREC>::Ptr;              ///< Pointer type of LaneDetecter(It's up to you)

    // calibration for xycar 168
    // static inline const cv::Mat mtx = (cv::Mat_<double>(3, 3) <<
    //     343.353, 0, 340.519,
    //     0,  344.648, 231.522,
    //     0, 0, 1 
    // );

    // static inline const cv::Mat dist = (cv::Mat_<double>(1, 4) <<
    //     -0.334698,  0.129289,  -0.001919,  0.000753 
    // );

    // calibration for xycar 155
    static inline const cv::Mat mtx = (cv::Mat_<double>(3, 3) <<
        379.323, 0, 320.000,
        0,  379.323, 240.000,
        0, 0, 1 
    );

    static inline const cv::Mat dist = (cv::Mat_<double>(1, 4) <<
        -0.387839,  0.143591,  -0.005453,  -0.000391
    );
    
    static inline cv::Mat mapx, mapy;
    PREC speed;
    int angle;
    clock_t start_time;
    clock_t end_time; 

    // vector<int
    static constexpr int32_t kXycarSteeringAangleLimit = 50; ///< Xycar Steering Angle Limit
    static constexpr double kFrameRate = 33.0;               ///< Frame rate
    /**
     * @brief Construct a new Lane Keeping System object
     */
    LaneKeepingSystem();

    /**
     * @brief Destroy the Lane Keeping System object
     */
    virtual ~LaneKeepingSystem();

    /**
     * @brief Run Lane Keeping System
     */
    void run();

private:
    /**
     * @brief Set the parameters from config file
     *
     * @param[in] config Configuration for searching and keeping Hough lines using Hough, Moving average and PID control
     */
    void setParams(const YAML::Node& config);

    /**
     * @brief Control the speed of xycar
     *
     * @param[in] steeringAngle Angle to steer xycar. If over max angle, deaccelerate, otherwise accelerate
     */
    void speedControl(PREC steeringAngle);

    /**
     * @brief publish the motor topic message
     *
     * @param[in] steeringAngle Angle to steer xycar actually
     */
    void drive(PREC steeringAngle);
    void driveSign(PREC steeringAngle,PREC speed );
    void driveForDuration(PREC steeringAngle, PREC speed, int duration);
    
    void imageCallback(const sensor_msgs::Image& message);   
    void yoloCallback(const yolov7::BoundingBoxes& boundingMsg);
    void lidarCallback(const sensor_msgs::LaserScan& lidarMsg);
    std::string getfilename();
    
    


private:
    ControllerPtr mPID;                      ///< PID Class for Control
    FilterPtr mMovingAverage;                ///< Moving Average Filter Class for Noise filtering
    DetectorPtr mLaneDetector;

    // ROS Variables
    ros::NodeHandle mNodeHandler;          ///< Node Hanlder for ROS. In this case Detector and Controler
    ros::Publisher mPublisher;             ///< Publisher to send message about
    ros::Subscriber mSubscriber;           ///< Subscriber to receive image
    std::string mPublishingTopicName;      ///< Topic name to publish
    std::string mSubscribedTopicName;      ///< Topic name to subscribe
    ros::Subscriber mSubscriberYolo;
    std::string mSubscribedTopicYolo;
    ros::Subscriber mSubscriberLiDAR;
    std::string mSubscribedTopicLiDAR;
    uint32_t mQueueSize;                   ///< Max queue size for message
    xycar_msgs::xycar_motor mMotorMessage; ///< Message for the motor of xycar

    // OpenCV Image processing Variables
    cv::Mat mFrame; ///< Image from camera. The raw image is converted into cv::Mat
    yolov7::BoundingBoxes mYolo;
    // std::vector<yolov7::BoundingBoxes> mbox_vector;

    // Xycar Device variables
    PREC mXycarSpeed;                 ///< Current speed of xycar
    PREC mXycarMaxSpeed;              ///< Max speed of xycar
    PREC mXycarMinSpeed;              ///< Min speed of xycar
    PREC mXycarSpeedControlThreshold; ///< Threshold of angular of xycar
    PREC mAccelerationStep;           ///< How much would accelrate xycar depending on threshold
    PREC mDecelerationStep;           ///< How much would deaccelrate xycar depending on threshold
    PREC temp_mDecelerationStep;
    cv::VideoWriter outputVideo; 
    // Debug Flag
    bool mDebugging; ///< Debugging or not

    int avoid_flag = 3; //0:left, 1:muiddle, 2:right, 3:ignore
};
} // namespace Xycar

#endif // LANE_KEEPING_SYSTEM_HPP_
