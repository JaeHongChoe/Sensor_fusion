#include "LaneKeepingSystem/LaneKeepingSystem.hpp"

#define Record 0
#if Record
#include <ctime>
#endif

namespace Xycar {
template <typename PREC>
LaneKeepingSystem<PREC>::LaneKeepingSystem()
{
    std::string configPath;
    mNodeHandler.getParam("config_path", configPath);
    YAML::Node config = YAML::LoadFile(configPath);

    mPID = new PIDController<PREC>(config["PID"]["P_GAIN"].as<PREC>(), config["PID"]["I_GAIN"].as<PREC>(), config["PID"]["D_GAIN"].as<PREC>());
    mMovingAverage = new MovingAverageFilter<PREC>(config["MOVING_AVERAGE_FILTER"]["SAMPLE_SIZE"].as<uint32_t>());
    mLaneDetector = new LaneDetector<PREC>(config);
 
    setParams(config);

    mPublisher = mNodeHandler.advertise<xycar_msgs::xycar_motor>(mPublishingTopicName, mQueueSize);
    // cam image init
    mSubscriber = mNodeHandler.subscribe(mSubscribedTopicName, mQueueSize, &LaneKeepingSystem::imageCallback, this);
    // YOLO init
    mSubscriberYolo = mNodeHandler.subscribe(mSubscribedTopicYolo, mQueueSize, &LaneKeepingSystem::yoloCallback, this);
    //LiDAR init
    mSubscriberLiDAR = mNodeHandler.subscribe(mSubscribedTopicLiDAR, mQueueSize, &LaneKeepingSystem::lidarCallback, this);


    cv::initUndistortRectifyMap(mtx, dist, cv::Mat(), cv::Mat(), cv::Size(640, 480), CV_32FC1, mapx, mapy);
}   


template <typename PREC>
void LaneKeepingSystem<PREC>::setParams(const YAML::Node& config)
{
    mPublishingTopicName = config["TOPIC"]["PUB_NAME"].as<std::string>();
    mSubscribedTopicName = config["TOPIC"]["SUB_NAME"].as<std::string>();
    mSubscribedTopicYolo = config["TOPIC"]["YOLO_NAME"].as<std::string>();
    mSubscribedTopicLiDAR = config["TOPIC"]["LIDAR_NAME"].as<std::string>();
    mQueueSize = config["TOPIC"]["QUEUE_SIZE"].as<uint32_t>();
    mXycarSpeed = config["XYCAR"]["START_SPEED"].as<PREC>();
    mXycarMaxSpeed = config["XYCAR"]["MAX_SPEED"].as<PREC>();
    mXycarMinSpeed = config["XYCAR"]["MIN_SPEED"].as<PREC>();
    mXycarSpeedControlThreshold = config["XYCAR"]["SPEED_CONTROL_THRESHOLD"].as<PREC>();
    mAccelerationStep = config["XYCAR"]["ACCELERATION_STEP"].as<PREC>();
    mDecelerationStep = config["XYCAR"]["DECELERATION_STEP"].as<PREC>();    
    mDebugging = config["DEBUG"].as<bool>();


#if (Record)
    outputVideo.open(getfilename(),  cv::VideoWriter::fourcc('m', 'p', '4', 'v'), 30.0, cv::Size(640, 480), true);
#endif
}

template <typename PREC>
LaneKeepingSystem<PREC>::~LaneKeepingSystem()
{
    delete mPID;
    delete mMovingAverage;
    delete mLaneDetector;
}

template <typename PREC>
void LaneKeepingSystem<PREC>::run()
{
    int lidarAngleThreshold = 20;

    ros::Rate rate(kFrameRate);  

    while (ros::ok())
    {                 
        ros::spinOnce();  
    
        if (mFrame.empty())
            continue;

        if (mYolo.bounding_boxes.empty()){
            std::cout <<"yolo loading........\n";
            continue;
        }

        double center = mLaneDetector->processImage(mFrame,mYolo);        
        double error = (center - mLaneDetector->getWidth()/2);

        auto steeringAngle = (int32_t)mPID->getControlOutput(error);
        temp_mDecelerationStep = std::round(std::abs(error)/10)*mDecelerationStep;

        // if(avoid_flag != 3){
        //     std::cout << "-------avoid_flag : " << avoid_flag << std::endl;
        // }

        //using lidar
        if(avoid_flag == 1){ //stop
            std::cout << "lidar stop !!!!" << std::endl;
            driveSign(0, 0);
            continue;

            // if(abs(steeringAngle) < lidarAngleThreshold){
            //     std::cout << "lidar stop <<<<<<<<" << std::endl;

            //     driveSign(0, 0);
            //     continue;
            // }else{
            //     avoid_flag = 3;
            // }
        }
        else if(avoid_flag == 0){ //left
            std::cout << "lidar left <<<<<<<<" << std::endl;
            driveSign(50, 5);
            sleep(1);
            driveSign(10, 5);
            sleep(1);
            driveSign(-40, 5);
            sleep(1);
            continue;
            // if(abs(steeringAngle) < lidarAngleThreshold ){
            //     std::cout << "lidar left <<<<<<<<" << std::endl;
            //     driveSign(50, 5);
            //     sleep(1);
            //     driveSign(10, 5);
            //     sleep(1);
            //     driveSign(-35, 5);
            //     sleep(1);
            //     continue;
            // }
            // else{
            //     avoid_flag = 3;
            // }
        }
        else if(avoid_flag == 2){ //right
            std::cout << "lidar right >>>>>>>>" << std::endl;
            driveSign(-40, 5);
            sleep(1);
            driveSign(30, 5);
            sleep(1);
            driveSign(50, 5);
            sleep(1);
            continue;
            // if(abs(steeringAngle) < lidarAngleThreshold ){
            //     std::cout << "lidar right <<<<<<<<" << std::endl;
            //     driveSign(-50, 5);
            //     sleep(1);
            //     driveSign(20, 5);
            //     sleep(1);
            //     driveSign(40, 5);
            //     sleep(1);
            //     continue;
            //     }
            // else{
            //     avoid_flag = 3;
            // }
        } 

        // stop line
        // std::cout << " bounding_boxes : " <<mYolo.bounding_boxes[0].id<<std::endl;
        if(mLaneDetector->stopline_flag == true && mYolo.bounding_boxes[0].id == -1){    
            driveForDuration(0,0,3000);  
            driveForDuration(0,6,2000);  

            std::cout << " stop line " <<std::endl;
            mLaneDetector->stopline_flag = false;
            continue;            
        }

        //sign
        if(mYolo.bounding_boxes[0].id != -1){
            switch (mYolo.bounding_boxes[0].id) {
                case 0: // left
                    driveForDuration(-50, 8, 2000);
                    std::cout << " left " <<std::endl;
                    break;
                    // if(mYolo.bounding_boxes[0].depth < 90){

                case 1: // right
                    driveForDuration(50, 8, 2000);
                    std::cout << " right " <<std::endl;
                    break;

                case 2: //stop
                case 3: //crosswalk
                    if(mYolo.bounding_boxes[0].depth < 60){
                        driveForDuration(0, 0, 3000);
                        driveForDuration(0, 6, 2000);

                        std::cout << " stop or crosswalk" <<std::endl;
                    }
                    else{
                        drive(steeringAngle);
                    }
                    mLaneDetector->stopline_flag = false;
                    break;

                case 4: // green ligth
                    drive(steeringAngle);
                    std::cout << " green " <<std::endl;
                    mLaneDetector->stopline_flag = false;
                    break;

                case 6: // red light
                    if(mYolo.bounding_boxes[0].depth < 60){
                        driveSign(0, 0);
                        std::cout << " red " <<std::endl;
                        break;
                    }
                    mLaneDetector->stopline_flag = false;
                     
                default:
                    drive(steeringAngle);
                    // std::cout << " drive " <<std::endl;
                    break;               
            }
        }else{         
            drive(steeringAngle);  
            // std::cout << " drive " <<std::endl;              
        }

        if (mDebugging)
        {       
            // cv::imshow("frame", mFrame);
            // cv::imshow("roi", mLaneDetector->getDebugROI());
            cv::imshow("Debug", mLaneDetector->getDebugFrame());

#if Record
            outputVideo.write( mLaneDetector->getDebugFrame());
            // outputVideo.write(mFrame);
#endif
            cv::waitKey(1);
        }
    }
    // outputVideo.release();
}

template <typename PREC>
void LaneKeepingSystem<PREC>::imageCallback(const sensor_msgs::Image& message)
{
    cv::Mat src = cv::Mat(message.height, message.width, CV_8UC3, const_cast<uint8_t*>(&message.data[0]), message.step);
    cv::cvtColor(src, mFrame, cv::COLOR_RGB2BGR);   

    cv::remap(mFrame, mFrame, mapx, mapy, cv::INTER_LINEAR);
}

template <typename PREC>
void LaneKeepingSystem<PREC>::yoloCallback(const yolov7::BoundingBoxes& boundingMsg)
{
    mYolo = boundingMsg;
}


template <typename PREC>
void LaneKeepingSystem<PREC>::lidarCallback(const sensor_msgs::LaserScan& lidarMsg){

    int left_cnt = 0;
    int right_cnt = 0;
    int middle_cnt = 0;

	for (int i = 0; i <= 56; i++) { // 0 ~ 40 degree
        // std::cout << lidarMsg.ranges[i] << "  ";
        if(lidarMsg.ranges[i] < 0.3 && lidarMsg.ranges[i] > 0.01){
            left_cnt += 1;
        }
        if(lidarMsg.ranges[i] < 0.35 && lidarMsg.ranges[i] > 0.01){
            middle_cnt += 1;
        }
	}
    
	for (int i = 448; i < 505; i++) { //320 ~ 360 degree
        if(lidarMsg.ranges[i] < 0.3 && lidarMsg.ranges[i] > 0.01){
            right_cnt += 1;
        }
        if(lidarMsg.ranges[i] < 0.35 && lidarMsg.ranges[i] > 0.01){
            middle_cnt += 1;
        }
	}

    if(middle_cnt >= 50){
        avoid_flag = 1;
    }
    else if(left_cnt >= 15){
        avoid_flag = 0;
    }
    else if(right_cnt >= 15){
        avoid_flag = 2;
    }
    else{
        avoid_flag = 3;
    }

    // std::cout << "-----------------------avoid_flag : " << avoid_flag << std::endl;
}

template <typename PREC>
void LaneKeepingSystem<PREC>::speedControl(PREC steeringAngle)
{
    if (std::abs(steeringAngle) > mXycarSpeedControlThreshold)
    {
        mXycarSpeed -= temp_mDecelerationStep;
        mXycarSpeed = std::max(mXycarSpeed, mXycarMinSpeed);
        return;
    }

    mXycarSpeed += mAccelerationStep;
    mXycarSpeed = std::min(mXycarSpeed, mXycarMaxSpeed);
}

template <typename PREC>
void LaneKeepingSystem<PREC>::drive(PREC steeringAngle)
{
    xycar_msgs::xycar_motor motorMessage;
    // motorMessage.angle = std::round(steeringAngle);
    motorMessage.angle = steeringAngle;

    speedControl(steeringAngle);
    mLaneDetector->setYOffset(mXycarSpeed);
    motorMessage.speed = std::round(mXycarSpeed);
    mPublisher.publish(motorMessage);
}

template <typename PREC>
void LaneKeepingSystem<PREC>::driveSign(PREC steeringAngle, PREC speed )
{
    xycar_msgs::xycar_motor motorMessage;
    // motorMessage.angle = std::round(steeringAngle);
    motorMessage.angle = steeringAngle;

    motorMessage.speed = std::round(speed);
    mPublisher.publish(motorMessage);
}

template <typename PREC>
void LaneKeepingSystem<PREC>::driveForDuration(PREC steeringAngle, PREC speed, int duration)
{
    start_time = clock();

    while(true){
        end_time = clock();
        driveSign(steeringAngle, speed);

        if(((end_time - start_time) / (double)CLOCKS_PER_SEC * 1000) > duration){
            break;
        }
    }
}

template <typename PREC>
std::string LaneKeepingSystem<PREC>::getfilename(){
    std::string str_buf;            
    time_t curTime = time(NULL); 
    struct tm* pLocal = localtime(&curTime);
    
    str_buf="/home/nvidia/xycar_ws/src/lane_detection_offline-main/"+std::to_string(pLocal->tm_year + 1900)+std::to_string(pLocal->tm_mon + 1)+std::to_string(pLocal->tm_mday)+ "_" + std::to_string(pLocal->tm_hour) + std::to_string(pLocal->tm_min) + std::to_string(pLocal->tm_sec)+".mp4";
    return str_buf;
}


template class LaneKeepingSystem<float>;
template class LaneKeepingSystem<double>;
} // namespace Xycar
