#!/usr/bin/env python3
import cv2
import time
import rospy
import math
import csv
import numpy as np
from sensor_msgs.msg import Image as Imageros
from sensor_msgs.msg import LaserScan
import matplotlib.pyplot as plt # Drawing Graph
import yaml

from yolov7.msg import BoundingBox, BoundingBoxes


DEBUG = True

xycar_image = np.empty(shape=[0])
lidar_angle = 0.012466637417674065
xycar_lidar = []
yolo_box = BoundingBoxes()

config_path = '/home/nvidia/xycar_ws/src/sensor_fusion/src/config.yaml'
with open(config_path, "r") as f:
    config_json = yaml.safe_load(f)

camera_matrix, dist = np.array(config_json["camera_matrix"]["data"]), np.array(config_json["distortion_coefficient"]["data"])
extrinsic_matrix = np.array(config_json["extrinsic_matrix"]["data"])

h, w = 480, 640
mapx, mapy = cv2.initUndistortRectifyMap(camera_matrix, dist, None, None, (w, h), 5)

def img_callback(data):
    global xycar_image
    global xycar_raw

    raw_img = np.frombuffer(data.data, dtype=np.uint8).reshape(data.height, data.width, -1)
    xycar_raw = raw_img
    xycar_image = cv2.remap(raw_img,mapx, mapy, cv2.INTER_LINEAR)

def lidar_callback(data):
    global xycar_lidar
    xycar_lidar = data.ranges
    
    global lidar_intensity
    global mean
    lidar_intensity = data.intensities
    if len(lidar_intensity) > 0:
        mean = sum(lidar_intensity) / len(lidar_intensity)

    # print(xycar_lidar)

def box_callback(data):
    '''
    bounding_boxes: 
  - 
    probability: 0.97314453125
    xmin: 359
    ymin: 180
    xmax: 462
    ymax: 303
    id: 2
    depth: 60
    '''
    global yolo_box
    yolo_box = data
    # print(yolo_box.bounding_boxes)

def cam_write(video_path):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    height, width = xycar_image.shape[:2]

    out_frame = cv2.VideoWriter(video_path, fourcc, 30, (width, height))

    return out_frame

def save_csv(data, filename):
    with open(filename, "a") as f:
        writer = csv.writer(f)
        writer.writerow(data)

def lidar_to_image(xycar_lidar, frame, canvas):
    point = []
    lidar_projection= []
    depth = []
    # tic = math.radians(360/505)
    tic = 0.012466637417674065
    for i in range(len(xycar_lidar)):
        # x<->y changed 02.14.11:24
        y = xycar_lidar[i]*math.cos(i*tic)
        x = xycar_lidar[i]*math.sin(i*tic)
        if 0.1 <  xycar_lidar[i] < 3.0 and y > 0:
            point.append([x, y, 0])
            depth.append(xycar_lidar[i])
        
        # draw_intensity(-x, y, lidar_intensity[i], canvas)
    if not len(point):
        return [],[]
    X = np.array(point, float)

    XYZ1 = np.vstack([X.T, np.ones((1, X.shape[0]))])
    xyz = np.dot(camera_matrix,np.dot(extrinsic_matrix, XYZ1))
    img_z = xyz[2, :]
    img_x = (xyz[0, :] / img_z).astype(np.int32)
    img_y = (xyz[1, :] / img_z).astype(np.int32)
    for i in range(len(img_x)):
        # cv2.circle(frame, (img_x[i], img_y[i]), 3, (0,0,255), -1)
        lidar_projection.append([img_x[i], img_y[i]])

    return lidar_projection, depth



def write_message(detection_results, boxes, scores, classes,depth):
    """ populate output message with input header and bounding boxes information """
    if boxes is None:
        return None

    minx, miny, maxx, maxy = boxes
    detection_msg = BoundingBox()
    detection_msg.xmin = int(minx)
    detection_msg.xmax = int(maxx)
    detection_msg.ymin = int(miny)
    detection_msg.ymax = int(maxy)
    detection_msg.probability = scores
    detection_msg.id = int(classes)
    detection_msg.depth = int(depth)
    detection_results.bounding_boxes.append(detection_msg)

    return detection_results


def publisher(boxes, confs, classes,depth):
    """ Publishes to detector_msgs
    Parameters:
    boxes (List(List(int))) : Bounding boxes of all objects
    confs (List(double))	: Probability scores of all objects
    classes  (List(int))	: Class ID of all classes
    """
    detection_results = BoundingBoxes()
    detection_results = write_message(detection_results, boxes, confs, classes, depth)
    lidar_pub.publish(detection_results)

def get_data():
    global lidar_pub

    end2end = True
    rate = rospy.Rate(10)
    image_sub = rospy.Subscriber("/usb_cam/image_raw", Imageros, img_callback)
    lidar_sub = rospy.Subscriber("/scan", LaserScan, lidar_callback, queue_size=1)
    box_sub = rospy.Subscriber('/detection_lidar', BoundingBoxes, box_callback)
    lidar_pub = rospy.Publisher('/lidar_pub', BoundingBoxes, queue_size=1)


    # time_flag = 1
    # file_name = "new6"
    # csv_path = f'/home/nvidia/xycar_ws/src/sensor_fusion/lidar_{file_name}.csv'
    # img_path = f'/home/nvidia/xycar_ws/src/sensor_fusion/image_{file_name}.png'
    # img_raw_path = f'/home/nvidia/xycar_ws/src/sensor_fusion/raw_image_{file_name}.png'

    while not rospy.is_shutdown():
        rate.sleep()
        # if xycar_image is empty, skip inference
        if xycar_image.shape[0] == 0:
            continue

        # out_frame = cam_write('/home/nvidia/xycar_ws/src/yolov7/src/results_yolo.avi')
        
        while True:
            frame = cv2.cvtColor(xycar_image, cv2.COLOR_BGR2RGB)   
            frame_raw = cv2.cvtColor(xycar_raw, cv2.COLOR_BGR2RGB)
            
            # Get empty canvas 
            canvas = np.zeros((750, 750, 3), dtype="uint8")

            # Get random numpy size 505
            data = np.random.rand(505)*10
            
            if len(xycar_lidar):
                lidar_project_point, lidar_depth = lidar_to_image(xycar_lidar, frame, canvas)
                
            # yolo box
            if (yolo_box.bounding_boxes):
                # if yolo_box.bounding_boxes[0].id == -1:
                #     publisher([0, 0, 0, 0], 0, -1, -1)
                #     continue

                yolo_box_use = yolo_box.bounding_boxes
                lidar_box_depth = []
                homo_depth = []
                # for bbox in yolo_box.bounding_boxes:
                for bbox in yolo_box_use:

                    box_depth = []

                    x0 = bbox.xmin
                    y0 = bbox.ymin
                    x1 = bbox.xmax
                    y1 = bbox.ymax

                    for i in range(len(lidar_project_point)):
                        if x0 < lidar_project_point[i][0] < x1 and y0 < lidar_project_point[i][1] < y1:
                            cv2.circle(frame, (lidar_project_point[i][0], lidar_project_point[i][1]), 3, (0,0,255), -1)
                            box_depth.append(lidar_depth[i])

                    median_depth = round(np.median(box_depth),2)
                    lidar_box_depth.append(median_depth)
                    homo_depth.append(bbox.depth)

                    if not math.isnan(median_depth) :
                        cv2.putText(frame, 'lidar : ' + str(int((median_depth - 0.05) * 100)), (x0, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                        cv2.putText(frame, 'img : ' + str(bbox.depth), (x0, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
                        cv2.rectangle(frame, (x0, y0), (x1, y1), (255,0,0), 2)

                if len(lidar_box_depth) == 1 and math.isnan(lidar_box_depth[0]):
                    idx = homo_depth.index(min(homo_depth))
                    result_box = yolo_box_use[idx]
                    
                    if result_box.depth < 100 :
                        publisher((result_box.xmin, result_box.ymin, result_box.xmax, result_box.ymax), result_box.probability, result_box.id, result_box.depth)
                    else:
                        publisher([0, 0, 0, 0], 0, -1, -1)
                else:
                    lidar_box_depth = [x if not math.isnan(x) else 5 for x in lidar_box_depth]
                    idx = lidar_box_depth.index(min(lidar_box_depth))
                    result_box = yolo_box_use[idx]

                    if lidar_box_depth[idx]*100 < 150:
                        publisher((result_box.xmin, result_box.ymin, result_box.xmax, result_box.ymax), result_box.probability, result_box.id, int(lidar_box_depth[idx]*100))
                    else:
                        publisher([0, 0, 0, 0], 0, -1, -1)
            else:
                # publisher(BoundingBoxes())
                publisher([0, 0, 0, 0], 0, -1, -1)


            if DEBUG : 
                cv2.imshow('frame', frame)
                # cv2.imshow('raw', frame_raw)
                # cv2.imshow("image", canvas)

            # out_yolo.write(frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        # out_frame.release()
        # cv2.destroyAllWindows()
        


if __name__ == '__main__':
    rospy.init_node('sensor_fusion', anonymous=True)
    get_data()