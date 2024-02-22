#!/usr/bin/env python3

import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np
import cv2
import matplotlib.pyplot as plt

import sys, os
import time
import numpy as np
import cv2
import tensorrt as trt
from PIL import Image,ImageDraw
import rospy

from std_msgs.msg import String
from yolov7.msg import BoundingBox, BoundingBoxes

from cv_bridge import CvBridge,CvBridgeError
from sensor_msgs.msg import Image as Imageros

DEBUG = False # visualization 

# from data_processing import PreprocessYOLO, PostprocessYOLO, ALL_CATEGORIES
# import common

# bridge = CvBridge()
xycar_image = np.empty(shape=[0])

class BaseEngine(object):
    def __init__(self):
        self.mean = None
        self.std = None
        self.n_classes = 9
        self.class_names = [ 'left', 'right', 'stop', 'crosswalk', 'green_light',  'yellow_light', 'red_light' , 'car', 'ignore']
        self.engine_path = '/home/nvidia/xycar_ws/src/yolov7/src/models/best.trt'

        # self.mtx = np.array([[349.766, 0, 319.369],[0, 350.415, 215.350],[0, 0, 1]])
        # self.dist = np.array([[ -0.334962, 0.100784, -0.000420, -0.000808 ]])

        # xycar 168
        # self.mtx = np.array([[ 343.353,  0,  340.519,], [0, 344.648,  231.522],  [0,  0,  1 ]])
        # self.dist = np.array([[ -0.334698,  0.129289,  -0.001919,  0.000753 ]])

        #xycar 155
        self.mtx = np.array([[379.323, 0, 320.000], [0,  379.323, 240.000],  [0,  0,  1 ]])
        self.dist = np.array([[ -0.387839,  0.143591,  -0.005453,  -0.000391]])
       
        self.mapx, self.mapy = cv2.initUndistortRectifyMap(self.mtx, self.dist, None, None, (640, 480), 5)
        self.homo_mat = np.array([
            [-1.39330020e-01, -1.15143435e+00,  3.10011197e+02],
            [-2.24482310e-02, -2.43835386e+00,  6.17336292e+02],
            [-3.54729928e-05, -4.26790689e-03,  1.00000000e+00]
            ])
        
        self.depth_thres = 107

        self.detection_pub = rospy.Publisher('/yolov3_trt_ros/detections', BoundingBoxes, queue_size=1)
        self.detect_lidar_pub = rospy.Publisher('/detection_lidar', BoundingBoxes, queue_size=1)


        logger = trt.Logger(trt.Logger.WARNING)
        logger.min_severity = trt.Logger.Severity.ERROR
        runtime = trt.Runtime(logger)
        trt.init_libnvinfer_plugins(logger,'') # initialize TensorRT plugins
        with open(self.engine_path, "rb") as f:
            serialized_engine = f.read()
        engine = runtime.deserialize_cuda_engine(serialized_engine)
        self.imgsz = engine.get_binding_shape(0)[2:]  # get the read shape of model, in case user input it wrong

        self.context = engine.create_execution_context()
        self.inputs, self.outputs, self.bindings = [], [], []
        self.stream = cuda.Stream()
        for binding in engine:
            size = trt.volume(engine.get_binding_shape(binding))
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))
            if engine.binding_is_input(binding):
                self.inputs.append({'host': host_mem, 'device': device_mem})
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem})

    def infer(self, img):
        self.inputs[0]['host'] = np.ravel(img)
        # transfer data to the gpu
        for inp in self.inputs:
            cuda.memcpy_htod_async(inp['device'], inp['host'], self.stream)
        # run inference
        self.context.execute_async_v2(
            bindings=self.bindings,
            stream_handle=self.stream.handle)
        # fetch outputs from gpu
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)
        # synchronize stream
        self.stream.synchronize()

        data = [out['host'] for out in self.outputs]
        return data

    def detect_video(self):
        end2end = True
        rate = rospy.Rate(10)
        image_sub = rospy.Subscriber("/usb_cam/image_raw", Imageros, self.img_callback)

        while not rospy.is_shutdown():
            rate.sleep()
            # if xycar_image is empty, skip inference
            if xycar_image.shape[0] == 0:
                continue

            # cap = cv2.VideoCapture(video_path)
            # fourcc = cv2.VideoWriter_fourcc(*'XVID')
            # fps = int(round(cap.get(cv2.CAP_PROP_FPS)))
            # print(xycar_image.shape[0])
            width = int(640)
            height = int(480)
            # out_yolo = cv2.VideoWriter('/home/nvidia/xycar_ws/src/yolov7/src/results_yolo.avi',fourcc,30,(width,height))
            # out_bev = cv2.VideoWriter('/home/nvidia/xycar_ws/src/yolov7/src/results_bev.avi',fourcc,30,(540,540))
            fps = 0
            import time
            while True:
                # ret, frame = cap.read()
                # if not ret:
                #     break
                frame = cv2.cvtColor(xycar_image, cv2.COLOR_BGR2RGB)
                
                if DEBUG : 
                    # bird eye view image
                    bev_img = np.zeros((540, 540, 3), dtype=np.uint8)
                    # vertical lines
                    for x in range(0, 540, 90):
                        cv2.line(bev_img, (x, 0), (x, 540), (255, 255, 255), 1)
                    # horizon lines
                    for y in range(0, 540, 90):
                        cv2.line(bev_img, (0, y), (540, y), (255, 255, 255), 1)
                else:
                    bev_img = None

                # cv2.imshow('frame', frame12)
                # TODO : check mean and std 
                blob, ratio = preproc(frame, self.imgsz, self.mean, self.std)
                t1 = time.time()
                data = self.infer(blob)
                # fps = (fps + (1. / (time.time() - t1))) / 2
                # frame = cv2.putText(frame12, "FPS:%d " %fps, (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    # (0, 0, 255), 2)

                # TODO: CHECK ALGO
                if end2end:
                    num, final_boxes, final_scores, final_cls_inds = data
                    # print(final_scores,final_cls_inds)
                    final_boxes = np.reshape(final_boxes/ratio, (-1, 4))
                    dets = np.concatenate([final_boxes[:num[0]], np.array(final_scores)[:num[0]].reshape(-1, 1), np.array(final_cls_inds)[:num[0]].reshape(-1, 1)], axis=-1)
                else:
                    predictions = np.reshape(data, (1, -1, int(5+self.n_classes)))[0]
                    dets = self.postprocess(predictions,ratio)
                    
                # print(dets)
                # if dets is not None:boxes
                
                if len(dets):
                    final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
                    frame, bev_img, new_box, new_score, new_ids, new_depth = self.vis(frame, bev_img, final_boxes, final_scores, final_cls_inds,
                                    conf=0.8, class_names=self.class_names)

                    self.publisher_for_lidar(new_box, new_score, new_ids, new_depth)
            
                    if len(new_depth) and min(new_depth) < self.depth_thres:
                        # print("depth yes")
                        idx = new_depth.index(min(new_depth))
                        self.publisher(new_box[idx], new_score[idx], new_ids[idx], new_depth[idx])
                    else:
                        # print("too far object")
                        self.publisher([0, 0, 0, 0], 0, -1, -1)
                else:
                    # print("no object")
                    self.publisher([0, 0, 0, 0], 0, -1, -1)
                    self.detect_lidar_pub.publish(BoundingBoxes())
                    # self.detect_lidar_pub.publish([[0, 0, 0, 0]], [0], [-1], [-1])

                if DEBUG : 
                    cv2.imshow('frame', frame)
                    # cv2.imshow('birdeye', bev_img)
                # cv2.waitKey(1)

                # out_yolo.write(frame)
                # out_bev.write(bev)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
            # out_yolo.release()
            # out_bev.release()
            # cap.release()
            # cv2.destroyAllWindows()

    def _write_message(self, detection_results, boxes, scores, classes,depth):
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
        
    def _write_message_lidar(self, detection_results, boxes, scores, classes,depth):
        """ populate output message with input header and bounding boxes information """
        if boxes is None:
            return None
	
        for box, score, category,d in zip(boxes, scores, classes, depth):
            minx, miny, maxx, maxy = box
            detection_msg = BoundingBox()
            detection_msg.xmin = int(minx)
            detection_msg.xmax = int(maxx)
            detection_msg.ymin = int(miny)
            detection_msg.ymax = int(maxy)
            detection_msg.probability = score
            detection_msg.id = int(category)
            detection_msg.depth = int(d)
            detection_results.bounding_boxes.append(detection_msg)
        return detection_results

    def publisher(self, boxes, confs, classes,depth):
        """ Publishes to detector_msgs
        Parameters:
        boxes (List(List(int))) : Bounding boxes of all objects
        confs (List(double))	: Probability scores of all objects
        classes  (List(int))	: Class ID of all classes
        """
        detection_results = BoundingBoxes()
        detection_results = self._write_message(detection_results, boxes, confs, classes, depth)
        self.detection_pub.publish(detection_results)

    def publisher_for_lidar(self, boxes, confs, classes,depth):
        """ Publishes to detector_msgs
        Parameters:
        boxes (List(List(int))) : Bounding boxes of all objects
        confs (List(double))	: Probability scores of all objects
        classes  (List(int))	: Class ID of all classes
        """
        detection_results = BoundingBoxes()
        detection_results = self._write_message_lidar(detection_results, boxes, confs, classes, depth)
        self.detect_lidar_pub.publish(detection_results)


    def inference(self, img_path, conf=0.005, end2end=False):
        origin_img = cv2.imread(img_path)
        img, ratio = preproc(origin_img, self.imgsz, self.mean, self.std)
        data = self.infer(img)
        if end2end:
            num, final_boxes, final_scores, final_cls_inds = data
            # print(final_boxes, final_scores,final_cls_inds )
            final_boxes = np.reshape(final_boxes/ratio, (-1, 4))
            dets = np.concatenate([final_boxes[:num[0]], np.array(final_scores)[:num[0]].reshape(-1, 1), np.array(final_cls_inds)[:num[0]].reshape(-1, 1)], axis=-1)
        else:
            predictions = np.reshape(data, (1, -1, int(5+self.n_classes)))[0]
            dets = self.postprocess(predictions,ratio)
        
        if dets is not None:
            # print(dets)
            final_boxes, final_scores, final_cls_inds = dets[:,
                                                             :4], dets[:, 4], dets[:, 5]

            
            origin_img = self.vis(origin_img, final_boxes, final_scores, final_cls_inds,
                             conf=0.8, class_names=self.class_names)
        

        return origin_img

    @staticmethod
    def postprocess(predictions, ratio):
        boxes = predictions[:, :4]
        scores = predictions[:, 4:5] * predictions[:, 5:]
        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.
        boxes_xyxy /= ratio
        dets = multiclass_nms(boxes_xyxy, scores, nms_thr=0.45, score_thr=0.1)
        return dets

    def get_fps(self):
        import time
        img = np.ones((1,3,self.imgsz[0], self.imgsz[1]))
        img = np.ascontiguousarray(img, dtype=np.float32)
        for _ in range(5):  # warmup
            _ = self.infer(img)

        t0 = time.perf_counter()
        for _ in range(100):  # calculate average time
            _ = self.infer(img)
        print(100/(time.perf_counter() - t0), 'FPS')

    def img_callback(self,data):
        global xycar_image
    
    # ## python2
    # xycar_image = bridge.imgmsg_to_cv2(data, "bgr8")

    # python3
        raw_img = np.frombuffer(data.data, dtype=np.uint8).reshape(data.height, data.width, -1)
        xycar_image = cv2.remap(raw_img, self.mapx, self.mapy, cv2.INTER_LINEAR)


    def vis(self, img, bev_img, boxes, scores, cls_ids, conf=0.5, class_names=None):
        new_box =[]
        new_scores =[]
        new_ids =[]
        new_depth = []

        # bev_img = img.copy()
        # bev_img = cv2.warpPerspective(bev_img, self.homo_mat, (540,540))
        # bev_img = np.zeros((540, 540, 3), dtype=np.uint8)

        for i in range(len(boxes)):
            box = boxes[i]
            cls_id = int(cls_ids[i])
            score = scores[i] +1
            if score < conf:
                continue

            new_box.append(box)
            new_scores.append(score)
            new_ids.append(cls_id)

            #estimate depth
            x0 = int(box[0])
            y0 = int(box[1])
            x1 = int(box[2])
            y1 = int(box[3])

            center_x = (x0 + x1)/2
            bbox_point = np.array([center_x, box[3], 1])
            estimate = np.dot(self.homo_mat, bbox_point)
            x, y, z = estimate[0], estimate[1], estimate[2]

            depth_x = x/z
            depth_y = y/z
            distance = int(np.sqrt(((270 - depth_x)/2)**2 + ((540-depth_y)/2)**2))
            # if (260 < depth_x < 280 and cls_id == 0):
            #     distance += 6
            # if (260 < depth_x < 280):
            #     distance += 6

            new_depth.append(distance)

            
            if DEBUG : 
                # draw box
                color = (_COLORS[cls_id] * 255).astype(np.uint8).tolist()
                text = '{}:{:.1f}%'.format(class_names[cls_id], score * 100)
                txt_color = (0, 0, 0) if np.mean(_COLORS[cls_id]) > 0.5 else (255, 255, 255)
                font = cv2.FONT_HERSHEY_SIMPLEX

                txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
                cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)

                txt_bk_color = (_COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
                cv2.rectangle(
                    img,
                    (x0, y0 + 1),
                    (x0 + txt_size[0] + 1, y0 + int(1.5 * txt_size[1])),
                    txt_bk_color,
                    -1
                )
                cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1)

                # draw depth
                depth_text = '{}:{}'.format(class_names[cls_id], distance)
                cv2.circle(bev_img, (int(depth_x), int(depth_y)), 10, color, -1)
                cv2.putText(bev_img, depth_text, (int(depth_x), int(depth_y) - 20), font, 1, color, thickness=1)
            
        
        if DEBUG : 
            if len(new_depth) and (min(new_depth) < self.depth_thres):
                cv2.putText(img, str(min(new_depth)), (320, 0), font, 3, txt_color, thickness=1 )
            
        return img,bev_img, new_box,new_scores,new_ids,new_depth


def nms(boxes, scores, nms_thr):
    """Single class NMS implemented in Numpy."""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= nms_thr)[0]
        order = order[inds + 1]

    return keep


def multiclass_nms(boxes, scores, nms_thr, score_thr):
    """Multiclass NMS implemented in Numpy"""
    final_dets = []
    num_classes = scores.shape[1]
    for cls_ind in range(num_classes):
        cls_scores = scores[:, cls_ind]
        valid_score_mask = cls_scores > score_thr
        if valid_score_mask.sum() == 0:
            continue
        else:
            valid_scores = cls_scores[valid_score_mask]
            valid_boxes = boxes[valid_score_mask]
            keep = nms(valid_boxes, valid_scores, nms_thr)
            if len(keep) > 0:
                cls_inds = np.ones((len(keep), 1)) * cls_ind
                dets = np.concatenate(
                    [valid_boxes[keep], valid_scores[keep, None], cls_inds], 1
                )
                final_dets.append(dets)
    if len(final_dets) == 0:
        return None
    return np.concatenate(final_dets, 0)


def preproc(image, input_size, mean, std, swap=(2, 0, 1)):
    if len(image.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3)) * 114.0
    else:
        padded_img = np.ones(input_size) * 114.0
    img = np.array(image)
    # print(img.shape)
    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.float32)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img
    # if use yolox set
    # padded_img = padded_img[:, :, ::-1]
    # padded_img /= 255.0
    padded_img = padded_img[:, :, ::-1]
    padded_img /= 255.0
    if mean is not None:
        padded_img -= mean
    if std is not None:
        padded_img /= std
    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img, r

def rainbow_fill(size=50):  # simpler way to generate rainbow color
    cmap = plt.get_cmap('jet')
    color_list = []

    for n in range(size):
        color = cmap(n/size)
        color_list.append(color[:3])  # might need rounding? (round(x, 3) for x in color)[:3]

    return np.array(color_list)


_COLORS = rainbow_fill(9).astype(np.float32).reshape(-1, 3)


if __name__ == '__main__':

    pred = BaseEngine()
    rospy.init_node('yolov7', anonymous=True)
    pred.detect_video()
    
    # pred.detect_video(video, conf=0.1, end2end=args.end2end) # set 0 use a webcam
