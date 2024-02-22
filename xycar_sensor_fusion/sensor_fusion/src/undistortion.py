import cv2
import os
import numpy as np
import yaml
import matplotlib.pyplot as plt 


def undistortion(path):
    config_path = './config.yaml'
    with open(config_path, "r") as f:
        config_json = yaml.safe_load(f)

    camera_matrix, dist = np.array(config_json["camera_matrix"]["data"]), np.array(config_json["distortion_coefficient"]["data"])

    img = cv2.imread(path)
    h, w = img.shape[:2]

    mapx, mapy = cv2.initUndistortRectifyMap(camera_matrix, dist, None, None, (w, h), 5)
    remap = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)

    cv2.imwrite("../image_new_undistort.png", remap)


def image_show(path):
    img = cv2.imread(path)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()

if __name__ == '__main__':
    image_path = '../image_new5.png'
    image_show(image_path)

    # distort_path = "../image_new.png"
    # undistortion(distort_path)