#!/usr/bin/env python3
import cv2
import yaml
import numpy as np
import matplotlib.pyplot as plt 

"""
lidar_points = np.array([
    # [0.2966, 0.4572, 0],
    # [0.2804, -0.4974, 0],
    [1.2232, 0.8543, 0],
    [1.2132, 0.4236, 0],
    [1.1817, -0.4970, 0],
    [1.1732, -0.8194, 0],
    [0.7613, 0.7590, 0],
    [0.7591, 0.4627, 0],
    [0.7280, -0.4690, 0],
    [0.7160, -0.8086, 0]
],  dtype=np.float32)
"""
config_path = './config.yaml'
with open(config_path, "r") as f:
    config_json = yaml.safe_load(f)

camera_matrix, dist = np.array(config_json["camera_matrix"]["data"]), np.array(config_json["distortion_coefficient"]["data"])
dist = np.array([0, 0 ,0, 0])

# distorted image
# image_points = np.array([
#     # [17.3081, 297.179],
#     # [600.881, 277.86],
#     [111.976, 263.996],
#     [181.837, 264.856],
#     [424.804, 257.352],
#     [493.624, 254.845],
#     [66.5582, 273.591],
#     [122.674, 276.64],
#     [474.01, 267.491],
#     [561.844, 295.562]
# ], dtype=np.float32)

# undistorted image
# image_points = np.array([
#     [80.0265, 267.63],
#     [171.374, 266.212],
#     [428.411, 257.451],
#     [509.017, 256.215],
#     [1.2928, 282.74],
#     [94.6163, 280.91],
#     [485.599, 269.321],
#     [614.3, 263.831]
# ], dtype=np.float32)

# image point : x(right), y(down)
image_points = np.array([
    [117.146, 306.516],
    [214.018, 303.339],
    [448.257, 296.56],
    [539.174, 291.032],
    [630.091, 290.238],
    [35.3871, 274.44],
    [80.6779, 273.811],
    [167.485, 271.924],
    [212.147, 270.037],
    [373.81, 266.891],
    [415.326, 263.746],
    [499.617, 263.117],
    [540.505, 261.23],
    [112.474, 263.309],
    [143.867, 263.309],
    [209.723, 262.285],
    [240.433, 260.579],
    [341.776, 258.873],
    [371.803, 257.508],
    [452.673, 255.461],
    [483.041, 254.779],
    [154.142, 258.628],
    [178.682, 258.628],
    [242.381, 257.061],
    [266.399, 256.8],
    [340.802, 254.451],
    [365.081, 253.667],
    [419.643, 252.884],
    [442.356, 252.101]
], dtype=np.float32)

# lidar point : x(right), y(depth), z(0, down)
lidar_points = np.array([
    [0.2361,0.3665,0],
    [0.1220,0.3640,0],
    [-0.2120,0.3578,0],
    [-0.3300,0.3567,0],
    [-0.4760,0.3619,0],
    [0.7535,0.9721,0],
    [0.6607,0.9716,0],
    [0.3755,0.9590,0],
    [0.2811,0.9554,0],
    [-0.2646,0.9425,0],
    [-0.3551,0.9412,0],
    [-0.6360,0.9352,0],
    [-0.7585,0.9298,0],
    [0.7541,1.4303,0],
    [0.6656,1.4307,0],
    [0.3416,1.4184,0],
    [0.2506,1.4241,0],
    [-0.2115,1.406,0],
    [-0.3195,1.403,0],
    [-0.6926,1.3967,0],
    [-0.8223,1.3876,0],
    [0.7678,1.8910,0],
    [0.6864,1.8901,0],
    [0.2823,1.8768,0],
    [0.1877,1.8796,0],
    [-0.3268,1.8564,0],
    [-0.3749,1.8585,0],
    [-0.7220,1.8436,0],
    [-0.8618,1.8523,0]
], dtype=np.float32)
'''
img5
[0.6756,0.9673]
[0.5725,0.9660]
[0.1356,1.2063]
[0.0597,1.2005]
[-0.2117,0.7195]
[-0.3117,0.7160]
[-0.6959,0.9454]
[-0.8010,0.9334]
img6
[0.5268,1.0625]
[0.2831,1.0588]
[-0.2117,0.7195]
[-0.4636,0.7195]
'''


_, rvec, tvec = cv2.solvePnP(lidar_points, image_points, camera_matrix, dist)
R, _ = cv2.Rodrigues(rvec)
r_inv = np.linalg.inv(R)

xyz = np.dot(-r_inv, tvec)
print("----camera position----")
print(xyz)

lidar_to_camera = np.hstack((R, tvec))
print("----lidar_to_camera----")
print(lidar_to_camera)

# test_lidar_points = np.array([
#     [0.5268,1.0625,0],
#     [0.2831,1.0588,0],
#     [-0.2117,0.7195,0],
#     [-0.4636,0.7195,0]
# ],  dtype=np.float32)

test_lidar_points = np.array([
    [0.7535,0.9721,0],
    [0.6607,0.9716,0],
    [0.3755,0.9590,0],
    [0.2811,0.9554,0],
    [-0.2646,0.9425,0],
    [-0.3551,0.9412,0],
    [-0.6360,0.9352,0],
    [-0.7585,0.9298,0]
],  dtype=np.float32)

XYZ1 = np.vstack([test_lidar_points.T, np.ones((1, test_lidar_points.shape[0]))])
xyz = np.dot(camera_matrix,np.dot(lidar_to_camera, XYZ1))
z = xyz[2, :]
x = (xyz[0, :] / z).astype(np.int32)
y = (xyz[1, :] / z).astype(np.int32)

# x = np.where(z > 0, (xyz[0, :] / z).astype(np.int32), -(xyz[0, :] / z).astype(np.int32))
# y = np.where(z > 0, (xyz[1, :] / z).astype(np.int32), -(xyz[1, :] / z).astype(np.int32))

img_x = x
img_y = y

print("x : ", img_x)
print("y : ", img_y)

filename = '../image_new2.png'
image = cv2.imread(filename, cv2.IMREAD_ANYCOLOR)

for i in range(len(img_x)):
    cv2.circle(image, (img_x[i], img_y[i]), 5, (0,0,255), -1)

plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.show()