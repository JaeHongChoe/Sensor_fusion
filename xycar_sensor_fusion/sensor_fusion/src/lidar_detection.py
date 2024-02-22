#!/usr/bin/env python3
import matplotlib.pyplot as plt # Drawing Graph
import numpy as np  # Array
import math
from sensor_msgs.msg import LaserScan
import time
import rospy

# Add   <node name="__" pkg="__" type="trt_detection.py" output="screen" />
# Global variables
tic = math.radians(360/505)
n = 505
point = []
lidar_points = None

def lidar_callback(data):
    global lidar_points
    lidar_points = data.ranges
    
rospy.init_node('my_lidar', anonymous=True)
rospy.Subscriber("/scan", LaserScan, lidar_callback, queue_size=1)

while not rospy.is_shutdown():
  if lidar_points == None:
      continue

  rtn = ""
  for i in range(12):
      rtn += str(format(lidar_points[i*30],'.2f')) + ", "

  print(rtn[:-2])
  time.sleep(1.0)

# Get world coordinate points (x, y, 0)
for i in range(len(frame)):
  try:
    float_number = round(float(frame[i]), 5)
  except ValueError:
    continue
  if float_number >= 0.1:
    x = float_number*math.cos(i*tic)
    y = float_number*math.sin(i*tic)
    point.append([x,y])
 
X = np.array(point, float)

plt.scatter(X[:,0], X[:,1])
plt.show()
