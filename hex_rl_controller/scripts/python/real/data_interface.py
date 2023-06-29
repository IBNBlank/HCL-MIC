#!/usr/bin/env python3
# -*- coding:utf-8 -*-
################################################################
# Copyright 2023 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2023-04-19
################################################################

import math
import rospy
import numpy as np

from geometry_msgs.msg import Twist, PointStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan

#### index ####
POSE_PX = 0
POSE_PY = 1
POSE_YAW = 2
VELOCITY_VX = 0
VELOCITY_VYAW = 1
INTENTION_VX = 0
INTENTION_YAW = 1


class DataInterface(object):

    def __init__(self):
        rospy.init_node(f"robot_real_env", anonymous=None)

        # Sensor Variable
        self.__laser_data = None
        self.__odom_data = None
        self.__intention_data = None
        self.__have_laser = False
        self.__have_odom = False
        self.__have_intention = False

        # Publisher
        self.__cmd_vel_pub = rospy.Publisher("rl_cmd_vel", Twist, queue_size=5)

        # Subscriber
        self.__laser_sub = rospy.Subscriber(f"scan",
                                            LaserScan,
                                            self.__laser_callback,
                                            queue_size=5)
        self.__odom_sub = rospy.Subscriber(f"odom",
                                           Odometry,
                                           self.__odom_callback,
                                           queue_size=5)
        self.__intention_sub = rospy.Subscriber(f"intention",
                                                Twist,
                                                self.__intention_callback,
                                                queue_size=5)

        # Wait until the first callback
        while (self.__laser_data is None) or (self.__odom_data is None) or (
                self.__intention_data is None):
            rospy.sleep(0.001)

    ########################################
    #### basic handle
    ########################################
    def sleep(self, duration):
        rospy.sleep(duration)

    def is_shutdown(self):
        return rospy.is_shutdown()

    ########################################
    #### get handle
    ########################################
    def get_laser(self):
        return np.array(self.__laser_data)

    def get_odom(self):
        return self.__odom_data

    def get_intention(self):
        return self.__intention_data

    def have_laser(self):
        return self.__have_laser

    def have_odom(self):
        return self.__have_odom

    def have_intention(self):
        return self.__have_intention

    def clear_laser(self):
        self.__have_laser = False

    def clear_odom(self):
        self.__have_odom = False

    def clear_intention(self):
        self.__have_intention = False

    ########################################
    #### publish handle
    ########################################
    def pub_cmd_vel(self, action):
        move_cmd = Twist()

        move_cmd.linear.x = action[VELOCITY_VX]
        move_cmd.linear.y = 0.0
        move_cmd.linear.z = 0.0

        move_cmd.angular.x = 0.0
        move_cmd.angular.y = 0.0
        move_cmd.angular.z = action[VELOCITY_VYAW]

        self.__cmd_vel_pub.publish(move_cmd)

    ########################################
    #### callback handle
    ########################################
    def __laser_callback(self, scan):
        self.__laser_data = scan.ranges
        self.__have_laser = True

    def __odom_callback(self, odometry):
        self.__odom_data = [
            odometry.pose.pose.position.x, odometry.pose.pose.position.y,
            2.0 * math.atan2(odometry.pose.pose.orientation.z,
                             odometry.pose.pose.orientation.w),
            odometry.twist.twist.linear.x, odometry.twist.twist.angular.z
        ]
        self.__have_odom = True

    def __intention_callback(self, intention):
        self.__intention_data = [intention.linear.x, intention.angular.z]
        self.__have_intention = True
