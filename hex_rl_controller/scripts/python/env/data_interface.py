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

from geometry_msgs.msg import Twist, Pose, PointStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_srvs.srv import Empty
from std_msgs.msg import Int8

#### index ####
POSE_PX = 0
POSE_PY = 1
POSE_YAW = 2
VELOCITY_VX = 0
VELOCITY_VYAW = 1
INTENTION_VX = 0
INTENTION_YAW = 1


class DataInterface(object):

    def __init__(self, index):
        self.__index = index
        rospy.init_node(f"robot_{index}_env", anonymous=None)

        # Sensor Variable
        self.__laser_data = None
        self.__odom_data = None
        self.__crashed_data = None
        self.__ground_truth_data = None
        self.__have_laser = False
        self.__have_odom = False
        self.__have_crashed = False
        self.__have_ground_truth = False

        # Service
        self.__reset_srv = rospy.ServiceProxy('reset_positions', Empty)

        # Publisher
        self.__pause_pub = rospy.Publisher(f"/hex_pause", Int8, queue_size=1)
        self.__cmd_vel_pub = rospy.Publisher(f"robot_{index}/cmd_vel",
                                             Twist,
                                             queue_size=1)
        self.__cmd_pose_pub = rospy.Publisher(f"robot_{index}/cmd_pose",
                                              Pose,
                                              queue_size=1)
        self.__goal_point_pub = rospy.Publisher(f"robot_{index}/goal_point",
                                                PointStamped,
                                                queue_size=1)
        self.__intention_pub = rospy.Publisher(f"robot_{index}/intention",
                                               Twist,
                                               queue_size=1)

        # Subscriber
        self.__laser_sub = rospy.Subscriber(f"robot_{index}/base_scan",
                                            LaserScan, self.__laser_callback)
        self.__odom_sub = rospy.Subscriber(f"robot_{index}/odom", Odometry,
                                           self.__odom_callback)
        self.__crashed_sub = rospy.Subscriber(f"robot_{index}/is_crashed", Int8,
                                              self.__crash_callback)
        self.__ground_truth_sub = rospy.Subscriber(
            f"robot_{index}/base_pose_ground_truth", Odometry,
            self.__ground_truth_callback)

        # Wait until the first callback
        while (self.__laser_data is None) or (self.__odom_data is None) or (
                self.__crashed_data is None) or (self.__ground_truth_data is
                                                 None):
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

    def get_crashed(self):
        return self.__crashed_data

    def get_ground_truth(self):
        return self.__ground_truth_data

    def have_laser(self):
        return self.__have_laser

    def have_odom(self):
        return self.__have_odom

    def have_crashed(self):
        return self.__have_crashed

    def have_ground_truth(self):
        return self.__have_ground_truth

    def clear_laser(self):
        self.__have_laser = False

    def clear_odom(self):
        self.__have_odom = False

    def clear_crashed(self):
        self.__have_crashed = False

    def clear_ground_truth(self):
        self.__have_ground_truth = False

    ########################################
    #### reset handle
    ########################################
    def reset_state(self, reset_pose):
        self.__odom_data = [
            reset_pose[POSE_PX], reset_pose[POSE_PY], reset_pose[POSE_YAW], 0.0,
            0.0
        ]
        self.__ground_truth_data = [
            reset_pose[POSE_PX], reset_pose[POSE_PY], reset_pose[POSE_YAW], 0.0,
            0.0
        ]
        self.__crashed_data = False

    ########################################
    #### service handle
    ########################################
    def srv_reset(self):
        self.__reset_srv()

    ########################################
    #### publish handle
    ########################################
    def pub_pause(self, pause):
        pause_msg = Int8()
        pause_msg.data = 1 if pause else 0
        
        self.__pause_pub.publish(pause_msg)
    
    def pub_cmd_vel(self, action):
        move_cmd = Twist()

        move_cmd.linear.x = action[VELOCITY_VX]
        move_cmd.linear.y = 0.0
        move_cmd.linear.z = 0.0

        move_cmd.angular.x = 0.0
        move_cmd.angular.y = 0.0
        move_cmd.angular.z = action[VELOCITY_VYAW]

        self.__cmd_vel_pub.publish(move_cmd)

    def pub_cmd_pose(self, pose):
        pose_cmd = Pose()

        pose_cmd.position.x = pose[POSE_PX]
        pose_cmd.position.y = pose[POSE_PY]
        pose_cmd.position.z = 0

        pose_cmd.orientation.x = 0
        pose_cmd.orientation.y = 0
        pose_cmd.orientation.z = np.sin(0.5 * pose[POSE_YAW])
        pose_cmd.orientation.w = np.cos(0.5 * pose[POSE_YAW])

        self.__cmd_pose_pub.publish(pose_cmd)

    def pub_goal_point(self, point):
        goal_point = PointStamped()

        goal_point.header.frame_id = f"/robot_{self.__index}/odom"
        goal_point.header.stamp = rospy.Time().now()

        goal_point.point.x = point[POSE_PX]
        goal_point.point.y = point[POSE_PY]
        goal_point.point.z = 0.0

        self.__goal_point_pub.publish(goal_point)

    def pub_intention(self, intention):
        intention_msg = Twist()

        intention_msg.linear.x = intention[INTENTION_VX]
        intention_msg.linear.y = 0.0
        intention_msg.linear.z = 0.0

        intention_msg.angular.x = 0.0
        intention_msg.angular.y = 0.0
        intention_msg.angular.z = intention[INTENTION_YAW]

        self.__intention_pub.publish(intention_msg)

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

    def __crash_callback(self, flag):
        self.__crashed_data = flag.data == 1
        self.__have_crashed = True

    def __ground_truth_callback(self, ground_truth):
        self.__ground_truth_data = [
            ground_truth.pose.pose.position.x,
            ground_truth.pose.pose.position.y,
            2.0 * math.atan2(ground_truth.pose.pose.orientation.z,
                             ground_truth.pose.pose.orientation.w),
            np.sqrt(ground_truth.twist.twist.linear.x *
                    ground_truth.twist.twist.linear.x +
                    ground_truth.twist.twist.linear.y *
                    ground_truth.twist.twist.linear.y),
            ground_truth.twist.twist.angular.z
        ]
        self.__have_ground_truth = True
