#!/usr/bin/env python3
# -*- coding:utf-8 -*-
################################################################
# Copyright 2023 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2023-06-27
################################################################

import numpy as np
import math
import cv2
import rospy
from sensor_msgs.msg import Joy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image

CMD_LINEAR_X = 0
CMD_ANGULAR_Z = 1
JOY_LINEAR_X = 0
JOY_ANGULAR_Z = 1
INTENSION_LINEAR = 0
INTENSION_ANGULAR = 1


class RealGui:

    def __init__(self):
        rospy.init_node(f"robot_real_gui", anonymous=None)

        # Variable
        self.__image_data = np.zeros((720, 960, 3), dtype=np.uint8)
        self.__cmd_data = [0.0, 0.0]
        self.__joy_data = [0.0, 0.0]
        self.__intension = [0.0, 0.0]
        self.__column_mid = int(265 * 0.5)
        self.__intension_linear_height = 0
        self.__intension_angular_height = self.__column_mid
        self.__cmd_linear_height = 0
        self.__cmd_angular_height = self.__column_mid

        # Publisher
        self.__intension_pub = rospy.Publisher("/intention",
                                               Twist,
                                               queue_size=5)

        # Subscriber
        self.__image_sub = rospy.Subscriber("/camera/image_raw",
                                            Image,
                                            self.__img_callback,
                                            queue_size=5)
        self.__cmd_sub = rospy.Subscriber("/rl_cmd_vel",
                                          Twist,
                                          self.__cmd_callback,
                                          queue_size=5)
        self.__joy_sub = rospy.Subscriber("/joy",
                                          Joy,
                                          self.__joy_callback,
                                          queue_size=5)

        # Image
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 2.0
        color = (255, 230, 230)
        self.__gui_image = np.zeros(
            (900, 1600, 3), dtype=np.uint8) + np.asarray([50, 0, 0],
                                                         dtype=np.uint8)
        text = "intention"
        org = (1180, 115)
        cv2.putText(self.__gui_image,
                    text,
                    org,
                    font,
                    fontScale,
                    color,
                    thickness=4)
        text = "cmd vel"
        org = (1190, 520)
        cv2.putText(self.__gui_image,
                    text,
                    org,
                    font,
                    fontScale,
                    color,
                    thickness=4)
        text = "Author: Dong Zhaorui"
        fontScale = 0.5
        org = (1400, 890)
        cv2.putText(self.__gui_image,
                    text,
                    org,
                    font,
                    fontScale,
                    color,
                    thickness=1)

    def __img_callback(self, msg):
        raw_image = np.ndarray(shape=(msg.height, msg.width, 3),
                               dtype=np.uint8,
                               buffer=msg.data)
        raw_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
        self.__image_data = cv2.resize(raw_image, (960, 720))

    def __cmd_callback(self, msg):
        self.__cmd_data[CMD_LINEAR_X] = msg.linear.x
        self.__cmd_data[CMD_ANGULAR_Z] = msg.angular.z

    def __joy_callback(self, msg):
        self.__joy_data[
            JOY_LINEAR_X] = 0.0 if msg.axes[1] < 0.0 else msg.axes[1]
        self.__joy_data[JOY_ANGULAR_Z] = msg.axes[2]

    def __cal_intension(self):
        self.__intension[INTENSION_LINEAR] = self.__joy_data[JOY_LINEAR_X] - 0.5
        self.__intension[
            INTENSION_ANGULAR] = self.__joy_data[JOY_ANGULAR_Z] * 0.5

    def __pub_intension(self):
        intension = Twist()
        intension.linear.x = self.__joy_data[JOY_LINEAR_X]
        intension.angular.z = self.__joy_data[JOY_ANGULAR_Z]
        self.__intension_pub.publish(intension)

    def __draw_gui_image(self):
        #### draw image ####
        self.__gui_image[90:810, 90:1050] = self.__image_data

        #### draw intension ####
        intension_image = np.zeros(
            (265, 370, 3), dtype=np.uint8) + np.asarray([50, 0, 0],
                                                        dtype=np.uint8)
        # linear
        intension_linear_height = 265.0 * (0.5 -
                                           self.__intension[INTENSION_LINEAR])
        if (not math.isnan(intension_linear_height)) and (
                not math.isinf(intension_linear_height)):
            self.__intension_linear_height = int(intension_linear_height)
            self.__intension_linear_height = 0 if self.__intension_linear_height < 0 else self.__intension_linear_height
            self.__intension_linear_height = 265 if self.__intension_linear_height > 265 else self.__intension_linear_height
        intension_image[self.__intension_linear_height:265,
                        0:140] = (100, 255, 100)
        # angular
        intension_angular_height = 265.0 * (0.5 -
                                            self.__intension[INTENSION_ANGULAR])
        if (not math.isnan(intension_angular_height)) and (
                not math.isinf(intension_angular_height)):
            self.__intension_angular_height = int(intension_angular_height)
            self.__intension_angular_height = 0 if self.__intension_angular_height < 0 else self.__intension_angular_height
            self.__intension_angular_height = 265 if self.__intension_angular_height > 265 else self.__intension_angular_height
        if self.__intension_angular_height < self.__column_mid:
            intension_image[self.__intension_angular_height:self.__column_mid,
                            230:370] = (100, 255, 100)
        else:
            intension_image[self.__column_mid:self.__intension_angular_height,
                            230:370] = (100, 255, 100)
        # draw
        self.__gui_image[140:405, 1140:1510] = intension_image

        #### draw cmd ####
        cmd_image = np.zeros(
            (265, 370, 3), dtype=np.uint8) + np.asarray([50, 0, 0],
                                                        dtype=np.uint8)
        # linear
        cmd_linear_height = 265.0 * (1.0 - self.__cmd_data[CMD_LINEAR_X])
        if (not math.isnan(cmd_linear_height)) and (
                not math.isinf(cmd_linear_height)):
            self.__cmd_linear_height = int(cmd_linear_height)
            self.__cmd_linear_height = 0 if self.__cmd_linear_height < 0 else self.__cmd_linear_height
            self.__cmd_linear_height = 265 if self.__cmd_linear_height > 265 else self.__cmd_linear_height
        cmd_image[self.__cmd_linear_height:265, 0:140] = (100, 100, 255)
        # angular
        cmd_angular_height = 265.0 * (0.5 -
                                      0.5 * self.__cmd_data[CMD_ANGULAR_Z])
        if (not math.isnan(cmd_angular_height)) and (
                not math.isinf(cmd_angular_height)):
            self.__cmd_angular_height = int(cmd_angular_height)
            self.__cmd_angular_height = 0 if self.__cmd_angular_height < 0 else self.__cmd_angular_height
            self.__cmd_angular_height = 265 if self.__cmd_angular_height > 265 else self.__cmd_angular_height
        if self.__cmd_angular_height < self.__column_mid:
            cmd_image[self.__cmd_angular_height:self.__column_mid,
                      230:370] = (100, 100, 255)
        else:
            cmd_image[self.__column_mid:self.__cmd_angular_height,
                      230:370] = (100, 100, 255)
        # draw
        self.__gui_image[545:810, 1140:1510] = cmd_image

    def __show_gui_image(self):
        self.__draw_gui_image()
        cv2.imshow("robot real gui", self.__gui_image)
        cv2.waitKey(1)

    def run(self):
        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            self.__cal_intension()
            self.__pub_intension()
            self.__draw_gui_image()
            self.__show_gui_image()
            rate.sleep()


def main():
    real_gui = RealGui()
    real_gui.run()


if __name__ == '__main__':
    main()