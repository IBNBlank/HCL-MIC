#!/usr/bin/env python3
# -*- coding:utf-8 -*-
################################################################
# Copyright 2023 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2023-06-28
################################################################

import math
import rospy
from geometry_msgs.msg import Twist

CMD_LINEAR_X = 0
CMD_ANGULAR_Z = 1


class FrequencyMultiplier:

    def __init__(self):
        rospy.init_node(f"robot_frequency_multiplier", anonymous=None)

        # Variable
        self.__cmd_data = [0.0, 0.0]

        # Publisher
        self.__cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=5)

        # Subscriber
        self.__cmd_sub = rospy.Subscriber("/rl_cmd_vel",
                                          Twist,
                                          self.__cmd_callback,
                                          queue_size=5)

    def __cmd_callback(self, msg):
        if not math.isnan(msg.linear.x) or not math.isinf(msg.linear.x):
            self.__cmd_data[CMD_LINEAR_X] = msg.linear.x
        if not math.isnan(msg.angular.z) or not math.isinf(msg.angular.z):
            self.__cmd_data[CMD_ANGULAR_Z] = msg.angular.z

    def __pub_cmd(self):
        cmd_msg = Twist()
        cmd_msg.linear.x = self.__cmd_data[CMD_LINEAR_X]
        cmd_msg.angular.z = self.__cmd_data[CMD_ANGULAR_Z]
        self.__cmd_pub.publish(cmd_msg)

    def run(self):
        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            self.__pub_cmd()
            rate.sleep()


def main():
    frequency_multiplier = FrequencyMultiplier()
    frequency_multiplier.run()


if __name__ == '__main__':
    main()