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

#### index ####
POSE_PX = 0
POSE_PY = 1
POSE_YAW = 2


class EnvAssist(object):

    def __init__(self, index):
        #### Config ####
        self.__index = index

        #### position table ####
        self.__init_pose = [[-10.50, 0.0, 0.00], [10.50, 0.0, np.pi]]
        self.__goal_point = [[[10.00, 0.00], [-10.00, 0.00]],
                             [[-10.00, 0.00], [10.00, 0.00]]]
        self.__goal_index = 0

    ########################################
    #### get handle
    ########################################
    def get_init_pose(self):
        init = self.__init_pose[self.__index]
        return init

    def get_goal_point(self):
        goal = self.__goal_point[self.__index][self.__goal_index]
        self.__goal_index = (self.__goal_index + 1) % 2
        return goal
