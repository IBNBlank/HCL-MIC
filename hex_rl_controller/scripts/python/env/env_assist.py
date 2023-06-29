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

    def __init__(self, index, stage_type, reward_gamma=0.0):
        #### Config ####
        self.__index = index
        self.__stage_type = stage_type
        self.__reward_gamma = reward_gamma

        #### position table ####
        if self.__stage_type == "stage_test":
            self.__init_pose = [[-10.50, 0.0, 0.00], [10.50, 0.0, np.pi]]
            self.__goal_point = [[[-10.00, 0.00], [10.00, 0.00]],
                                 [[10.00, 0.00], [-10.00, 0.00]]]
            self.__goal_index = 0
        elif self.__stage_type == "stage_0":
            self.__init_pose = [[-26.0, 26.0, 0.0], [0.0, 26.0, 0.0],
                                [26.0, 26.0, 0.0], [-26.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0], [26.0, 0.0, 0.0],
                                [-26.0, -26.0, 0.0], [0.0, -26.0, 0.0],
                                [26.0, -26.0, 0.0]]
            self.__goal_point = [[9.0, 0.0], [7.281152949, 5.290067271],
                                 [2.781152949, 8.559508647],
                                 [-2.781152949, 8.559508647],
                                 [-7.281152949, 5.290067271], [-9.0, 0.0],
                                 [-7.281152949, -5.290067271],
                                 [-2.781152949, -8.559508647],
                                 [2.781152949, -8.559508647],
                                 [7.281152949, -5.290067271]]
        elif self.__stage_type == "stage_1":
            self.__init_pose = [[6.0, 0.0, 0.0],
                                [5.795554958, 1.552914271, 0.0],
                                [5.196152423, 3.0, 0.0],
                                [4.242640687, 4.242640687, 0.0],
                                [3.0, 5.196152423, 0.0],
                                [1.552914271, 5.795554958,
                                 0.0], [0.0, 6.0, 0.0],
                                [-1.552914271, 5.795554958, 0.0],
                                [-3.0, 5.196152423, 0.0],
                                [-4.242640687, 4.242640687, 0.0],
                                [-5.196152423, 3.0, 0.0],
                                [-5.795554958, 1.552914271, 0.0],
                                [-6.0, 0.0, 0.0],
                                [-5.795554958, -1.552914271, 0.0],
                                [-5.196152423, -3.0, 0.0],
                                [-4.242640687, -4.242640687, 0.0],
                                [-3.0, -5.196152423, 0.0],
                                [-1.552914271, -5.795554958, 0.0],
                                [0.0, -6.0, 0.0],
                                [1.552914271, -5.795554958, 0.0],
                                [3.0, -5.196152423, 0.0],
                                [4.242640687, -4.242640687, 0.0],
                                [5.196152423, -3.0, 0.0],
                                [5.795554958, -1.552914271, 0.0]]
            self.__goal_point = [[-6.0, 0.0, 0.0],
                                 [-5.795554958, -1.552914271, 0.0],
                                 [-5.196152423, -3.0, 0.0],
                                 [-4.242640687, -4.242640687, 0.0],
                                 [-3.0, -5.196152423, 0.0],
                                 [-1.552914271, -5.795554958, 0.0],
                                 [0.0, -6.0, 0.0],
                                 [1.552914271, -5.795554958, 0.0],
                                 [3.0, -5.196152423, 0.0],
                                 [4.242640687, -4.242640687, 0.0],
                                 [5.196152423, -3.0, 0.0],
                                 [5.795554958, -1.552914271, 0.0],
                                 [6.0, 0.0,
                                  0.0], [5.795554958, 1.552914271, 0.0],
                                 [5.196152423, 3.0, 0.0],
                                 [4.242640687, 4.242640687, 0.0],
                                 [3.0, 5.196152423, 0.0],
                                 [1.552914271, 5.795554958, 0.0],
                                 [0.0, 6.0, 0.0],
                                 [-1.552914271, 5.795554958, 0.0],
                                 [-3.0, 5.196152423, 0.0],
                                 [-4.242640687, 4.242640687, 0.0],
                                 [-5.196152423, 3.0, 0.0],
                                 [-5.795554958, 1.552914271, 0.0]]
            self.__origin_bias = [[8.0, 8.0, 0.0], [-8.0, 8.0, 0.0],
                                  [-8.0, -8.0, 0.0], [8.0, -8.0, 0.0]]
        elif self.__stage_type == "stage_2":
            self.__init_pose = [[-22.50, 22.50, 0.00], [-12.00, 22.50, 0.00],
                                [-1.50, 22.50, 0.00], [-22.50, 12.00, 0.00],
                                [-1.50, 12.00, 0.00], [-22.50, 1.50, 0.00],
                                [-12.00, 1.50, 0.00], [-1.50, 1.50, 0.00],
                                [19.425, 19.425, 0.00], [19.425, 4.56, 0.00],
                                [4.56, 19.425, 0.00], [4.56, 4.56, 0.00],
                                [22.50, 12.00, 0.00], [1.50, 12.00, 0.00],
                                [12.00, 1.50, 0.00], [12.00, 22.50, 0.00],
                                [-1.50, -15.75, 0.00], [-8.25, -8.25, 0.00],
                                [-8.25, -22.50, 0.00], [-15.75, -1.50, 0.00],
                                [-15.75, -15.75, 0.00], [-22.50, -8.25, 0.00],
                                [22.50, -22.50, 0.00], [12.00, -22.50, 0.00],
                                [1.50, -22.50, 0.00], [22.50, -12.00, 0.00],
                                [1.50, -12.00, 0.00], [22.50, -1.50, 0.00],
                                [12.00, -1.50, 0.00], [1.50, -1.50, 0.00]]
            self.__goal_point = [[-21.00, 21.00], [-12.00,
                                                   21.00], [-3.00, 21.00],
                                 [-21.00, 12.00],
                                 [-3.00, 12.00], [-21.00, 3.00], [-12.00, 3.00],
                                 [-3.00, 3.00], [18.36, 18.36], [18.36, 5.64],
                                 [5.64, 18.36], [5.64, 5.64], [21.00, 12.00],
                                 [3.00, 12.00], [12.00, 3.00], [12.00, 21.00],
                                 [-3.00, -15.75], [-8.25,
                                                   -8.25], [-8.25, -21.00],
                                 [-15.75, -3.00], [-15.75, -15.75],
                                 [-21.00, -8.25], [21.00, -21.00],
                                 [12.00, -21.00], [3.00,
                                                   -21.00], [21.00, -12.00],
                                 [3.00, -12.00], [21.00, -3.00], [12.00, -3.00],
                                 [3.00, -3.00], [-12.00,
                                                 12.00], [12.00, -12.00]]
            self.__goal_child = [[1, 3], [0, 2, 30], [1, 4], [0, 5, 30],
                                 [2, 7, 30], [3, 6], [5, 7, 30], [4, 6],
                                 [12, 15], [12, 14], [13, 15], [13, 14], [8, 9],
                                 [10, 11], [9, 11], [8, 10], [16, 17],
                                 [15, 18, 19], [15, 19], [16, 20], [16, 17, 20],
                                 [18, 19], [22, 24], [21, 23, 31], [22, 25],
                                 [21, 26], [23, 28, 31], [24, 27], [26, 28, 31],
                                 [25, 27], [1, 3, 4, 6], [22, 25, 27]]
            self.__goal_index = self.__index

        #### reward scaling ####
        self.__running_reward = 0.0
        self.__running_count = 0
        self.__running_mean = 0.0
        self.__running_total_error = 0.0

    ########################################
    #### get handle
    ########################################
    def get_init_pose(self):
        init = [0.0, 0.0, 0.0]

        if self.__stage_type == "stage_test":
            init = self.__init_pose[self.__index]
        elif self.__stage_type == "stage_0":
            init = self.__init_pose[self.__index]
            init[POSE_YAW] = np.random.uniform(-np.pi, np.pi)
        elif self.__stage_type == "stage_1":
            origin_index = self.__index // 4
            origin_bias = self.__origin_bias[origin_index]
            index_bias = np.random.randint(0, 3)
            init_local = self.__init_pose[
                (self.__index % 4 + origin_index * 2) % 8 * 3 + index_bias]
            init[POSE_PX] = init_local[POSE_PX] + origin_bias[POSE_PX]
            init[POSE_PY] = init_local[POSE_PY] + origin_bias[POSE_PY]
            init[POSE_YAW] = np.random.uniform(-np.pi, np.pi)
        elif self.__stage_type == "stage_2":
            init = self.__init_pose[self.__index]
            init[POSE_YAW] = np.random.uniform(-np.pi, np.pi)

        return init

    def get_goal_point(self):
        goal = [0.0, 0.0]

        if self.__stage_type == "stage_test":
            goal = self.__goal_point[self.__index][self.__goal_index]
            self.__goal_index = (self.__goal_index + 1) % 2
        elif self.__stage_type == "stage_0":
            goal = self.__random_goal()
        elif self.__stage_type == "stage_1":
            origin_index = self.__index // 4
            origin_bias = self.__origin_bias[origin_index]
            index_bias = np.random.randint(0, 3)
            goal_local = self.__goal_point[
                (self.__index % 4 + origin_index * 2) % 8 * 3 + index_bias]
            goal[POSE_PX] = goal_local[POSE_PX] + origin_bias[POSE_PX]
            goal[POSE_PY] = goal_local[POSE_PY] + origin_bias[POSE_PY]
        elif self.__stage_type == "stage_2":
            goal = self.__random_goal()

        return goal

    ########################################
    #### random handle
    ########################################
    def __random_goal(self):
        goal = [0.0, 0.0]

        if self.__stage_type == "stage_0":
            goal_bias = self.__goal_point[np.random.randint(
                0, len(self.__goal_point))]
            goal[POSE_PX] = goal_bias[POSE_PX] + self.__init_pose[
                self.__index][POSE_PX]
            goal[POSE_PY] = goal_bias[POSE_PY] + self.__init_pose[
                self.__index][POSE_PY]
        elif self.__stage_type == "stage_2":
            new_index = self.__goal_child[self.__goal_index][np.random.randint(
                0, len(self.__goal_child[self.__goal_index]))]
            self.__goal_index = new_index
            goal = self.__goal_point[new_index]

        return goal

    ########################################
    #### reward scaling handle
    ########################################
    def reward_scaling(self, raw_reward):
        self.__running_reward = self.__reward_gamma * self.__running_reward + raw_reward

        self.__running_count += 1
        if self.__running_count == 1:
            self.__running_mean = self.__running_reward
            running_std = self.__running_reward
        else:
            old_mean = self.__running_mean
            self.__running_mean = old_mean + (self.__running_reward -
                                              old_mean) / self.__running_count
            self.__running_total_error += (self.__running_reward - old_mean) * (
                self.__running_reward - self.__running_mean)
            running_std = np.sqrt(self.__running_total_error /
                                  self.__running_count)

        reward = raw_reward / (running_std + 1e-8)
        return reward

    def reset_reward(self):
        self.__running_reward = 0.0