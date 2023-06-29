#!/usr/bin/env python3
# -*- coding:utf-8 -*-
################################################################
# Copyright 2023 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2023-04-19
################################################################

import math
import numpy as np
import sys
from rospkg import RosPack

ROS_PKG_PATH = RosPack().get_path('hex_rl_controller')
sys.path.append(f"{ROS_PKG_PATH}/scripts/python")
from gazebo.env_assist import EnvAssist
from gazebo.data_interface import DataInterface

#### param ####
GOAL_SIZE = 1.0
MAX_RANGE = 8.0
MAX_LINEAR = 1.2
MAX_ANGULAR = 1.2

DELTA_T = 0.1
MAX_LINEAR_ACCELERATION = 1.5
MAX_ANGULAR_ACCELERATION = 0.5 * np.pi
MAX_DELTA_LINEAR = MAX_LINEAR_ACCELERATION * DELTA_T
MAX_DELTA_ANGULAR = MAX_ANGULAR_ACCELERATION * DELTA_T

#### index ####
POSE_PX = 0
POSE_PY = 1
POSE_YAW = 2
INTENTION_VX = 0
INTENTION_YAW = 1
TARGET_VX = 0
TARGET_YAW = 1
ODOM_PX = 0
ODOM_PY = 1
ODOM_YAW = 2
ODOM_VX = 3
ODOM_VYAW = 4
ACTION_VX = 0
ACTION_VYAW = 1


class GazeboEnv(object):

    def __init__(self,
                 index,
                 env_type,
                 max_step=150,
                 beam_num=512,
                 grid_size=None):
        #### Config ####
        self.__index = index
        self.__env_type = env_type
        self.__max_step = max_step

        #### Assist ####
        self.__env_assist = EnvAssist(self.__index)

        #### Variable ####
        # sensor variable
        self.__laser = None
        self.__odom = None
        # work variable
        self.__init_pose = [0.0, 0.0, 0.0]
        self.__goal = [0.0, 0.0]
        self.__target = [0.0, 0.0]
        self.__last_target = [0.0, 0.0]
        self.__laser_obs = None
        self.__terminated = 0
        self.__info = "Work"
        # process variable
        if self.__env_type == "polar":
            self.__beam_num = beam_num
        else:
            self.__grid_size = grid_size

        #### Ros Interface ####
        self.__data_interface = DataInterface(index)

        # init
        while not self.__get_current_state():
            self.__data_interface.sleep(0.001)
        self.__reset_goal()

    ########################################
    #### message handle
    ########################################
    def collect_message(self, step):
        if self.__get_current_state():
            observation = self.__observation(step)
            last_terminated, last_info = self.__calculate_message(step)
            return last_terminated, last_info, observation
        else:
            return None, None, None

    def __get_current_state(self):
        ready = True
        ready = ready and self.__data_interface.have_laser()
        ready = ready and self.__data_interface.have_odom()

        if ready:
            self.__laser = self.__get_transformed_laser()
            self.__odom = self.__data_interface.get_odom()

            self.__data_interface.clear_laser()
            self.__data_interface.clear_odom()

        return ready

    def __get_transformed_laser(self):
        if self.__env_type == "polar":
            return self.__polar_laser()
        else:
            return self.__grid_laser()

    def __polar_laser(self):
        raw_laser = self.__data_interface.get_laser()
        raw_laser[np.isnan(raw_laser)] = MAX_RANGE
        raw_laser[np.isinf(raw_laser)] = MAX_RANGE

        raw_beam_num = len(raw_laser)
        step = raw_beam_num / self.__beam_num

        sparse_laser = []
        for beam_index in range(self.__beam_num):
            min_range = MAX_RANGE
            beam_start = int(beam_index * step)
            for step_index in range(int(step)):
                raw_index = beam_start + step_index
                if raw_index < raw_beam_num and min_range > raw_laser[raw_index]:
                    min_range = raw_laser[raw_index]
            sparse_laser.append(min_range)
        sparse_laser = np.array(sparse_laser)

        # normalize
        return sparse_laser / MAX_RANGE - 0.5

    def __grid_laser(self):
        raw_laser = self.__data_interface.get_laser()
        return None

    def __observation(self, step):
        # laser
        if step == 1:
            self.__laser_obs = [self.__laser, self.__laser, self.__laser]
        else:
            self.__laser_obs.pop(0)
            self.__laser_obs.append(self.__laser)

        # velocity
        velocity = [
            self.__odom[ODOM_VX] / MAX_LINEAR - 0.5,
            0.5 * self.__odom[ODOM_VYAW] / MAX_ANGULAR
        ]

        # intention
        intention = [0.0, 0.0]
        self.__last_target = self.__target

        delta_x = self.__goal[POSE_PX] - self.__odom[ODOM_PX]
        delta_y = self.__goal[POSE_PY] - self.__odom[ODOM_PY]
        noisy_distance = np.sqrt(delta_x * delta_x + delta_y * delta_y)
        noisy_yaw = math.atan2(delta_y, delta_x)
        self.__target[TARGET_YAW] = noisy_yaw

        delta_yaw = noisy_yaw - self.__odom[ODOM_YAW]
        if delta_yaw > np.pi:
            delta_yaw -= 2.0 * np.pi
        elif delta_yaw < -np.pi:
            delta_yaw += 2.0 * np.pi

        if np.abs(delta_yaw) > 1.0:
            intention[INTENTION_YAW] = 0.5 * np.sign(delta_yaw)
            self.__target[TARGET_VX] = 0.0
        else:
            intention[INTENTION_YAW] = 0.5 * delta_yaw
            self.__target[TARGET_VX] = MAX_LINEAR * (1.0 - np.abs(delta_yaw))
            if self.__target[TARGET_VX] > noisy_distance:
                self.__target[TARGET_VX] = noisy_distance
        if step == 1:
            self.__last_target = self.__target

        intention[INTENTION_VX] = self.__target[TARGET_VX] / MAX_LINEAR - 0.5
        self.__data_interface.pub_intention(intention)

        return [
            np.array(self.__laser_obs).astype(float),
            np.array(velocity).astype(float),
            np.array(intention).astype(float)
        ]

    def __calculate_message(self, step):
        # reach terminal
        delta_x = self.__goal[POSE_PX] - self.__odom[ODOM_PX]
        delta_y = self.__goal[POSE_PY] - self.__odom[ODOM_PY]
        goal_distance = np.sqrt(delta_x * delta_x + delta_y * delta_y)
        if goal_distance < GOAL_SIZE and step > 2:
            self.__terminated += 1
            self.__info = "Reach Goal"
            self.__reset_goal()
        # timeout terminal
        elif step > self.__max_step:
            self.__terminated = 2
            self.__info = "Time out"

        return self.__terminated, self.__info

    ########################################
    #### reset handle
    ########################################
    def reset_state(self):
        self.__data_interface.reset_state(self.__init_pose)

    def __reset_goal(self):
        self.__goal = self.__env_assist.get_goal_point()
        self.__data_interface.pub_goal_point(self.__goal)

    ########################################
    #### ros handle
    ########################################
    def execute_action(self, action):
        real_action = [
            action[ACTION_VX] * MAX_LINEAR,
            2.0 * (action[ACTION_VYAW] - 0.5) * MAX_ANGULAR
        ]

        if real_action[ACTION_VX] > self.__odom[ODOM_VX] + MAX_DELTA_LINEAR:
            real_action[ACTION_VX] = self.__odom[ODOM_VX] + MAX_DELTA_LINEAR
        elif real_action[ACTION_VX] < self.__odom[ODOM_VX] - MAX_DELTA_LINEAR:
            real_action[ACTION_VX] = self.__odom[ODOM_VX] - MAX_DELTA_LINEAR
        if real_action[ACTION_VYAW] > self.__odom[ODOM_VYAW] + MAX_DELTA_ANGULAR:
            real_action[
                ACTION_VYAW] = self.__odom[ODOM_VYAW] + MAX_DELTA_ANGULAR
        elif real_action[
                ACTION_VYAW] < self.__odom[ODOM_VYAW] - MAX_DELTA_ANGULAR:
            real_action[
                ACTION_VYAW] = self.__odom[ODOM_VYAW] - MAX_DELTA_ANGULAR

        self.__data_interface.pub_cmd_vel(real_action)

    def sleep(self, duration):
        self.__data_interface.sleep(duration)

    def is_shutdown(self):
        return self.__data_interface.is_shutdown()
