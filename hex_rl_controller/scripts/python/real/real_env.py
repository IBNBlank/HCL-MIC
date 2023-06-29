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
from real.data_interface import DataInterface

#### param ####
MAX_RANGE = 8.0
MAX_LINEAR = 1.2
MAX_ANGULAR = 1.2

DELTA_T = 0.1
MAX_LINEAR_ACCELERATION = 2.0
MAX_ANGULAR_ACCELERATION = 1.0 * np.pi
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


class RealEnv(object):

    def __init__(self, env_type, beam_num=512):
        #### Config ####
        self.__env_type = env_type

        #### Variable ####
        # sensor variable
        self.__laser = None
        self.__odom = None
        self.__intention = None
        # work variable
        self.__laser_obs = None
        # process variable
        self.__beam_num = beam_num

        #### Ros Interface ####
        self.__data_interface = DataInterface()

        # init
        while not self.__get_current_state():
            self.__data_interface.sleep(0.001)

    ########################################
    #### message handle
    ########################################
    def collect_message(self):
        if self.__get_current_state():
            observation = self.__observation()
            return observation
        else:
            return None

    def __get_current_state(self):
        ready = True
        ready = ready and self.__data_interface.have_laser()
        ready = ready and self.__data_interface.have_odom()
        ready = ready and self.__data_interface.have_intention()

        if ready:
            self.__laser = self.__get_transformed_laser()
            self.__odom = self.__data_interface.get_odom()
            self.__intention = self.__data_interface.get_intention()

            self.__data_interface.clear_laser()
            self.__data_interface.clear_odom()
            self.__data_interface.clear_intention()

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

    def __observation(self):
        # laser
        if self.__laser_obs is None:
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
        intention = [
            self.__intention[INTENTION_VX] / MAX_LINEAR - 0.5,
            0.5 * self.__intention[INTENTION_YAW] / MAX_ANGULAR
        ]

        return [
            np.array(self.__laser_obs).astype(float),
            np.array(velocity).astype(float),
            np.array(intention).astype(float)
        ]

    ########################################
    #### ros handle
    ########################################
    def execute_action(self, action):
        real_action = [
            action[ACTION_VX] * MAX_LINEAR,
            2.0 * (action[ACTION_VYAW] - 0.5) * MAX_ANGULAR
        ]
            
        if self.__intention[INTENTION_VX] > 0.5:
            final_action = real_action
        else:
            final_action = self.__intention
         
        if final_action[ACTION_VX] > self.__odom[ODOM_VX] + MAX_DELTA_LINEAR:
            final_action[ACTION_VX] = self.__odom[ODOM_VX] + MAX_DELTA_LINEAR
        elif final_action[ACTION_VX] < self.__odom[ODOM_VX] - MAX_DELTA_LINEAR:
            final_action[ACTION_VX] = self.__odom[ODOM_VX] - MAX_DELTA_LINEAR
        if final_action[ACTION_VYAW] > self.__odom[ODOM_VYAW] + MAX_DELTA_ANGULAR:
            final_action[
                ACTION_VYAW] = self.__odom[ODOM_VYAW] + MAX_DELTA_ANGULAR
        elif final_action[
                ACTION_VYAW] < self.__odom[ODOM_VYAW] - MAX_DELTA_ANGULAR:
            final_action[
                ACTION_VYAW] = self.__odom[ODOM_VYAW] - MAX_DELTA_ANGULAR 
            
        self.__data_interface.pub_cmd_vel(final_action)

    def sleep(self, duration):
        self.__data_interface.sleep(duration)

    def is_shutdown(self):
        return self.__data_interface.is_shutdown()
