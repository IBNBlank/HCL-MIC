#!/usr/bin/env python3
# -*- coding:utf-8 -*-
################################################################
# Copyright 2023 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2023-04-23
################################################################

import numpy as np
import pandas as pd
import os
import json


class Memory(object):

    def __init__(self):
        # state
        self.__lasers = []
        self.__velocitys = []
        self.__intentions = []
        # action
        self.__actions = []
        # env
        self.__rewards = []
        self.__terminateds = []
        # training helper
        self.__value = []
        self.__logprobs = []

    ########################################
    #### get handle
    ########################################
    def get_lasers(self):
        return self.__lasers

    def get_velocitys(self):
        return self.__velocitys

    def get_intentions(self):
        return self.__intentions

    def get_actions(self):
        return self.__actions

    def get_rewards(self):
        return self.__rewards

    def get_terminateds(self):
        return self.__terminateds

    def get_values(self):
        return self.__value

    def get_logprobs(self):
        return self.__logprobs

    ########################################
    #### clear handle
    ########################################
    def clear_memory(self):
        self.__lasers = []
        self.__velocitys = []
        self.__intentions = []
        self.__actions = []
        self.__rewards = []
        self.__terminateds = []
        self.__value = []
        self.__logprobs = []

    ########################################
    #### add handle
    ########################################
    def add_observation(self, observation):
        laser_list, velocity_list, intention_list = [], [], []

        for item in observation:
            laser_list.append(item[0])
            velocity_list.append(item[1])
            intention_list.append(item[2])

        self.__lasers.append(laser_list)
        self.__velocitys.append(velocity_list)
        self.__intentions.append(intention_list)

    def add_action(self, action):
        self.__actions.append(action)

    def add_reward(self, reward):
        self.__rewards.append(reward)

    def add_terminated(self, terminated):
        self.__terminateds.append(terminated)

    def add_value(self, value):
        self.__value.append(value)

    def add_logprob(self, logprob):
        self.__logprobs.append(logprob)

    ########################################
    #### io handle
    ########################################
    def save_data(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

        changed_lasers = []
        changed_velocities = []
        changed_intentions = []
        for agent_items in self.__lasers:
            for item in agent_items:
                changed_lasers.append(str(item.tolist()))
        for agent_items in self.__velocitys:
            for item in agent_items:
                changed_velocities.append(str(item.tolist()))
        for agent_items in self.__intentions:
            for item in agent_items:
                changed_intentions.append(str(item.tolist()))

        data_frame = pd.DataFrame({
            "lasers": changed_lasers,
            "velocities": changed_velocities,
            "intentions": changed_intentions
        })
        if not os.path.exists(f"{path}/pretrain.csv"):
            data_frame.to_csv(f"{path}/pretrain.csv",
                              mode="w",
                              header=True,
                              index=False)
        else:
            data_frame.to_csv(f"{path}/pretrain.csv",
                              mode="a",
                              header=False,
                              index=False)

    def load_data(self, path):
        if not os.path.exists(path):
            print("#### no train data ####")
            return False

        chunker = pd.read_csv(path, chunksize=512)
        for data_frame in chunker:
            str_lasers = data_frame["lasers"]
            str_velocitys = data_frame["velocities"]
            str_intentions = data_frame["intentions"]

            for item in str_lasers:
                self.__lasers.append(np.array(json.loads(item)))
            for item in str_velocitys:
                self.__velocitys.append(np.array(json.loads(item)))
            for item in str_intentions:
                self.__intentions.append(np.array(json.loads(item)))

        print(f"load {path} finish")