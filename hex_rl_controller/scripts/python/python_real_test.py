#!/usr/bin/env python3
# -*- coding:utf-8 -*-
################################################################
# Copyright 2023 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2023-04-24
################################################################

import os
import logging
import sys
import time
from rospkg import RosPack

ROS_PKG_PATH = RosPack().get_path('hex_rl_controller')
LOG_PATH = f"{ROS_PKG_PATH}/log"
POLICY_PATH = f"{ROS_PKG_PATH}/policy"

sys.path.append(f"{ROS_PKG_PATH}/scripts/python")
from real.real_env import RealEnv
from model.ppo import Ppo

STAGE_TYPE = "real_test"
ENV_TYPE = "polar"
NUM_AGENT = 1


class RealTest():

    def __init__(self,
                 encoder_file=None,
                 state_file=None,
                 actor_file=None,
                 critic_file=None):
        #### logger ####
        # path
        log_dir = f"{LOG_PATH}/{STAGE_TYPE}"
        log_out = f"{log_dir}/out.log"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        # handler
        log_level = logging.INFO
        log_formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s")
        file_handler_out = logging.FileHandler(log_out, mode='a')
        file_handler_out.setLevel(log_level)
        file_handler_out.setFormatter(log_formatter)
        stream_handler_out = logging.StreamHandler(sys.stdout)
        stream_handler_out.setLevel(log_level)
        stream_handler_out.setFormatter(log_formatter)
        # logger
        self.__logger_out = logging.getLogger('logger_out')
        self.__logger_out.setLevel(log_level)
        self.__logger_out.addHandler(file_handler_out)
        self.__logger_out.addHandler(stream_handler_out)

        #### real env ####
        self.__real_env = RealEnv(env_type=ENV_TYPE, beam_num=256)

        #### ppo handle ####
        self.__ppo = Ppo(stage_type=STAGE_TYPE,
                         env_type=ENV_TYPE,
                         encoder_file=encoder_file,
                         state_file=state_file,
                         actor_file=actor_file,
                         critic_file=critic_file)

    def run(self):
        #### test ####
        while not self.__real_env.is_shutdown():
            # gather observation
            observation = None
            while observation is None:
                observation = self.__real_env.collect_message()
                self.__real_env.sleep(0.001)

            # execute action
            print("run start")
            start_time = time.clock()
            observation_list = [observation]
            action_list = self.__ppo.evaluate(observation_list)
            action = action_list[0]
            self.__real_env.execute_action(action)
            print(f"run finish: {(time.clock() - start_time) * 1000.0}ms")

        #### test summary ####
        os.system("rosnode kill --all")


def get_pretrained_file(file_list):
    file_name = None

    for item in file_list:
        if os.path.exists(f"{POLICY_PATH}/{item}"):
            file_name = item
            break

    if not file_name is None:
        print(f"#### Loading {file_name} ####")
    else:
        print(f"#### No Pretrained file ####")

    return file_name


if __name__ == '__main__':
    #### policy ####
    # pretrained parameters path
    encoder_pretrained_file = None
    state_pretrained_file = None
    actor_pretrained_file = None
    critic_pretrained_file = None

    encoder_pretrained_file = get_pretrained_file([
        "stage_2/encoder/encoder.pth", "stage_1/encoder/encoder.pth",
        "stage_0/encoder/encoder.pth", "stage_pretrain/encoder/encoder.pth"
    ])
    state_pretrained_file = get_pretrained_file([
        "stage_2/state/state.pth", "stage_1/state/state.pth",
        "stage_0/state/state.pth", "stage_pretrain/state/state.pth"
    ])
    actor_pretrained_file = get_pretrained_file([
        "stage_2/actor/actor.pth", "stage_1/actor/actor.pth",
        "stage_0/actor/actor.pth"
    ])
    critic_pretrained_file = get_pretrained_file([
        "stage_2/critic/critic.pth", "stage_1/critic/critic.pth",
        "stage_0/critic/critic.pth"
    ])

    if (encoder_pretrained_file is None) or (state_pretrained_file is None) or (
            actor_pretrained_file is None) or (critic_pretrained_file is None):
        exit()

    #### real test class ####
    real_test = RealTest(encoder_file=encoder_pretrained_file,
                         state_file=state_pretrained_file,
                         actor_file=actor_pretrained_file,
                         critic_file=critic_pretrained_file)

    try:
        real_test.run()
    except KeyboardInterrupt:
        pass
