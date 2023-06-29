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
from mpi4py import MPI
from rospkg import RosPack

ROS_PKG_PATH = RosPack().get_path('hex_rl_controller')
LOG_PATH = f"{ROS_PKG_PATH}/log"
POLICY_PATH = f"{ROS_PKG_PATH}/policy"

sys.path.append(f"{ROS_PKG_PATH}/scripts/python")
from gazebo.gazebo_env import GazeboEnv
from model.ppo import Ppo
from model.memory import Memory

STAGE_TYPE = "gazebo_test"
ENV_TYPE = "polar"
NUM_AGENT = 2


class GazeboTest():

    def __init__(self,
                 comm,
                 encoder_file=None,
                 state_file=None,
                 actor_file=None,
                 critic_file=None):
        #### config ####
        self.__comm = comm
        self.__rank = comm.Get_rank()

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

        #### gazebo env ####
        self.__gazebo_env = GazeboEnv(index=self.__rank,
                                      env_type=ENV_TYPE,
                                      max_step=2000,
                                      beam_num=256)

        if self.__rank == 0:
            #### ppo handle ####
            self.__ppo = Ppo(stage_type=STAGE_TYPE,
                             env_type=ENV_TYPE,
                             encoder_file=encoder_file,
                             state_file=state_file,
                             actor_file=actor_file,
                             critic_file=critic_file)

    def run(self):
        #### initial before test ####
        # test variable
        group_terminal = False
        step = 1

        # reset agent
        action_list = None

        #### test ####
        while not group_terminal and not self.__gazebo_env.is_shutdown():
            # gather observations from all agents
            last_terminated = None
            while last_terminated is None:
                if step == 1:
                    self.__gazebo_env.reset_state()
                last_terminated, last_info, observation = self.__gazebo_env.collect_message(
                    step)
                self.__gazebo_env.sleep(0.001)

            # execute action
            observation_list = self.__comm.gather(observation, root=0)
            if self.__rank == 0:
                action_list = self.__ppo.evaluate(observation_list)
            action = self.__comm.scatter(action_list, root=0)
            self.__gazebo_env.execute_action(action)

            # collect test data
            last_terminated_list = self.__comm.gather(last_terminated, root=0)
            last_info_list = self.__comm.gather(last_info, root=0)

            # finish process
            last_terminated_list = self.__comm.bcast(last_terminated_list,
                                                     root=0)
            group_terminal = self.__get_group_terminal(last_terminated_list)
            step += 1

        #### test summary ####
        if self.__rank == 0:
            for rank in range(NUM_AGENT):
                self.__logger_out.info(
                    f"rank{str(rank).zfill(2)}:  {last_info_list[rank]}")
            os.system("rosnode kill --all")
            os.system("killall mpiexec")

    def __get_group_terminal(self, terminal_list):
        group_terminal = True
        for item in terminal_list:
            if item < 2:
                group_terminal = False
                break
        return group_terminal


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
    #### mpi4py ####
    mpi_comm = MPI.COMM_WORLD

    #### policy ####
    # pretrained parameters path
    encoder_pretrained_file = None
    state_pretrained_file = None
    actor_pretrained_file = None
    critic_pretrained_file = None
    if mpi_comm.Get_rank() == 0:
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

        if (encoder_pretrained_file is
                None) or (state_pretrained_file is
                          None) or (actor_pretrained_file is
                                    None) or (critic_pretrained_file is None):
            exit()

    #### gazebo test class ####
    gazebo_test = GazeboTest(comm=mpi_comm,
                             encoder_file=encoder_pretrained_file,
                             state_file=state_pretrained_file,
                             actor_file=actor_pretrained_file,
                             critic_file=critic_pretrained_file)

    try:
        gazebo_test.run()
    except KeyboardInterrupt:
        pass