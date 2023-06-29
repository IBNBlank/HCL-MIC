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
PRETRAIN_PATH = f"{ROS_PKG_PATH}/pretrain_data"

sys.path.append(f"{ROS_PKG_PATH}/scripts/python")
from env.stage_env import StageEnv
from model.ppo import Ppo
from model.memory import Memory

STAGE_TYPE = "stage_0"
ENV_TYPE = "polar"
NUM_STEP = 256
NUM_AGENT = 9

NUM_MAX_UPDATE = 2000
NUM_MINI_BATCH = 256
NUM_EPOCH = 8

GAMMA = 0.99
LAMDA = 0.95
COEFF_ENTROPY = 1e-2
CLIP_VALUE = 0.2

ENCODER_LR = 3e-5
STATE_LR = 3e-5
ACTOR_LR = 3e-4
CRITIC_LR = 3e-4

SEED = 2056


class Stage0():

    def __init__(self,
                 comm,
                 reward_gamma=0.0,
                 encoder_file=None,
                 state_file=None,
                 actor_file=None,
                 critic_file=None,
                 collect_data=False,
                 freeze_state=False,
                 freeze_encoder=False):
        #### config ####
        self.__comm = comm
        self.__rank = comm.Get_rank()

        #### logger ####
        # path
        log_dir = f"{LOG_PATH}/{STAGE_TYPE}"
        log_out = f"{log_dir}/out.log"
        log_cal = f"{log_dir}/cal.log"
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
        file_handler_cal = logging.FileHandler(log_cal, mode='a')
        file_handler_cal.setLevel(log_level)
        file_handler_cal.setFormatter(log_formatter)
        # logger
        self.__logger_out = logging.getLogger('logger_out')
        self.__logger_out.setLevel(log_level)
        self.__logger_out.addHandler(file_handler_out)
        self.__logger_out.addHandler(stream_handler_out)
        self.__logger_cal = logging.getLogger('logger_cal')
        self.__logger_cal.setLevel(log_level)
        self.__logger_cal.addHandler(file_handler_cal)

        #### stage env ####
        self.__stage_env = StageEnv(index=self.__rank,
                                    stage_type=STAGE_TYPE,
                                    env_type=ENV_TYPE,
                                    reward_gamma=reward_gamma,
                                    max_step=150,
                                    beam_num=256)

        if self.__rank == 0:
            #### ppo ####
            self.__ppo = Ppo(stage_type=STAGE_TYPE,
                             env_type=ENV_TYPE,
                             num_step=NUM_STEP,
                             num_agent=NUM_AGENT,
                             num_max_update=NUM_MAX_UPDATE,
                             num_mini_batch=NUM_MINI_BATCH,
                             num_epoch=NUM_EPOCH,
                             gamma=GAMMA,
                             lamda=LAMDA,
                             coeff_entropy=COEFF_ENTROPY,
                             clip_value=CLIP_VALUE,
                             encoder_lr=ENCODER_LR,
                             state_lr=STATE_LR,
                             actor_lr=ACTOR_LR,
                             critic_lr=CRITIC_LR,
                             use_lr_decay=True,
                             seed=SEED,
                             encoder_file=encoder_file,
                             state_file=state_file,
                             actor_file=actor_file,
                             critic_file=critic_file,
                             freeze_state=freeze_state,
                             freeze_encoder=freeze_encoder)

            #### memory ####
            self.__collect_data = collect_data
            self.__memory = Memory()

        #### global variable ####
        self.__global_step = 0

    def run(self):
        #### initial before training ####
        episode_num = 1
        finish = False
        # reset world
        if self.__rank == 0:
            self.__stage_env.reset_world()
            self.__logger_out.info("#### start ####")

        #### training ####
        while not finish and not self.__stage_env.is_shutdown():
            self.__single_episode(episode_num)
            episode_num += 1
            if self.__rank == 0:
                finish = self.__ppo.is_finish()

        if self.__rank == 0:
            self.__ppo.save_policy(final=True)
            os.system("rosnode kill --all")
            os.system("killall mpiexec")

    def __single_episode(self, episode_num):
        #### initial before episode ####
        # episode variable
        episode_terminal = False
        episode_reward = 0.0
        episode_step = 1

        # reset agent
        action_list = None
        self.__stage_env.reset_agent()
        self.__stage_env.reset_reward()

        #### episode training ####
        while not episode_terminal and not self.__stage_env.is_shutdown():
            # gather observations from all agents
            last_reward = None
            while last_reward is None:
                if episode_step == 1:
                    self.__stage_env.reset_state()
                last_reward, last_terminated, last_info, observation = self.__stage_env.collect_message(
                    episode_step)
                self.__stage_env.sleep(0.001)

            # execute action
            observation_list = self.__comm.gather(observation, root=0)
            if self.__rank == 0:
                action_list, logprob_list, value_list = self.__ppo.choose_action(
                    observation_list)
            action = self.__comm.scatter(action_list, root=0)
            self.__stage_env.execute_action(action)

            # collect training data
            last_reward_list = self.__comm.gather(last_reward, root=0)
            last_terminated_list = self.__comm.gather(last_terminated, root=0)
            if self.__rank == 0:
                if self.__global_step != 0:
                    self.__memory.add_reward(last_reward_list)
                    self.__memory.add_terminated(last_terminated_list)

                    if not self.__global_step % NUM_STEP:
                        self.__stage_env.pause_env(True)
                        
                        # update
                        self.__memory.add_value(value_list)
                        self.__ppo.update(memory=self.__memory)

                        if self.__collect_data:
                            self.__memory.save_data(
                                f"{PRETRAIN_PATH}/{STAGE_TYPE}")
                        self.__memory.clear_memory()
                        self.__logger_out.info(
                            f"{STAGE_TYPE} update {self.__ppo.get_update_steps()}"
                        )
                        
                        self.__stage_env.pause_env(False)

                self.__memory.add_observation(observation_list)
                self.__memory.add_action(action_list)
                self.__memory.add_value(value_list)
                self.__memory.add_logprob(logprob_list)

            # finish process
            episode_terminal = last_terminated
            episode_reward += last_reward
            episode_step += 1
            self.__global_step += 1

        #### finish episode ####
        self.__logger_out.info(
            f"Env {self.__rank}, Episode {episode_num}, step {episode_step - 1}, Reward {episode_reward}, {last_info}"
        )
        self.__logger_cal.info(episode_reward)


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
    # policy directory
    encoder_dir = f"{POLICY_PATH}/{STAGE_TYPE}/encoder/"
    state_dir = f"{POLICY_PATH}/{STAGE_TYPE}/state/"
    actor_dir = f"{POLICY_PATH}/{STAGE_TYPE}/actor/"
    critic_dir = f"{POLICY_PATH}/{STAGE_TYPE}/critic/"
    if not os.path.exists(encoder_dir):
        os.makedirs(encoder_dir)
    if not os.path.exists(state_dir):
        os.makedirs(state_dir)
    if not os.path.exists(actor_dir):
        os.makedirs(actor_dir)
    if not os.path.exists(critic_dir):
        os.makedirs(critic_dir)
    # pretrained parameters path
    encoder_pretrained_file = None
    state_pretrained_file = None
    actor_pretrained_file = None
    critic_pretrained_file = None
    if mpi_comm.Get_rank() == 0:
        encoder_pretrained_file = get_pretrained_file([
            "stage_0/encoder/encoder.pth", "stage_pretrain/encoder/encoder.pth"
        ])
        state_pretrained_file = get_pretrained_file(
            ["stage_0/state/state.pth", "stage_pretrain/state/state.pth"])
        actor_pretrained_file = get_pretrained_file(["stage_0/actor/actor.pth"])
        critic_pretrained_file = get_pretrained_file(
            ["stage_0/critic/critic.pth"])

    #### stage0 class ####
    stage0 = Stage0(comm=mpi_comm,
                    reward_gamma=0.0,
                    encoder_file=encoder_pretrained_file,
                    state_file=state_pretrained_file,
                    actor_file=actor_pretrained_file,
                    critic_file=critic_pretrained_file,
                    collect_data=False,
                    freeze_state=False,
                    freeze_encoder=True)

    try:
        stage0.run()
    except KeyboardInterrupt:
        pass