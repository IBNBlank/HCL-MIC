#!/usr/bin/env python3
# -*- coding:utf-8 -*-
################################################################
# Copyright 2023 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2023-04-21
################################################################

import numpy as np
import random
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.autograd import Variable
from torch.nn import functional as F
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

import os
import logging
import sys
from rospkg import RosPack

ROS_PKG_PATH = RosPack().get_path('hex_rl_controller')
LOG_PATH = f"{ROS_PKG_PATH}/log"
POLICY_PATH = f"{ROS_PKG_PATH}/policy"

sys.path.append(f"{ROS_PKG_PATH}/scripts/python")
from model.net import PolarEncoderNet
from model.net import StateNet
from model.net import ActorNet, CriticNet


class Ppo():

    def __init__(self,
                 stage_type,
                 env_type,
                 num_step=128,
                 num_agent=1,
                 num_max_update=1e3,
                 num_mini_batch=128,
                 num_epoch=2,
                 gamma=0.99,
                 lamda=0.95,
                 coeff_entropy=1e-2,
                 clip_value=0.1,
                 encoder_lr=3e-4,
                 state_lr=3e-4,
                 actor_lr=3e-4,
                 critic_lr=3e-4,
                 use_lr_decay=True,
                 seed=2056,
                 encoder_file=None,
                 state_file=None,
                 actor_file=None,
                 critic_file=None,
                 freeze_state=False,
                 freeze_encoder=False):
        #### config ####
        self.__stage_type = stage_type
        self.__env_type = env_type
        self.__num_step = num_step
        self.__num_agent = num_agent
        self.__num_max_update = num_max_update
        self.__num_mini_batch = num_mini_batch
        self.__num_epoch = num_epoch
        self.__gamma = gamma
        self.__lamda = lamda
        self.__coeff_entropy = coeff_entropy
        self.__clip_value = clip_value
        self.__encoder_lr = encoder_lr
        self.__state_lr = state_lr
        self.__actor_lr = actor_lr
        self.__critic_lr = critic_lr
        self.__use_lr_decay = use_lr_decay

        #### seed ####
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        #### logger ####
        # path
        log_dir = f"{LOG_PATH}/{stage_type}"
        log_ppo = f"{log_dir}/ppo.log"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        # handler
        log_level = logging.INFO
        log_formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s")
        file_handler_ppo = logging.FileHandler(log_ppo, mode='a')
        file_handler_ppo.setLevel(log_level)
        file_handler_ppo.setFormatter(log_formatter)
        # logger
        self.__logger_ppo = logging.getLogger('logger_ppo')
        self.__logger_ppo.setLevel(log_level)
        self.__logger_ppo.addHandler(file_handler_ppo)

        #### policy ####
        # net
        self.__encoder_net = PolarEncoderNet()
        self.__state_net = StateNet()
        self.__actor_net = ActorNet()
        self.__critic_net = CriticNet()
        self.__encoder_net.cuda()
        self.__state_net.cuda()
        self.__actor_net.cuda()
        self.__critic_net.cuda()
        # load pretrained file
        if not encoder_file is None:
            self.__encoder_net.load_state_dict(
                torch.load(f"{POLICY_PATH}/{encoder_file}"))
        if not state_file is None:
            self.__state_net.load_state_dict(
                torch.load(f"{POLICY_PATH}/{state_file}"))
        if not actor_file is None:
            self.__actor_net.load_state_dict(
                torch.load(f"{POLICY_PATH}/{actor_file}"))
        if not critic_file is None:
            self.__critic_net.load_state_dict(
                torch.load(f"{POLICY_PATH}/{critic_file}"))
        # optimizer
        self.__encoder_opt = Adam(self.__encoder_net.parameters(),
                                  lr=self.__encoder_lr)
        self.__state_opt = Adam(self.__state_net.parameters(),
                                lr=self.__state_lr,
                                eps=1e-5)
        self.__actor_opt = Adam(self.__actor_net.parameters(),
                                lr=self.__actor_lr,
                                eps=1e-5)
        self.__critic_opt = Adam(self.__critic_net.parameters(),
                                 lr=self.__critic_lr,
                                 eps=1e-5)

        #### update ####
        self.__freeze_state = freeze_state
        self.__freeze_encoder = freeze_encoder
        self.__update_steps = 0

    ########################################
    #### get handle
    ########################################
    def get_update_steps(self):
        return self.__update_steps

    def is_finish(self):
        return self.__update_steps >= self.__num_max_update

    def choose_action(self, observation_list):
        laser_list, velocity_list, intention_list = [], [], []
        for item in observation_list:
            laser_list.append(item[0])
            velocity_list.append(item[1])
            intention_list.append(item[2])

        lasers = Variable(torch.from_numpy(
            np.asarray(laser_list))).float().cuda()
        velocities = Variable(torch.from_numpy(
            np.asarray(velocity_list))).float().cuda()
        intentions = Variable(torch.from_numpy(
            np.asarray(intention_list))).float().cuda()

        with torch.no_grad():
            features = self.__encoder_net(lasers)
            states = self.__state_net(features, velocities, intentions)
            dist = self.__actor_net.get_dist(states)
            actions = dist.sample()
            logprobs = dist.log_prob(actions)
            values = self.__critic_net(states)

            actions = actions.cpu().numpy()
            logprobs = logprobs.cpu().numpy()
            values = values.cpu().numpy()

        return actions, logprobs, values

    def evaluate(self, observation_list):
        laser_list, velocity_list, intention_list = [], [], []
        for item in observation_list:
            laser_list.append(item[0])
            velocity_list.append(item[1])
            intention_list.append(item[2])

        lasers = Variable(torch.from_numpy(
            np.asarray(laser_list))).float().cuda()
        velocities = Variable(torch.from_numpy(
            np.asarray(velocity_list))).float().cuda()
        intentions = Variable(torch.from_numpy(
            np.asarray(intention_list))).float().cuda()

        with torch.no_grad():
            features = self.__encoder_net(lasers)
            states = self.__state_net(features, velocities, intentions)
            actions = self.__actor_net.mean(states).cpu().numpy()

        return actions

    ########################################
    #### update handle
    ########################################
    def update(self, memory, filter_index=None):
        # get memory
        laser_batch = np.asarray(memory.get_lasers())
        velocity_batch = np.asarray(memory.get_velocitys())
        intention_batch = np.asarray(memory.get_intentions())
        action_batch = np.asarray(memory.get_actions())
        reward_batch = np.asarray(memory.get_rewards())
        terminated_batch = np.asarray(memory.get_terminateds())
        value_batch = np.asarray(memory.get_values())
        old_logprob_batch = np.asarray(memory.get_logprobs())
        # print(f"laser: {laser_batch.shape}")
        # print(f"velocity: {velocity_batch.shape}")
        # print(f"intention: {intention_batch.shape}")
        # print(f"action: {action_batch.shape}")
        # print(f"reward: {reward_batch.shape}")
        # print(f"terminated: {terminated_batch.shape}")
        # print(f"value: {value_batch.shape}")
        # print(f"old_logprob: {old_logprob_batch.shape}")

        # calculate advantage
        value_batch = value_batch.reshape(self.__num_step + 1, self.__num_agent)
        advantage_batch = np.zeros((self.__num_step, self.__num_agent))
        gae = np.zeros((self.__num_agent,))
        for t in range(self.__num_step - 1, -1, -1):
            delta = reward_batch[t, :] + self.__gamma * (1.0 - terminated_batch[
                t, :]) * value_batch[t + 1, :] - value_batch[t, :]
            gae = delta + self.__gamma * self.__lamda * (
                1.0 - terminated_batch[t, :]) * gae
            advantage_batch[t, :] = gae
        target_batch = advantage_batch + value_batch[:-1, :]

        # advantage normalization
        advantage_batch = (advantage_batch -
                           advantage_batch.mean()) / advantage_batch.std()

        # reshape data
        record_num = self.__num_step * self.__num_agent
        lasers = laser_batch.reshape((record_num, 3, 256))
        velocitys = velocity_batch.reshape((record_num, 2))
        intentions = intention_batch.reshape((record_num, 2))
        actions = action_batch.reshape((record_num, 2))
        old_logprobs = old_logprob_batch.reshape((record_num, 2))
        advantages = advantage_batch.reshape((record_num, 1))
        targets = target_batch.reshape((record_num, 1))
        if not filter_index is None:
            lasers = np.delete(lasers, filter_index, axis=0)
            velocitys = np.delete(velocitys, filter_index, axis=0)
            intentions = np.delete(intentions, filter_index, axis=0)
            actions = np.delete(actions, filter_index, axis=0)
            old_logprobs = np.delete(old_logprobs, filter_index, axis=0)
            advantages = np.delete(advantages, filter_index, axis=0)
            targets = np.delete(targets, filter_index, axis=0)
            record_num = old_logprobs.shape[0]

        # update
        for _ in range(self.__num_epoch):
            for index in BatchSampler(SubsetRandomSampler(range(record_num)),
                                      batch_size=self.__num_mini_batch,
                                      drop_last=False):
                sampled_lasers = Variable(torch.from_numpy(
                    lasers[index])).float().cuda()
                sampled_velocitys = Variable(torch.from_numpy(
                    velocitys[index])).float().cuda()
                sampled_intentions = Variable(
                    torch.from_numpy(intentions[index])).float().cuda()
                sampled_actions = Variable(torch.from_numpy(
                    actions[index])).float().cuda()
                sampled_old_logprobs = Variable(
                    torch.from_numpy(old_logprobs[index])).float().cuda()
                sampled_advantages = Variable(
                    torch.from_numpy(advantages[index])).float().cuda()
                sampled_targets = Variable(torch.from_numpy(
                    targets[index])).float().cuda()

                # evaluate action
                features = self.__encoder_net(sampled_lasers)
                states = self.__state_net(features, sampled_velocitys,
                                          sampled_intentions)
                dist = self.__actor_net.get_dist(states)
                dist_entropy = dist.entropy().sum(1, keepdim=True)
                new_logprobs = dist.log_prob(sampled_actions)
                new_values = self.__critic_net(states)

                # actor loss
                ratios = torch.exp(
                    new_logprobs.sum(1, keepdim=True) -
                    sampled_old_logprobs.sum(1, keepdim=True))
                sampled_advantages = sampled_advantages.view(-1, 1)
                surr1 = ratios * sampled_advantages
                surr2 = torch.clamp(ratios, 1 - self.__clip_value,
                                    1 + self.__clip_value) * sampled_advantages
                actor_loss = -torch.min(
                    surr1, surr2) - self.__coeff_entropy * dist_entropy

                # critic loss
                sampled_targets = sampled_targets.view(-1, 1)
                critic_loss = F.mse_loss(new_values, sampled_targets)

                # update
                self.__actor_opt.zero_grad()
                self.__critic_opt.zero_grad()
                self.__state_opt.zero_grad()
                self.__encoder_opt.zero_grad()
                loss = actor_loss.mean() + 1.0 * critic_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.__actor_net.parameters(),
                                               0.5)
                torch.nn.utils.clip_grad_norm_(self.__critic_net.parameters(),
                                               0.5)
                self.__actor_opt.step()
                self.__critic_opt.step()
                if not self.__freeze_state:
                    torch.nn.utils.clip_grad_norm_(
                        self.__state_net.parameters(), 0.5)
                    self.__state_opt.step()
                    if not self.__freeze_encoder:
                        torch.nn.utils.clip_grad_norm_(
                            self.__encoder_net.parameters(), 0.5)
                        self.__encoder_opt.step()

                # summary log
                info_actor_loss = float(
                    actor_loss.mean().detach().cpu().numpy())
                info_critic_loss = float(critic_loss.detach().cpu().numpy())
                info_dist_entropy = float(
                    dist_entropy.mean().detach().cpu().numpy())
                self.__logger_ppo.info(
                    f"{info_actor_loss}, {info_critic_loss}, {info_dist_entropy}"
                )

        self.__update_steps += 1
        if self.__update_steps % 200 == 0:
            self.save_policy()
        if self.__use_lr_decay:
            self.__lr_decay()

    def __lr_decay(self):
        # coeff
        # coeff_decay = 1.0 - self.__update_steps / self.__num_max_update
        coeff_decay = 0.999

        # calculate learning rate now
        encoder_lr_now = self.__encoder_lr * coeff_decay
        state_lr_now = self.__state_lr * coeff_decay
        actor_lr_now = self.__actor_lr * coeff_decay
        critic_lr_now = self.__critic_lr * coeff_decay

        # update learning rate
        for p in self.__encoder_opt.param_groups:
            p['lr'] = encoder_lr_now
        for p in self.__state_opt.param_groups:
            p['lr'] = state_lr_now
        for p in self.__actor_opt.param_groups:
            p['lr'] = actor_lr_now
        for p in self.__critic_opt.param_groups:
            p['lr'] = critic_lr_now

    ########################################
    #### update handle
    ########################################
    def save_policy(self, final=False):
        if final:
            encoder_path = f"{POLICY_PATH}/{self.__stage_type}/encoder/encoder.pth"
            state_path = f"{POLICY_PATH}/{self.__stage_type}/state/state.pth"
            actor_path = f"{POLICY_PATH}/{self.__stage_type}/actor/actor.pth"
            critic_path = f"{POLICY_PATH}/{self.__stage_type}/critic/critic.pth"
        else:
            encoder_path = f"{POLICY_PATH}/{self.__stage_type}/encoder/encoder_update_{self.__update_steps}.pth"
            state_path = f"{POLICY_PATH}/{self.__stage_type}/state/state_update_{self.__update_steps}.pth"
            actor_path = f"{POLICY_PATH}/{self.__stage_type}/actor/actor_update_{self.__update_steps}.pth"
            critic_path = f"{POLICY_PATH}/{self.__stage_type}/critic/critic_update_{self.__update_steps}.pth"
        if not self.__freeze_state:
            torch.save(self.__state_net.state_dict(), state_path)
            if not self.__freeze_encoder:
                torch.save(self.__encoder_net.state_dict(), encoder_path)
        torch.save(self.__actor_net.state_dict(), actor_path)
        torch.save(self.__critic_net.state_dict(), critic_path)
        print(f"#### model saved : update {self.__update_steps} ####")
