#!/usr/bin/env python3
# -*- coding:utf-8 -*-
################################################################
# Copyright 2023 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2023-05-10
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
from model.net import PolarEncoderNet, PolarDecoderNet
from model.net import StateNet, StateRestoreNet


class AutoEncoder():

    def __init__(self,
                 stage_type,
                 env_type,
                 num_mini_batch=128,
                 num_epoch=2,
                 encoder_lr=3e-4,
                 state_lr=3e-4,
                 use_lr_decay=True,
                 pure_lidar=False,
                 seed=2056,
                 encoder_file=None,
                 state_file=None,
                 state_restore_file=None,
                 decoder_file=None):
        #### config ####
        self.__stage_type = stage_type
        self.__env_type = env_type
        self.__num_max_update = 1e3
        self.__num_mini_batch = num_mini_batch
        self.__num_epoch = num_epoch
        self.__encoder_lr = encoder_lr
        self.__state_lr = state_lr
        self.__use_lr_decay = use_lr_decay
        self.__pure_lidar = pure_lidar

        #### seed ####
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        #### logger ####
        # path
        log_dir = f"{LOG_PATH}/{stage_type}"
        log_pretrain = f"{log_dir}/pretrain.log"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        # handler
        log_level = logging.INFO
        log_formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s")
        file_handler_pretrain = logging.FileHandler(log_pretrain, mode='a')
        file_handler_pretrain.setLevel(log_level)
        file_handler_pretrain.setFormatter(log_formatter)
        # logger
        self.__logger_pretrain = logging.getLogger('logger_pretrain')
        self.__logger_pretrain.setLevel(log_level)
        self.__logger_pretrain.addHandler(file_handler_pretrain)

        #### model ####
        # net
        self.__encoder_net = PolarEncoderNet()
        self.__decoder_net = PolarDecoderNet()
        self.__encoder_net.cuda()
        self.__decoder_net.cuda()
        if not pure_lidar:
            self.__state_net = StateNet()
            self.__state_restore_net = StateRestoreNet()
            self.__state_net.cuda()
            self.__state_restore_net.cuda()

        # load pretrained file
        if not encoder_file is None:
            self.__encoder_net.load_state_dict(
                torch.load(f"{POLICY_PATH}/{encoder_file}"))
        if not decoder_file is None:
            self.__decoder_net.load_state_dict(
                torch.load(f"{POLICY_PATH}/{decoder_file}"))
        if not pure_lidar:
            if not state_file is None:
                self.__state_net.load_state_dict(
                    torch.load(f"{POLICY_PATH}/{state_file}"))
            if not state_restore_file is None:
                self.__state_restore_net.load_state_dict(
                    torch.load(f"{POLICY_PATH}/{state_restore_file}"))
        # optimizer
        self.__encoder_opt = Adam(self.__encoder_net.parameters(),
                                  lr=self.__encoder_lr)
        self.__decoder_opt = Adam(self.__decoder_net.parameters(),
                                  lr=self.__encoder_lr)
        if not pure_lidar:
            self.__state_opt = Adam(self.__state_net.parameters(),
                                    lr=self.__state_lr,
                                    eps=1e-5)
            self.__state_restore_opt = Adam(
                self.__state_restore_net.parameters(),
                lr=self.__state_lr,
                eps=1e-5)

        #### update ####
        self.__update_steps = 0

    ########################################
    #### update handle
    ########################################
    def update(self, memory):
        # get memory
        laser_batch = np.asarray(memory.get_lasers())
        velocity_batch = np.asarray(memory.get_velocitys())
        intention_batch = np.asarray(memory.get_intentions())
        record_num = laser_batch.shape[0]
        self.__num_max_update = np.ceil(
            record_num / (self.__num_epoch * self.__num_mini_batch))

        # update
        update_lr_count = 0
        for index in BatchSampler(SubsetRandomSampler(range(record_num)),
                                  batch_size=self.__num_mini_batch,
                                  drop_last=False):
            sampled_lasers = Variable(torch.from_numpy(
                laser_batch[index])).float().cuda()
            sampled_velocitys = Variable(torch.from_numpy(
                velocity_batch[index])).float().cuda()
            sampled_intentions = Variable(torch.from_numpy(
                intention_batch[index])).float().cuda()

            if self.__pure_lidar:
                # evaluate
                features = self.__encoder_net(sampled_lasers)
                restored_lasers = self.__decoder_net(features)

                # feature loss
                feature_loss = F.mse_loss(restored_lasers, sampled_lasers)

                # update
                self.__encoder_opt.zero_grad()
                self.__decoder_opt.zero_grad()
                loss = feature_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.__encoder_net.parameters(),
                                               0.5)
                torch.nn.utils.clip_grad_norm_(self.__decoder_net.parameters(),
                                               0.5)
                self.__encoder_opt.step()
                self.__decoder_opt.step()

                # summary log
                info_feature_loss = float(feature_loss.detach().cpu().numpy())
                self.__logger_pretrain.info(f"{info_feature_loss}")
            else:
                # evaluate
                features = self.__encoder_net(sampled_lasers)

                states = self.__state_net(features, sampled_velocitys,
                                          sampled_intentions)
                restored_features, restored_velocities, restored_intentions = self.__state_restore_net(
                    states)
                restored_lasers = self.__decoder_net(restored_features)

                # state loss
                state_loss = F.mse_loss(restored_velocities,
                                        sampled_velocitys) + F.mse_loss(
                                            restored_intentions, sampled_intentions)

                # feature loss
                feature_loss = F.mse_loss(restored_lasers, sampled_lasers)

                # update
                self.__encoder_opt.zero_grad()
                self.__state_opt.zero_grad()
                self.__state_restore_opt.zero_grad()
                self.__decoder_opt.zero_grad()
                loss = state_loss + feature_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.__encoder_net.parameters(),
                                               0.5)
                torch.nn.utils.clip_grad_norm_(self.__state_net.parameters(),
                                               0.5)
                torch.nn.utils.clip_grad_norm_(
                    self.__state_restore_net.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_(self.__decoder_net.parameters(),
                                               0.5)
                self.__encoder_opt.step()
                self.__state_opt.step()
                self.__state_restore_opt.step()
                self.__decoder_opt.step()

                # summary log
                info_state_loss = float(state_loss.detach().cpu().numpy())
                info_feature_loss = float(feature_loss.detach().cpu().numpy())
                self.__logger_pretrain.info(
                    f"{info_state_loss}, {info_feature_loss}")

            update_lr_count += 1
            if update_lr_count % self.__num_epoch == 0:
                self.__update_steps += 1
                print(f"#### update {self.__update_steps} finish ####")
                if self.__use_lr_decay:
                    self.__lr_decay()
                if self.__update_steps % 200 == 0:
                    self.__save_policy()

        self.__save_policy(final=True)

    def __lr_decay(self):
        # coeff
        coeff_decay = 1.0 - self.__update_steps / self.__num_max_update

        # calculate learning rate now
        encoder_lr_now = self.__encoder_lr * coeff_decay
        state_lr_now = self.__state_lr * coeff_decay

        # update learning rate
        for p in self.__encoder_opt.param_groups:
            p['lr'] = encoder_lr_now
        for p in self.__decoder_opt.param_groups:
            p['lr'] = encoder_lr_now
        if not self.__pure_lidar:
            for p in self.__state_opt.param_groups:
                p['lr'] = state_lr_now
            for p in self.__state_restore_opt.param_groups:
                p['lr'] = state_lr_now

    def __save_policy(self, final=False):
        if final:
            encoder_path = f"{POLICY_PATH}/{self.__stage_type}/encoder/encoder.pth"
            state_path = f"{POLICY_PATH}/{self.__stage_type}/state/state.pth"
            state_restore_path = f"{POLICY_PATH}/{self.__stage_type}/state_restore/state_restore.pth"
            decoder_path = f"{POLICY_PATH}/{self.__stage_type}/decoder/decoder.pth"
        else:
            encoder_path = f"{POLICY_PATH}/{self.__stage_type}/encoder/encoder_update_{self.__update_steps}.pth"
            state_path = f"{POLICY_PATH}/{self.__stage_type}/state/state_update_{self.__update_steps}.pth"
            state_restore_path = f"{POLICY_PATH}/{self.__stage_type}/state_restore/state_restore_update_{self.__update_steps}.pth"
            decoder_path = f"{POLICY_PATH}/{self.__stage_type}/decoder/decoder_update_{self.__update_steps}.pth"
        torch.save(self.__encoder_net.state_dict(), encoder_path)
        torch.save(self.__decoder_net.state_dict(), decoder_path)
        if not self.__pure_lidar:
            torch.save(self.__state_net.state_dict(), state_path)
            torch.save(self.__state_restore_net.state_dict(),
                       state_restore_path)
        print(f"#### model saved : update {self.__update_steps} ####")
