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
from rospkg import RosPack

ROS_PKG_PATH = RosPack().get_path('hex_rl_controller')
LOG_PATH = f"{ROS_PKG_PATH}/log"
POLICY_PATH = f"{ROS_PKG_PATH}/policy"
PRETRAIN_PATH = f"{ROS_PKG_PATH}/pretrain_data"

sys.path.append(f"{ROS_PKG_PATH}/scripts/python")
from model.auto_encoder import AutoEncoder
from model.memory import Memory

STAGE_TYPE = "stage_pretrain"
PRETRAIN_TYPE = "stage_0"
ENV_TYPE = "polar"

NUM_MINI_BATCH = 128
NUM_EPOCH = 5

ENCODER_LR = 1e-4
STATE_LR = 1e-4

SEED = 2056


class StagePretrain():

    def __init__(self,
                 pure_lidar=False,
                 encoder_file=None,
                 state_file=None,
                 state_restore_file=None,
                 decoder_file=None):
        #### auto encoder ####
        self.__auto_encoder = AutoEncoder(stage_type=STAGE_TYPE,
                                          env_type=ENV_TYPE,
                                          num_mini_batch=NUM_MINI_BATCH,
                                          num_epoch=NUM_EPOCH,
                                          encoder_lr=ENCODER_LR,
                                          state_lr=STATE_LR,
                                          use_lr_decay=True,
                                          seed=SEED,
                                          pure_lidar=pure_lidar,
                                          encoder_file=encoder_file,
                                          state_file=state_file,
                                          state_restore_file=state_restore_file,
                                          decoder_file=decoder_file)

        #### memory ####
        self.__memory = Memory()

    def run(self):
        #### read data ####
        self.__memory.load_data(f"{PRETRAIN_PATH}/{PRETRAIN_TYPE}/pretrain.csv")

        #### pretrain ####
        self.__auto_encoder.update(self.__memory)


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
    # policy directory
    encoder_dir = f"{POLICY_PATH}/{STAGE_TYPE}/encoder/"
    state_dir = f"{POLICY_PATH}/{STAGE_TYPE}/state/"
    state_restore_dir = f"{POLICY_PATH}/{STAGE_TYPE}/state_restore/"
    decoder_dir = f"{POLICY_PATH}/{STAGE_TYPE}/decoder/"
    if not os.path.exists(encoder_dir):
        os.makedirs(encoder_dir)
    if not os.path.exists(state_dir):
        os.makedirs(state_dir)
    if not os.path.exists(state_restore_dir):
        os.makedirs(state_restore_dir)
    if not os.path.exists(decoder_dir):
        os.makedirs(decoder_dir)
    # pretrained parameters path
    encoder_pretrained_file = get_pretrained_file(
        ["stage_pretrain/encoder/encoder.pth"])
    state_pretrained_file = get_pretrained_file(
        ["stage_pretrain/state/state.pth"])
    state_restore_pretrained_file = get_pretrained_file(
        ["stage_pretrain/state_restore/state_restore.pth"])
    decoder_pretrained_file = get_pretrained_file(
        ["stage_pretrain/decoder/decoder.pth"])

    #### stage1 class ####
    stage_pretrain = StagePretrain(
        pure_lidar=False,
        encoder_file=encoder_pretrained_file,
        state_file=state_pretrained_file,
        state_restore_file=state_restore_pretrained_file,
        decoder_file=decoder_pretrained_file)

    try:
        stage_pretrain.run()
    except KeyboardInterrupt:
        pass