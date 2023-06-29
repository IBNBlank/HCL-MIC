#!/usr/bin/env python3
# -*- coding:utf-8 -*-
################################################################
# Copyright 2023 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2023-04-19
################################################################

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions import Beta


def orthogonal_init(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)


class PolarEncoderNet(nn.Module):

    def __init__(self):
        super(PolarEncoderNet, self).__init__()
        # activate
        self.activate = nn.Tanh()

        # net
        self.conv1 = nn.Conv1d(in_channels=3,
                               out_channels=32,
                               kernel_size=5,
                               stride=2,
                               padding=1)
        self.conv2 = nn.Conv1d(in_channels=32,
                               out_channels=32,
                               kernel_size=3,
                               stride=2,
                               padding=1)
        self.flat = nn.Flatten()
        self.fc = nn.Linear(64 * 32, 128)

        # init
        orthogonal_init(self.fc)

    def forward(self, raw):
        feature = self.activate(self.conv1(raw))
        feature = self.activate(self.conv2(feature))
        feature = self.activate(self.flat(feature))
        feature = self.activate(self.fc(feature))

        return feature


class PolarDecoderNet(nn.Module):

    def __init__(self):
        super(PolarDecoderNet, self).__init__()
        # activate
        self.activate = nn.Tanh()

        # net
        self.fc = nn.Linear(128, 64 * 32)
        self.unflat = nn.Unflatten(-1, (32, 64))
        self.deconv1 = nn.ConvTranspose1d(in_channels=32,
                                          out_channels=32,
                                          kernel_size=3,
                                          stride=2,
                                          padding=1)
        self.deconv2 = nn.ConvTranspose1d(in_channels=32,
                                          out_channels=3,
                                          kernel_size=5,
                                          stride=2,
                                          padding=1,
                                          output_padding=1)

        # init
        orthogonal_init(self.fc)

    def forward(self, feature):
        raw = self.activate(self.fc(feature))
        raw = self.activate(self.unflat(raw))
        raw = self.activate(self.deconv1(raw))
        raw = self.activate(self.deconv2(raw))

        return raw


# class GridEncoder(nn.Module):

#     def __init__(self):
#         super(GridEncoder, self).__init__()

# class GridDecoder(nn.Module):

#     def __init__(self):
#         super(GridDecoder, self).__init__()


class StateNet(nn.Module):

    def __init__(self):
        super(StateNet, self).__init__()
        # activate
        self.activate = nn.Tanh()

        # net
        self.fc = nn.Linear(128 + 2 + 2, 64)

        # init
        orthogonal_init(self.fc)

    def forward(self, feature, velocity, intention):
        state = torch.cat((feature, velocity, intention), dim=-1)
        state = self.activate(self.fc(state))

        return state


class StateRestoreNet(nn.Module):

    def __init__(self):
        super(StateRestoreNet, self).__init__()
        # activate
        self.activate = nn.Tanh()

        # net
        self.fc = nn.Linear(64, 128 + 2 + 2)

        # init
        orthogonal_init(self.fc)

    def forward(self, state):
        state = self.activate(self.fc(state))
        feature, velocity, intention = torch.split(state, (128, 2, 2), dim=-1)

        return feature, velocity, intention


class ActorNet(nn.Module):

    def __init__(self):
        super(ActorNet, self).__init__()
        # activate
        self.activate = nn.Tanh()

        # net
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 4)

        # init
        orthogonal_init(self.fc1)
        orthogonal_init(self.fc2)

    def forward(self, state):
        result = self.activate(self.fc1(state))
        result = F.softplus(self.fc2(result)) + 1.0
        alpha, beta = torch.split(result, (2, 2), dim=-1)

        return alpha, beta

    def get_dist(self, state):
        alpha, beta = self.forward(state)
        dist = Beta(alpha, beta)

        return dist

    def mean(self, state):
        alpha, beta = self.forward(state)
        mean = alpha / (alpha + beta)

        return mean


class CriticNet(nn.Module):

    def __init__(self):
        super(CriticNet, self).__init__()
        # activate
        self.activate = nn.Tanh()

        # net
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 1)

        # init
        orthogonal_init(self.fc1)
        orthogonal_init(self.fc2)

    def forward(self, state):
        value = self.activate(self.fc1(state))
        value = self.fc2(value)

        return value