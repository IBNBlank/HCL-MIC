#!/usr/bin/env bash
################################################################
# Copyright 2023 Dong Zhaorui. All rights reserved.
# Author: Dong Zhaorui 847235539@qq.com
# Date  : 2023-04-20
################################################################

pkg_path=`rospack find hex_rl_controller`
python ${pkg_path}/scripts/python/python_stage_pretrain.py