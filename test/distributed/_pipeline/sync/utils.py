# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from torch import nn
from typing import List

def convert_to_balance(pipe: nn.Sequential, balance: List[int]):
    device_idx = 0
    pipe_idx = 0
    balanced_pipe = []
    for num_layers in balance:
        layers = []
        for i in range(num_layers):
            layers.append(pipe[pipe_idx])
            pipe_idx += 1
        balanced_pipe.append(nn.Sequential(*layers).to(device_idx))
        device_idx += 1

    return nn.Sequential(*balanced_pipe)
