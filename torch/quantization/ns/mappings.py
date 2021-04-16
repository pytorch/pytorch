import torch
import torch.nn as nn
import torch.nn.functional as F
toq = torch.ops.quantized

import torch.nn.quantized as nnq
import torch.nn.quantized.dynamic as nnqd
import torch.nn.intrinsic.quantized as nniq
import torch.nn.intrinsic as nni

# TODO(future PR): make configurable
# TODO(future PR): fill out coverage
FUNS_IO_TYPE_FP32 = set([
    F.linear,
    F.conv1d,
    F.conv2d,
    F.conv3d,
    # TODO(future PR): move this to a new category, since
    # i/o can be fp32 or int8
    torch.cat,
    F.relu,
])

# TODO(future PR): make configurable
# TODO(future PR): fill out coverage
FUNS_IO_TYPE_INT8 = set([
    toq.linear,
    toq.linear_relu,
    toq.conv1d,
    toq.conv1d_relu,
    toq.conv2d,
    toq.conv2d_relu,
    toq.conv3d,
    toq.conv3d_relu,
    toq.cat,
])

# TODO(future PR): make configurable
# TODO(future PR): fill out coverage
MODS_IO_TYPE_FP32 = set([
    nn.Linear,
    nn.Conv1d,
    nn.Conv2d,
    nni.ConvReLU2d,
    nn.Conv3d,
    nn.LSTM,
    # note: nnqd.Linear is an instance of nnq.Linear, so this
    # check has to happen before the int8 module check
    nnqd.Linear,
    nnqd.LSTM,
])

# TODO(future PR): make configurable
# TODO(future PR): fill out coverage
MODS_IO_TYPE_INT8 = set([
    nnq.Linear,
    nnq.Conv1d,
    nnq.Conv2d,
    nniq.ConvReLU2d,
    nnq.Conv3d,
])
