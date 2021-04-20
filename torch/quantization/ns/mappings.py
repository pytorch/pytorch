import operator

import torch
import torch.nn as nn
import torch.nn.functional as F
toq = torch.ops.quantized

import torch.nn.quantized as nnq
import torch.nn.quantized.dynamic as nnqd
import torch.nn.intrinsic.quantized as nniq
import torch.nn.intrinsic.qat as nniqat
import torch.nn.intrinsic as nni
import torch.nn.qat as nnqat

from typing import Callable, Set

FUNS_IO_TYPE_FP32 = set([
    F.linear,
    F.conv1d,
    F.conv2d,
    F.conv3d,
    F.elu,
    F.hardswish,
    F.instance_norm,
    F.layer_norm,
    F.leaky_relu,
    F.silu,
])

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
    toq.elu,
    toq.hardswish,
    toq.instance_norm,
    toq.layer_norm,
    toq.leaky_relu,
])

FUNS_IO_TYPE_FP32_OR_INT8 = set([
    F.relu,
    F.tanh,
    torch.tanh,
    F.sigmoid,
    torch.sigmoid,
    F.hardsigmoid,
    operator.floordiv,
    torch.adaptive_avg_pool1d,
    F.adaptive_avg_pool2d,
    F.adaptive_avg_pool3d,
    F.dropout,
    F.hardtanh,
    F.hardtanh_,
    F.interpolate,
    F.max_pool1d,
    F.max_pool2d,
    F.max_pool3d,
    F.relu6,
    torch.avg_pool1d,
    torch._C._nn.avg_pool2d,
    torch._C._nn.avg_pool3d,
    torch.cat,
    torch.chunk,
    torch.clamp,
    torch.flatten,
    torch.transpose,
    torch.max,
    torch.mean,
    torch.min,
    torch.repeat_interleave,
    torch.sort,
    torch.squeeze,
    torch.stack,
    torch.unsqueeze,
])

FUNS_UNMATCHABLE: Set[Callable] = set([
    torch.quantize_per_tensor,
    operator.getitem,
])

MODS_IO_TYPE_FP32 = set([
    nn.Linear,
    nnqat.Linear,
    nnqd.Linear,
    torch.nn.modules.linear._LinearWithBias,
    nn.Conv1d,
    nn.Conv2d,
    nn.Conv3d,
    nnqat.Conv2d,
    nnqat.Conv3d,
    nn.LSTM,
    # note: nnqd.Linear is an instance of nnq.Linear, so this
    # check has to happen before the int8 module check
    nnqd.LSTM,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
    nn.ConvTranspose1d,
    nn.ConvTranspose2d,
    nn.ELU,
    nn.GroupNorm,
    nn.InstanceNorm1d,
    nn.InstanceNorm2d,
    nn.InstanceNorm3d,
    nn.LayerNorm,
    nn.Hardswish,
    nn.LeakyReLU,
    nn.ReLU6,
    nn.SiLU,
    nni.BNReLU2d,
    nni.BNReLU3d,
    nni.ConvReLU1d,
    nni.ConvReLU2d,
    nni.ConvReLU3d,
    nni.LinearReLU,
    nni.ConvBn1d,
    nni.ConvBn2d,
    nni.ConvBn3d,
    nniqat.ConvBn1d,
    nniqat.ConvBn2d,
    nniqat.ConvBn3d,
    nniqat.ConvBnReLU1d,
    nniqat.ConvBnReLU2d,
    nniqat.ConvBnReLU3d,
    nniqat.ConvReLU2d,
    nniqat.ConvReLU3d,
    nniqat.LinearReLU,
])

MODS_IO_TYPE_INT8 = set([
    nnq.Linear,
    nnq.Conv1d,
    nnq.Conv2d,
    nniq.ConvReLU2d,
    nnq.Conv3d,
    nnq.BatchNorm2d,
    nnq.BatchNorm3d,
    nnq.ConvTranspose1d,
    nnq.ConvTranspose2d,
    nnq.ELU,
    nnq.GroupNorm,
    nnq.InstanceNorm1d,
    nnq.InstanceNorm2d,
    nnq.InstanceNorm3d,
    nnq.LayerNorm,
    nnq.Hardswish,
    nnq.LeakyReLU,
    nnq.ReLU6,
    nniq.BNReLU2d,
    nniq.BNReLU3d,
    nniq.ConvReLU1d,
    nniq.ConvReLU2d,
    nniq.ConvReLU3d,
    nniq.LinearReLU,
])

MODS_IO_TYPE_FP32_OR_INT8 = set([
    nn.ReLU,
    nn.Tanh,
    nn.Sigmoid,
    nn.Hardsigmoid,
    nn.AdaptiveAvgPool1d,
    nn.AdaptiveAvgPool2d,
    nn.AdaptiveAvgPool3d,
    nn.AvgPool1d,
    nn.AvgPool2d,
    nn.AvgPool3d,
    nn.Dropout,
    nn.Hardtanh,
    nn.Identity,
    nn.MaxPool1d,
    nn.MaxPool2d,
    nn.MaxPool3d,
    nn.ReLU6,
])

MODS_UNMATCHABLE: Set[Callable] = set([
    torch.quantization.ObserverBase,
    torch.quantization.FakeQuantizeBase,
])

METHS_IO_TYPE_FP32_OR_INT8 = set([
    'sigmoid_',
    'sigmoid',
    'tanh_',
    'tanh',
    'hardsigmoid_',
    'hardsigmoid',
    'relu_',
    'relu',
])

METHS_UNMATCHABLE = set([
    'to',
    'dequantize',
    'reshape',
    'view',
    'unsqueeze_',
    'unsqueeze',
    'transpose',
    'squeeze_',
    'squeeze',
    'size',
    'shape',
    'resize_',
    'repeat_interleave',
    'repeat',
    'permute',
    'numel',
    'mean',
    'detach_',
    'detach',
    'contiguous',
    'clamp',
    'chunk',
])
