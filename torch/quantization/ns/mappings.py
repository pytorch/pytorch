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

from .ns_types import NSNodeTargetType

from typing import Set, Dict


def get_base_name_to_sets_of_related_ops() -> Dict[str, Set[NSNodeTargetType]]:
    base_name_to_sets_of_related_ops: Dict[str, Set[NSNodeTargetType]] = {
        # conv modules
        'torch.nn.Conv1d': set([
            nn.Conv1d,
            nnq.Conv1d,
            nniqat.ConvBn1d,
            nniqat.ConvBnReLU1d,
            nniq.ConvReLU1d,
            nni.ConvReLU1d,
        ]),
        'torch.nn.Conv2d': set([
            nn.Conv2d,
            nnq.Conv2d,
            nnqat.Conv2d,
            nniqat.ConvBn2d,
            nniqat.ConvBnReLU2d,
            nniqat.ConvReLU2d,
            nniq.ConvReLU2d,
            nni.ConvReLU2d,
        ]),
        'torch.nn.Conv3d': set([
            nn.Conv3d,
            nnq.Conv3d,
            nnqat.Conv3d,
            nniqat.ConvBn3d,
            nniqat.ConvBnReLU3d,
            nniqat.ConvReLU3d,
            nniq.ConvReLU3d,
            nni.ConvReLU3d,
        ]),
        # conv functionals
        'torch.nn.functional.conv1d': set([
            F.conv1d,
            toq.conv1d,
            toq.conv1d_relu,
        ]),
        'torch.nn.functional.conv2d': set([
            F.conv2d,
            toq.conv2d,
            toq.conv2d_relu,
        ]),
        'torch.nn.functional.conv3d': set([
            F.conv3d,
            toq.conv3d,
            toq.conv3d_relu,
        ]),
        # linear modules
        'torch.nn.Linear': set([
            nn.Linear,
            nnq.Linear,
            nni.LinearReLU,
            nniq.LinearReLU,
            nnqat.Linear,
            nnqd.Linear,
            nniqat.LinearReLU,
            nn.modules.linear._LinearWithBias,
        ]),
        # linear functionals
        'torch.nn.functional.linear': set([
            F.linear,
            toq.linear,
            toq.linear_relu,
        ]),
        # average pool
        'torch.nn.AvgPool1d': set([
            nn.AvgPool1d,
            torch.avg_pool1d,
        ]),
        'torch.nn.AvgPool2d': set([
            nn.AvgPool2d,
            torch._C._nn.avg_pool2d,
        ]),
        'torch.nn.AvgPool3d': set([
            nn.AvgPool3d,
            torch._C._nn.avg_pool3d,
        ]),
        # adaptive average pool
        'torch.nn.AdaptiveAvgPool1d': set([
            nn.AdaptiveAvgPool1d,
            F.adaptive_avg_pool1d,
        ]),
        'torch.nn.AdaptiveAvgPool2d': set([
            nn.AdaptiveAvgPool2d,
            F.adaptive_avg_pool2d,
        ]),
        'torch.nn.AdaptiveAvgPool3d': set([
            nn.AdaptiveAvgPool3d,
            F.adaptive_avg_pool3d,
        ]),
        # LSTM
        'torch.nn.LSTM': set([
            nn.LSTM,
            nnqd.LSTM,
        ]),
        # add
        'torch.add': set([
            torch.add,
            toq.add,
            operator.add,  # x + y
            toq.add_relu,
        ]),
        # cat
        'torch.cat': set([
            torch.cat,
            toq.cat,
        ]),
        # mul
        'torch.mul': set([
            torch.mul,
            toq.mul,
            operator.mul,
            toq.mul_relu,
        ]),
        # relu
        'torch.relu': set([
            F.relu,
            nn.ReLU,
            'relu',
            'relu_',
            torch.relu,
        ]),
        # relu6
        'torch.nn.functional.relu6': set([
            F.relu6,
        ]),
        # maxpool
        'torch.nn.MaxPool1d': set([
            nn.MaxPool1d,
            F.max_pool1d,
        ]),
        'torch.nn.MaxPool2d': set([
            nn.MaxPool2d,
            F.max_pool2d,
        ]),
        'torch.nn.MaxPool3d': set([
            nn.MaxPool3d,
            F.max_pool3d,
        ]),
        # sigmoid
        'torch.sigmoid': set([
            torch.sigmoid,
            'sigmoid',
            'sigmoid_',
            nn.Sigmoid,
        ]),
        # BatchNorm
        'torch.nn.BatchNorm2d': set([
            nn.BatchNorm2d,
            nnq.BatchNorm2d,
        ]),
        'torch.nn.BatchNorm3d': set([
            nn.BatchNorm3d,
            nnq.BatchNorm3d,
        ]),
        # ConvTranspose
        'torch.nn.ConvTranspose1d': set([
            nn.ConvTranspose1d,
            nnq.ConvTranspose1d,
        ]),
        'torch.nn.ConvTranspose2d': set([
            nn.ConvTranspose2d,
            nnq.ConvTranspose2d,
        ]),
        # ELU
        'torch.nn.ELU': set([
            nn.ELU,
            nnq.ELU,
        ]),
        # Embedding
        'torch.nn.Embedding': set([
            nn.Embedding,
            nnq.Embedding,
        ]),
        # EmbeddingBag
        'torch.nn.EmbeddingBag': set([
            nn.EmbeddingBag,
            nnq.EmbeddingBag,
        ]),
        # GroupNorm
        'torch.nn.GroupNorm': set([
            nn.GroupNorm,
            nnq.GroupNorm,
        ]),
        # Hardswish
        'torch.nn.Hardswish': set([
            nn.Hardswish,
            nnq.Hardswish,
        ]),
        # InstanceNorm
        'torch.nn.InstanceNorm1d': set([
            nn.InstanceNorm1d,
            nnq.InstanceNorm1d,
        ]),
        'torch.nn.InstanceNorm2d': set([
            nn.InstanceNorm2d,
            nnq.InstanceNorm2d,
        ]),
        'torch.nn.InstanceNorm3d': set([
            nn.InstanceNorm3d,
            nnq.InstanceNorm3d,
        ]),
        # LayerNorm
        'torch.nn.LayerNorm': set([
            nn.LayerNorm,
            nnq.LayerNorm,
        ]),
        # LeakyReLU
        'torch.nn.LeakyReLU': set([
            nn.LeakyReLU,
            nnq.LeakyReLU,
        ]),
        # ReLU6
        'torch.nn.ReLU6': set([
            nn.ReLU6,
            nnq.ReLU6,
        ]),
        # BNReLU2d
        'torch.nn.intrinsic.BNReLU2d': set([
            nni.BNReLU2d,
            nniq.BNReLU2d,
        ]),
        'torch.nn.intrinsic.BNReLU3d': set([
            nni.BNReLU3d,
            nniq.BNReLU3d,
        ]),
        # F.elu
        'torch.nn.functional.elu': set([
            F.elu,
            toq.elu,
        ]),
        # F.hardswish
        'torch.nn.functional.hardswish': set([
            F.hardswish,
            toq.hardswish,
        ]),
        # F.instance_norm
        'torch.nn.functional.instance_norm': set([
            F.instance_norm,
            toq.instance_norm,
        ]),
        # F.layer_norm
        'torch.nn.functional.layer_norm': set([
            F.layer_norm,
            toq.layer_norm,
        ]),
        # F.leaky_relu
        'torch.nn.functional.leaky_relu': set([
            F.leaky_relu,
            toq.leaky_relu,
        ]),
        # F.silu
        'torch.nn.SiLU': set([
            nn.SiLU,
            F.silu,
        ]),
        # F.tanh
        'torch.nn.Tanh': set([
            nn.Tanh,
            F.tanh,
            torch.tanh,
            'tanh_',
            'tanh',
        ]),
        # F.hardsigmoid
        'torch.nn.Hardsigmoid': set([
            'hardsigmoid_',
            'hardsigmoid',
            F.hardsigmoid,
            nn.Hardsigmoid,
        ]),
        # F.hardtanh
        'torch.nn.Hardtanh': set([
            nn.Hardtanh,
            F.hardtanh,
            F.hardtanh_,
        ]),
        # floordiv
        'operator.floordiv': set([
            operator.floordiv,
        ]),
        # unsqueeze
        'torch.unsqueeze': set([
            torch.unsqueeze,
        ]),
        # stack
        'torch.stack': set([
            torch.stack,
        ]),
        # squeeze
        'torch.squeeze': set([
            torch.squeeze,
        ]),
        # sort
        'torch.sort': set([
            torch.sort,
        ]),
        # repeat_interleave
        'torch.repeat_interleave': set([
            torch.repeat_interleave,
        ]),
        # min
        'torch.min': set([
            torch.min,
        ]),
        # mean
        'torch.mean': set([
            torch.mean,
        ]),
        # max
        'torch.max': set([
            torch.max,
        ]),
        # transpose
        'torch.transpose': set([
            torch.transpose,
        ]),
        # flatten
        'torch.flatten': set([
            torch.flatten,
        ]),
        # clamp
        'torch.clamp': set([
            torch.clamp,
        ]),
        # chunk
        'torch.chunk': set([
            torch.chunk,
        ]),
        # interpolate
        'torch.nn.functional.interpolate': set([
            torch.nn.functional.interpolate,
        ]),
        # dropout
        'torch.nn.Dropout': set([
            nn.Dropout,
            F.dropout,
        ]),
    }
    return base_name_to_sets_of_related_ops


# TODO(future PR): clean this up
def get_node_type_to_io_type_map() -> Dict[str, Set[NSNodeTargetType]]:
    FUNS_IO_TYPE_FP32: Set[NSNodeTargetType] = set([
        F.linear,
        F.conv1d,
        F.conv2d,
        F.conv3d,
        torch.cat,
        F.elu,
        F.hardswish,
        F.instance_norm,
        F.layer_norm,
        F.leaky_relu,
        F.silu,
        # TODO(future PR): implement shadowing for binary ops and
        # uncomment below
        # operator.add,
        # operator.mul,
    ])

    FUNS_IO_TYPE_FP16: Set[NSNodeTargetType] = set()

    FUNS_IO_TYPE_INT8: Set[NSNodeTargetType] = set([
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
        # TODO(future PR): implement shadowing for binary ops and
        # uncomment below
        # toq.add,
        # toq.mul,
    ])

    FUNS_IO_TYPE_FP32_OR_INT8: Set[NSNodeTargetType] = set([
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

    MODS_IO_TYPE_FP32: Set[NSNodeTargetType] = set([
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

    MODS_IO_TYPE_INT8: Set[NSNodeTargetType] = set([
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

    MODS_IO_TYPE_FP32_OR_INT8: Set[NSNodeTargetType] = set([
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

    METHS_IO_TYPE_FP32_OR_INT8: Set[NSNodeTargetType] = set([
        'sigmoid_',
        'sigmoid',
        'tanh_',
        'tanh',
        'hardsigmoid_',
        'hardsigmoid',
        'relu_',
        'relu',
    ])

    return {
        'funs_io_type_fp32': FUNS_IO_TYPE_FP32,
        'funs_io_type_fp16': FUNS_IO_TYPE_FP16,
        'funs_io_type_int8': FUNS_IO_TYPE_INT8,
        'funs_io_type_fp32_or_int8': FUNS_IO_TYPE_FP32_OR_INT8,
        'mods_io_type_fp32': MODS_IO_TYPE_FP32,
        'mods_io_type_int8': MODS_IO_TYPE_INT8,
        'mods_io_type_fp32_or_int8': MODS_IO_TYPE_FP32_OR_INT8,
        'meths_io_type_fp32_or_int8': METHS_IO_TYPE_FP32_OR_INT8,
    }


def get_unmatchable_types_map() -> Dict[str, Set[NSNodeTargetType]]:

    FUNS_UNMATCHABLE: Set[NSNodeTargetType] = set([
        torch.quantize_per_tensor,
        operator.getitem,
    ])

    MODS_UNMATCHABLE: Set[NSNodeTargetType] = set([
        torch.quantization.ObserverBase,
        torch.quantization.FakeQuantizeBase,
        nn.Identity,
    ])

    METHS_UNMATCHABLE: Set[NSNodeTargetType] = set([
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

    return {
        'funs_unmatchable': FUNS_UNMATCHABLE,
        'mods_unmatchable': MODS_UNMATCHABLE,
        'meths_unmatchable': METHS_UNMATCHABLE,
    }
