import operator
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.intrinsic as nni
import torch.nn.intrinsic.qat as nniqat
import torch.nn.qat as nnqat
import torch.nn.quantized._reference as nnqr
from collections import namedtuple
from typing import List, Dict, Any
from .observation_type import ObservationType
from ..fuser_method_mappings import (
    reverse_sequential_wrapper2,
    reverse2,
    reverse3,
    fuse_conv_bn,
    fuse_conv_bn_relu,
    fuse_linear_bn,
    fuse_convtranspose_bn,
)

# TODO: rename to be more explict, e.g. qat_conv_relu
_ConvMetadata = namedtuple(
    "_ConvMetadata",
    ["root", "transpose", "bn", "reference", "transpose_reference",
     "fused_conv_relu", "fused_conv_bn", "fused_conv_bn_relu",
     "qat", "relu_qat", "bn_qat", "bn_relu_qat",
     "func"])
_Conv1dMetadata = _ConvMetadata(
    nn.Conv1d, nn.ConvTranspose1d, nn.BatchNorm1d, nnqr.Conv1d, nnqr.ConvTranspose1d,
    nni.ConvReLU1d, nni.ConvBn1d, nni.ConvBnReLU1d,
    nnqat.Conv1d, nniqat.ConvReLU1d, nniqat.ConvBn1d, nniqat.ConvBnReLU1d,
    F.conv1d)
_Conv2dMetadata = _ConvMetadata(
    nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d, nnqr.Conv2d, nnqr.ConvTranspose2d,
    nni.ConvReLU2d, nni.ConvBn2d, nni.ConvBnReLU2d,
    nnqat.Conv2d, nniqat.ConvReLU2d, nniqat.ConvBn2d, nniqat.ConvBnReLU2d,
    F.conv2d)
_Conv3dMetadata = _ConvMetadata(
    nn.Conv3d, nn.ConvTranspose3d, nn.BatchNorm3d, nnqr.Conv3d, nnqr.ConvTranspose3d,
    nni.ConvReLU3d, nni.ConvBn3d, nni.ConvBnReLU3d,
    nnqat.Conv3d, nniqat.ConvReLU3d, nniqat.ConvBn3d, nniqat.ConvBnReLU3d,
    F.conv3d)

def _get_binary_op_configs(dtype_configs):
    binary_op_configs: List[Dict[str, Any]] = []
    num_tensor_args_to_observation_type_mapping = {
        # TODO: this is not used right now since we have extra check in prepare
        # will need to change this to NO_OBSERVER later after we implemented
        # Tensor dtype inference properly
        0: ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
        1: ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT,
        2: ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
    }
    for op_with_quantized_bop_scalar_variant in [
            operator.add, torch.add, operator.mul, torch.mul]:
        binary_op_configs.append({
            "pattern": (torch.nn.ReLU, op_with_quantized_bop_scalar_variant),
            "num_tensor_args_to_observation_type": num_tensor_args_to_observation_type_mapping,
            "dtype_configs": dtype_configs,
        })
        binary_op_configs.append({
            "pattern": (torch.nn.functional.relu, op_with_quantized_bop_scalar_variant),
            "num_tensor_args_to_observation_type": num_tensor_args_to_observation_type_mapping,
            "dtype_configs": dtype_configs,
        })
        binary_op_configs.append({
            "pattern": (torch.relu, op_with_quantized_bop_scalar_variant),
            "num_tensor_args_to_observation_type": num_tensor_args_to_observation_type_mapping,
            "dtype_configs": dtype_configs,
        })
        binary_op_configs.append({
            "pattern": op_with_quantized_bop_scalar_variant,
            "num_tensor_args_to_observation_type": num_tensor_args_to_observation_type_mapping,
            "dtype_configs": dtype_configs,
        })
    return binary_op_configs

def _get_linear_configs(dtype_configs):
    """
    Return all configs related to linear modules and ops.
    """
    observation_type = ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT
    linear_configs = []

    # (1) Single linear modules/functions
    # -------------------------------------
    # linear module
    linear_configs.append({
        # Please see README under this folder for pattern format
        "pattern": torch.nn.Linear,
        "observation_type": observation_type,
        "dtype_configs": dtype_configs,
        # the root module for the pattern, used to query the reference quantized module
        # e.g. for a (torch.nn.ReLU, torch.nn.Linear) pattern, the root will be torch.nn.Linear
        "root_module": torch.nn.Linear,
        # the corresponding reference quantized module for the root module
        "reference_quantized_module_for_root": nnqr.Linear,
        "qat_module": nnqat.Linear,
    })
    # linear qat module
    linear_configs.append({
        "pattern": nnqat.Linear,
        "observation_type": observation_type,
        "dtype_configs": dtype_configs,
        "root_module": torch.nn.Linear,
        "reference_quantized_module_for_root": nnqr.Linear,
    })
    # functional linear
    linear_configs.append({
        "pattern": torch.nn.functional.linear,
        "observation_type": observation_type,
        "dtype_configs": dtype_configs,
    })

    # (2) Linear + relu
    # -------------------
    # 2.1 linear module + relu fusion config
    # linear relu, linear module + relu module
    linear_configs.append({
        "pattern": (torch.nn.ReLU, torch.nn.Linear),
        "dtype_configs": dtype_configs,
        "fuser_method": reverse_sequential_wrapper2(nni.LinearReLU),
        "fused_module": nni.LinearReLU,
    })
    # linear relu, linear module + functional relu
    linear_configs.append({
        "pattern": (torch.nn.functional.relu, torch.nn.Linear),
        "dtype_configs": dtype_configs,
        "fuser_method": reverse_sequential_wrapper2(nni.LinearReLU),
        "fused_module": nni.LinearReLU,
    })

    # 2.2 linear module + relu, fused module configs
    # linear relu, fused module
    linear_configs.append({
        "pattern": nni.LinearReLU,
        "observation_type": observation_type,
        "dtype_configs": dtype_configs,
        "root_module": torch.nn.Linear,
        "reference_quantized_module_for_root": nnqr.Linear,
        "qat_module": nniqat.LinearReLU,
    })
    # linear relu, qat fused module
    linear_configs.append({
        "pattern": nniqat.LinearReLU,
        "observation_type": observation_type,
        "dtype_configs": dtype_configs,
        "root_module": torch.nn.Linear,
        "reference_quantized_module_for_root": nnqr.Linear,
    })
    # 2.3 functional linear + relu configs
    # linear relu, functional linear + relu module
    linear_configs.append({
        "pattern": (torch.nn.ReLU, F.linear),
        "observation_type": observation_type,
        "dtype_configs": dtype_configs,
    })
    # linear relu, functional linear + functional relu
    linear_configs.append({
        "pattern": (F.relu, F.linear),
        "observation_type": observation_type,
        "dtype_configs": dtype_configs,
    })

    # (3) Linear + batchnorm
    # ------------------------
    # 3.1 linear bn fusion
    linear_configs.append({
        "pattern": (nn.BatchNorm1d, nn.Linear),
        "dtype_configs": dtype_configs,
        "fuser_method": reverse2(fuse_linear_bn),
        "fused_module": nni.LinearBn1d,
    })

    # 3.2 linear bn fused
    # linear bn, fused module
    linear_configs.append({
        "pattern": nni.LinearBn1d,
        "observation_type": observation_type,
        "dtype_configs": dtype_configs,
        "root_module": torch.nn.Linear,
        "reference_quantized_module_for_root": nnqr.Linear,
        "qat_module": nniqat.LinearBn1d,
    })
    # linear bn, qat fused module
    linear_configs.append({
        "pattern": nniqat.LinearBn1d,
        "observation_type": observation_type,
        "dtype_configs": dtype_configs,
        "root_module": torch.nn.Linear,
        "reference_quantized_module_for_root": nnqr.Linear,
    })
    return linear_configs

def _get_conv_configs(dtype_configs):
    """
    Return all configs related to conv modules and ops.
    """
    conv_configs = []
    observation_type = ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT
    for convs in [_Conv1dMetadata, _Conv2dMetadata, _Conv3dMetadata]:

        # (1) Single conv modules/functions
        # -----------------------------------
        # conv module
        conv_configs.append({
            "pattern": convs.root,
            "observation_type": observation_type,
            "dtype_configs": dtype_configs,
            "root_module": convs.root,
            "reference_quantized_module_for_root": convs.reference,
            "qat_module": convs.qat,
        })
        # conv qat module
        conv_configs.append({
            "pattern": convs.qat,
            "observation_type": observation_type,
            "dtype_configs": dtype_configs,
            "root_module": convs.root,
            "reference_quantized_module_for_root": convs.reference,
        })
        # functional conv
        conv_configs.append({
            "pattern": convs.func,
            "observation_type": observation_type,
            "dtype_configs": dtype_configs,
        })

        # (2) Conv + relu
        # -----------------
        # 2.1 conv module + relu fusion configs
        # conv relu fusion, conv module + relu module
        conv_configs.append({
            "pattern": (torch.nn.ReLU, convs.root),
            "dtype_configs": dtype_configs,
            "fuser_method": reverse_sequential_wrapper2(convs.fused_conv_relu),
            "fused_module": convs.fused_conv_relu,
        })
        # conv relu fusion, conv module + functional relu
        conv_configs.append({
            "pattern": (F.relu, convs.root),
            "dtype_configs": dtype_configs,
            "fuser_method": reverse_sequential_wrapper2(convs.fused_conv_relu),
            "fused_module": convs.fused_conv_relu,
        })
        # 2.2 conv module + relu fused module configs
        # conv relu, fused module
        conv_configs.append({
            "pattern": convs.fused_conv_relu,
            "observation_type": observation_type,
            "dtype_configs": dtype_configs,
            "root_module": convs.root,
            "reference_quantized_module_for_root": convs.reference,
            "qat_module": convs.relu_qat,
        })
        # conv relu, qat fused module
        conv_configs.append({
            "pattern": convs.relu_qat,
            "observation_type": observation_type,
            "dtype_configs": dtype_configs,
            "root_module": convs.root,
            "reference_quantized_module_for_root": convs.reference,
        })
        # 2.3 functional conv + relu configs
        # conv relu, functional conv + relu module
        conv_configs.append({
            "pattern": (torch.nn.ReLU, convs.func),
            "observation_type": observation_type,
            "dtype_configs": dtype_configs,
        })
        # conv relu, functional conv + functional relu
        conv_configs.append({
            "pattern": (F.relu, convs.func),
            "observation_type": observation_type,
            "dtype_configs": dtype_configs,
        })

        # fused conv relu
        conv_configs.append({
            "pattern": convs.fused_conv_relu,
            "dtype_configs": dtype_configs,
            "qat_module": convs.relu_qat,
        })

        conv_configs.append({
            "pattern": convs.relu_qat,
            "dtype_configs": dtype_configs,
            "root_module": convs.root,
            "reference_quantized_module_for_root": convs.reference,
        })

        # (3) Conv + batchnorm (+ relu)
        # -------------------------------
        # 3.1 conv bn fusion configs
        # conv + bn fusion
        conv_configs.append({
            "pattern": (convs.bn, convs.root),
            "dtype_configs": dtype_configs,
            "fuser_method": reverse2(fuse_conv_bn),
            "fused_module": convs.fused_conv_bn,
        })
        # conv + bn + relu module fusion
        conv_configs.append({
            "pattern": (nn.ReLU, (convs.bn, convs.root)),
            "dtype_configs": dtype_configs,
            "fuser_method": reverse3(fuse_conv_bn_relu),
            "fused_module": convs.fused_conv_bn_relu,
        })
        # conv + bn + relu functional fusion
        conv_configs.append({
            "pattern": (F.relu, (convs.bn, convs.root)),
            "dtype_configs": dtype_configs,
            "root_module": convs.root,
            "fuser_method": reverse3(fuse_conv_bn_relu),
            "fused_module": convs.fused_conv_bn_relu,
        })
        # TODO: we can add fusion for torch.relu as well

        # 3.2 conv + bn (+ relu) fused module configs
        # fused conv bn
        conv_configs.append({
            "pattern": convs.fused_conv_bn,
            "dtype_configs": dtype_configs,
            "qat_module": convs.bn_qat,
        })

        # fused conv bn relu
        conv_configs.append({
            "pattern": convs.fused_conv_bn_relu,
            "dtype_configs": dtype_configs,
            "qat_module": convs.bn_relu_qat,
        })

        # conv bn, qat fused module
        conv_configs.append({
            "pattern": convs.bn_qat,
            "observation_type": observation_type,
            "dtype_configs": dtype_configs,
            "root_module": convs.root,
            "reference_quantized_module_for_root": convs.reference,
        })
        # conv bn relu, qat fused module
        conv_configs.append({
            "pattern": convs.bn_relu_qat,
            "observation_type": observation_type,
            "dtype_configs": dtype_configs,
            "root_module": convs.root,
            "reference_quantized_module_for_root": convs.reference,
        })

        # (4) conv transpose and its fusion
        # 4.1 conv transpose config
        conv_configs.append({
            "pattern": convs.transpose,
            "dtype_configs": dtype_configs,
            "root_module": convs.transpose,
            "reference_quantized_module_for_root": convs.transpose_reference,
        })

        # 4.2 conv transpose + bn fusion
        conv_configs.append({
            "pattern": (convs.bn, convs.transpose),
            "dtype_configs": dtype_configs,
            "fuser_method": reverse2(fuse_convtranspose_bn),
            "root_module": convs.transpose,
            "reference_quantized_module_for_root": convs.transpose_reference,
        })

    return conv_configs

def _get_share_qparams_op_configs(dtype_configs):
    """ Get the operator config for the operators that works for both float and quantized input
    if input is quantized, the output Tensor shares the same quantization parameter
    with input.
    Example operator: avgpool2d, reshape, transpose, maxpool2d
    Example observed operator:
    observer_0 - avgpool2d - observer_0 (same observer instance as input)
    """

    def _get_share_qprams_op_backend_config(op):
        return {
            "pattern": op,
            "observation_type": ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT,
            "dtype_configs": dtype_configs,
        }

    share_qparams_ops = [
        torch.nn.AdaptiveAvgPool1d,
        torch.nn.AdaptiveAvgPool2d,
        torch.nn.AdaptiveAvgPool3d,
        torch.nn.AvgPool1d,
        torch.nn.AvgPool2d,
        torch.nn.AvgPool3d,
        torch.nn.Hardtanh,
        torch.nn.Identity,
        torch.nn.MaxPool1d,
        torch.nn.MaxPool2d,
        torch.nn.MaxPool3d,
        torch.nn.ReLU,
        torch.nn.ReLU6,
        torch.adaptive_avg_pool1d,
        torch.nn.functional.adaptive_avg_pool2d,
        torch.nn.functional.adaptive_avg_pool3d,
        torch.nn.functional.hardtanh,
        torch.nn.functional.hardtanh_,
        torch.nn.functional.interpolate,
        torch.nn.functional.max_pool1d,
        torch.nn.functional.max_pool2d,
        torch.nn.functional.max_pool3d,
        torch.nn.functional.relu,
        torch.nn.functional.relu6,
        torch.avg_pool1d,
        torch._C._nn.avg_pool2d,
        torch._C._nn.avg_pool3d,
        torch.clamp,
        torch.flatten,
        torch.mean,
        torch.repeat_interleave,
        torch.transpose,
        torch.squeeze,
        torch.stack,
        torch.unsqueeze,
        operator.floordiv,
        "contiguous",
        "clamp",
        "detach",
        "detach_",
        "mean",
        "permute",
        "repeat",
        "repeat_interleave",
        "reshape",
        "resize_",
        "relu",
        "relu_",
        "shape",
        "size",
        "squeeze",
        "squeeze_",
        "transpose",
        "unsqueeze",
        "unsqueeze_",
        "view"
    ]
    return [_get_share_qprams_op_backend_config(op) for op in share_qparams_ops]

__all__ = [
    "_get_binary_op_configs",
    "_get_linear_configs",
    "_get_conv_configs",
    "_get_share_qparams_op_configs",
]
