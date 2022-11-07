import copy
import operator
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.intrinsic as nni
import torch.ao.nn.intrinsic.qat as nniqat
import torch.nn.qat as nnqat
import torch.nn.quantized._reference as nnqr
from collections import namedtuple
from typing import List
from .backend_config import (
    BackendPatternConfig,
    DTypeConfig,
    ObservationType,
)
from ..fake_quantize import FixedQParamsFakeQuantize
from ..fuser_method_mappings import (
    reverse_sequential_wrapper2,
    reverse2,
    reverse3,
    fuse_conv_bn,
    fuse_conv_bn_relu,
    fuse_linear_bn,
    fuse_convtranspose_bn,
)
from ..observer import FixedQParamsObserver
from ..qconfig_mapping import _FIXED_QPARAMS_OP_TO_OBSERVER

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

def _get_binary_op_configs(dtype_configs: List[DTypeConfig]) -> List[BackendPatternConfig]:
    binary_op_configs: List[BackendPatternConfig] = []
    num_tensor_args_to_observation_type_mapping = {
        # TODO: this is not used right now since we have extra check in prepare
        # will need to change this to NO_OBSERVER later after we implemented
        # Tensor dtype inference properly
        0: ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
        1: ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT,
        2: ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
    }
    for op_with_quantized_bop_scalar_variant in [operator.add, torch.add, operator.mul, torch.mul]:
        bop_patterns = [
            (torch.nn.ReLU, op_with_quantized_bop_scalar_variant),
            (torch.nn.functional.relu, op_with_quantized_bop_scalar_variant),
            (torch.relu, op_with_quantized_bop_scalar_variant),
            op_with_quantized_bop_scalar_variant
        ]
        for bop_pattern in bop_patterns:
            binary_op_configs.append(
                BackendPatternConfig(bop_pattern)
                    .set_dtype_configs(dtype_configs)  # noqa: E131
                    ._set_num_tensor_args_to_observation_type(num_tensor_args_to_observation_type_mapping))
    # matmul
    binary_op_configs.append(
        BackendPatternConfig(torch.matmul)
        .set_dtype_configs(dtype_configs)  # noqa: E131
    )
    return binary_op_configs

def _get_linear_configs(dtype_configs: List[DTypeConfig]) -> List[BackendPatternConfig]:
    """
    Return all configs related to linear modules and ops.
    """
    observation_type = ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT
    linear_configs: List[BackendPatternConfig] = []

    # (1) Single linear modules/functions
    # -------------------------------------
    # linear module
    linear_configs.append(
        BackendPatternConfig(torch.nn.Linear)
            .set_observation_type(observation_type)  # noqa: E131
            .set_dtype_configs(dtype_configs)
            .set_root_module(torch.nn.Linear)
            .set_reference_quantized_module(nnqr.Linear)
            .set_qat_module(nnqat.Linear))
    # linear qat module
    linear_configs.append(
        BackendPatternConfig(nnqat.Linear)
            .set_observation_type(observation_type)  # noqa: E131
            .set_dtype_configs(dtype_configs)
            .set_root_module(torch.nn.Linear)
            .set_reference_quantized_module(nnqr.Linear))
    # functional linear
    linear_configs.append(
        BackendPatternConfig(torch.nn.functional.linear)
            .set_observation_type(observation_type)  # noqa: E131
            .set_dtype_configs(dtype_configs)
            ._set_input_type_to_index({"weight": 1, "bias": 2}))

    # (2) Linear + relu
    # -------------------
    # 2.1 linear module + relu fusion config
    # linear relu, linear module + relu module
    linear_configs.append(
        BackendPatternConfig((torch.nn.ReLU, torch.nn.Linear))
            .set_dtype_configs(dtype_configs)  # noqa: E131
            .set_fuser_method(reverse_sequential_wrapper2(nni.LinearReLU))
            .set_fused_module(nni.LinearReLU))
    # linear relu, linear module + functional relu
    linear_configs.append(
        BackendPatternConfig((torch.nn.functional.relu, torch.nn.Linear))
            .set_dtype_configs(dtype_configs)  # noqa: E131
            .set_fuser_method(reverse_sequential_wrapper2(nni.LinearReLU))
            .set_fused_module(nni.LinearReLU))

    # 2.2 linear module + relu, fused module configs
    # linear relu, fused module
    linear_configs.append(
        BackendPatternConfig(nni.LinearReLU)
            .set_observation_type(observation_type)  # noqa: E131
            .set_dtype_configs(dtype_configs)
            .set_root_module(torch.nn.Linear)
            .set_reference_quantized_module(nnqr.Linear)
            .set_qat_module(nniqat.LinearReLU))
    # linear relu, qat fused module
    linear_configs.append(
        BackendPatternConfig(nniqat.LinearReLU)
            .set_observation_type(observation_type)  # noqa: E131
            .set_dtype_configs(dtype_configs)
            .set_root_module(torch.nn.Linear)
            .set_reference_quantized_module(nnqr.Linear))
    # 2.3 functional linear + relu configs
    # linear relu, functional linear + relu module
    linear_configs.append(
        BackendPatternConfig((torch.nn.ReLU, F.linear))
            .set_observation_type(observation_type)  # noqa: E131
            .set_dtype_configs(dtype_configs))
    # linear relu, functional linear + functional relu
    linear_configs.append(
        BackendPatternConfig((F.relu, F.linear))
            .set_observation_type(observation_type)  # noqa: E131
            .set_dtype_configs(dtype_configs))

    # (3) Linear + batchnorm
    # ------------------------
    # 3.1 linear bn fusion
    linear_configs.append(
        BackendPatternConfig((nn.BatchNorm1d, nn.Linear))
            .set_dtype_configs(dtype_configs)  # noqa: E131
            .set_fuser_method(reverse2(fuse_linear_bn))
            .set_fused_module(nni.LinearBn1d))

    # 3.2 linear bn fused
    # linear bn, fused module
    linear_configs.append(
        BackendPatternConfig(nni.LinearBn1d)
            .set_observation_type(observation_type)  # noqa: E131
            .set_dtype_configs(dtype_configs)
            .set_root_module(torch.nn.Linear)
            .set_reference_quantized_module(nnqr.Linear)
            .set_qat_module(nniqat.LinearBn1d))
    # linear bn, qat fused module
    linear_configs.append(
        BackendPatternConfig(nniqat.LinearBn1d)
            .set_observation_type(observation_type)  # noqa: E131
            .set_dtype_configs(dtype_configs)
            .set_root_module(torch.nn.Linear)
            .set_reference_quantized_module(nnqr.Linear))
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
        conv_configs.append(
            BackendPatternConfig(convs.root)
                .set_observation_type(observation_type)  # noqa: E131
                .set_dtype_configs(dtype_configs)
                .set_root_module(convs.root)
                .set_reference_quantized_module(convs.reference)
                .set_qat_module(convs.qat))
        # conv qat module
        conv_configs.append(
            BackendPatternConfig(convs.qat)
                .set_observation_type(observation_type)  # noqa: E131
                .set_dtype_configs(dtype_configs)
                .set_root_module(convs.root)
                .set_reference_quantized_module(convs.reference))
        # functional conv
        conv_configs.append(
            BackendPatternConfig(convs.func)
                .set_observation_type(observation_type)  # noqa: E131
                .set_dtype_configs(dtype_configs)
                ._set_input_type_to_index({"weight": 1, "bias": 2}))

        # (2) Conv + relu
        # -----------------
        # 2.1 conv module + relu fusion configs
        # conv relu fusion, conv module + relu module
        conv_configs.append(
            BackendPatternConfig((torch.nn.ReLU, convs.root))
                .set_dtype_configs(dtype_configs)  # noqa: E131
                .set_fuser_method(reverse_sequential_wrapper2(convs.fused_conv_relu))
                .set_fused_module(convs.fused_conv_relu))
        # conv relu fusion, conv module + functional relu
        conv_configs.append(
            BackendPatternConfig((F.relu, convs.root))
                .set_dtype_configs(dtype_configs)  # noqa: E131
                .set_fuser_method(reverse_sequential_wrapper2(convs.fused_conv_relu))
                .set_fused_module(convs.fused_conv_relu))
        # 2.2 conv module + relu fused module configs
        # conv relu, fused module
        conv_configs.append(
            BackendPatternConfig(convs.fused_conv_relu)
                .set_observation_type(observation_type)  # noqa: E131
                .set_dtype_configs(dtype_configs)
                .set_root_module(convs.root)
                .set_reference_quantized_module(convs.reference)
                .set_qat_module(convs.relu_qat))
        # conv relu, qat fused module
        conv_configs.append(
            BackendPatternConfig(convs.relu_qat)
                .set_observation_type(observation_type)  # noqa: E131
                .set_dtype_configs(dtype_configs)
                .set_root_module(convs.root)
                .set_reference_quantized_module(convs.reference))
        # 2.3 functional conv + relu configs
        # conv relu, functional conv + relu module
        conv_configs.append(
            BackendPatternConfig((torch.nn.ReLU, convs.func))
                .set_observation_type(observation_type)  # noqa: E131
                .set_dtype_configs(dtype_configs))
        # conv relu, functional conv + functional relu
        conv_configs.append(
            BackendPatternConfig((F.relu, convs.func))
                .set_observation_type(observation_type)  # noqa: E131
                .set_dtype_configs(dtype_configs))

        # fused conv relu
        conv_configs.append(
            BackendPatternConfig(convs.fused_conv_relu)
                .set_dtype_configs(dtype_configs)  # noqa: E131
                .set_qat_module(convs.relu_qat))

        conv_configs.append(
            BackendPatternConfig(convs.relu_qat)
                .set_dtype_configs(dtype_configs)  # noqa: E131
                .set_root_module(convs.root)
                .set_reference_quantized_module(convs.reference))

        # (3) Conv + batchnorm (+ relu)
        # -------------------------------
        # 3.1 conv bn fusion configs
        # conv + bn fusion
        conv_configs.append(
            BackendPatternConfig((convs.bn, convs.root))
                .set_dtype_configs(dtype_configs)  # noqa: E131
                .set_fuser_method(reverse2(fuse_conv_bn))
                .set_fused_module(convs.fused_conv_bn))
        # conv + bn + relu module fusion
        conv_configs.append(
            BackendPatternConfig((nn.ReLU, (convs.bn, convs.root)))
                .set_dtype_configs(dtype_configs)  # noqa: E131
                .set_fuser_method(reverse3(fuse_conv_bn_relu))
                .set_fused_module(convs.fused_conv_bn_relu))
        # conv + bn + relu functional fusion
        conv_configs.append(
            BackendPatternConfig((F.relu, (convs.bn, convs.root)))
                .set_dtype_configs(dtype_configs)  # noqa: E131
                .set_root_module(convs.root)
                .set_fuser_method(reverse3(fuse_conv_bn_relu))
                .set_fused_module(convs.fused_conv_bn_relu))
        # TODO: we can add fusion for torch.relu as well

        # 3.2 conv + bn (+ relu) fused module configs
        # fused conv bn
        conv_configs.append(
            BackendPatternConfig(convs.fused_conv_bn)
                .set_dtype_configs(dtype_configs)  # noqa: E131
                .set_qat_module(convs.bn_qat))

        # fused conv bn relu
        conv_configs.append(
            BackendPatternConfig(convs.fused_conv_bn_relu)
                .set_dtype_configs(dtype_configs)  # noqa: E131
                .set_qat_module(convs.bn_relu_qat))

        # conv bn, qat fused module
        conv_configs.append(
            BackendPatternConfig(convs.bn_qat)
                .set_observation_type(observation_type)  # noqa: E131
                .set_dtype_configs(dtype_configs)
                .set_root_module(convs.root)
                .set_reference_quantized_module(convs.reference))
        # conv bn relu, qat fused module
        conv_configs.append(
            BackendPatternConfig(convs.bn_relu_qat)
                .set_observation_type(observation_type)  # noqa: E131
                .set_dtype_configs(dtype_configs)
                .set_root_module(convs.root)
                .set_reference_quantized_module(convs.reference))

        # (4) conv transpose and its fusion
        # 4.1 conv transpose config
        conv_configs.append(
            BackendPatternConfig(convs.transpose)
                .set_dtype_configs(dtype_configs)  # noqa: E131
                .set_root_module(convs.transpose)
                .set_reference_quantized_module(convs.transpose_reference))

        # 4.2 conv transpose + bn fusion
        conv_configs.append(
            BackendPatternConfig((convs.bn, convs.transpose))
                .set_dtype_configs(dtype_configs)  # noqa: E131
                .set_fuser_method(reverse2(fuse_convtranspose_bn))
                .set_root_module(convs.transpose)
                .set_reference_quantized_module(convs.transpose_reference))

    return conv_configs

def _get_cat_config(dtype_configs: List[DTypeConfig]) -> BackendPatternConfig:
    return BackendPatternConfig(torch.cat) \
        .set_observation_type(ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT) \
        .set_dtype_configs(dtype_configs)

def _get_ln_configs(dtype_configs: List[DTypeConfig]) -> List[BackendPatternConfig]:
    ln_configs = []
    ln_configs.append(
        BackendPatternConfig(torch.nn.LayerNorm)
        .set_observation_type(ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT)  # noqa: E131
        .set_dtype_configs(dtype_configs)
    )
    ln_configs.append(
        BackendPatternConfig(torch.nn.functional.layer_norm)
        .set_observation_type(ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT)  # noqa: E131
        .set_dtype_configs(dtype_configs)
        ._set_input_type_to_index({"weight": 2, "bias": 3})
    )
    return ln_configs

def _get_default_op_configs(dtype_configs: List[DTypeConfig]) -> List[BackendPatternConfig]:
    configs = []
    default_ops = [
        torch.nn.ELU,
        torch.nn.LeakyReLU,
        torch.nn.Hardswish,
        torch.nn.InstanceNorm1d,
        torch.nn.InstanceNorm2d,
        torch.nn.InstanceNorm3d,
        torch.nn.Dropout,
        torch.nn.PReLU,
        torch.nn.functional.elu,
        torch.nn.functional.hardswish,
        torch.nn.functional.leaky_relu,
        torch.nn.functional.dropout,
    ]
    for op in default_ops:
        configs.append(
            BackendPatternConfig(op)
                .set_observation_type(ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT)  # noqa: E131
                .set_dtype_configs(dtype_configs))

    configs.append(
        BackendPatternConfig(torch.nn.functional.group_norm)
        .set_observation_type(ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT)  # noqa: E131
        .set_dtype_configs(dtype_configs)
        ._set_input_type_to_index({"weight": 2, "bias": 3})
    )

    configs.append(
        BackendPatternConfig(torch.nn.functional.instance_norm)
        .set_observation_type(ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT)  # noqa: E131
        .set_dtype_configs(dtype_configs)
        ._set_input_type_to_index({"weight": 3, "bias": 4})
    )
    return configs

def _add_fixed_qparams_to_dtype_configs(
    dtype_configs: List[DTypeConfig],
    observer: FixedQParamsObserver,
) -> List[DTypeConfig]:
    """
    Return a copy of the list of DTypeConfigs where activations are subject to the fixed qparams
    constraints specified in the observer.

    If `scale_min_lower_bound` or `scale_max_upper_bound` is specified in the activations, throw an
    exception since these settings are incompatible with fixed qparams ops.
    """
    fixed_scale = observer.scale
    fixed_zero_point = observer.zero_point
    new_dtype_configs = []
    for dtype_config in dtype_configs:
        dc = copy.deepcopy(dtype_config)
        for dtype_with_constraints in [dc.input_dtype_with_constraints, dc.output_dtype_with_constraints]:
            dtype_with_constraints.fixed_scale = fixed_scale
            dtype_with_constraints.fixed_zero_point = fixed_zero_point
            if dtype_with_constraints.scale_min_lower_bound is not None:
                raise ValueError("scale_min_lower_bound is invalid for fixed qparams ops: %s" % dtype_config)
            if dtype_with_constraints.scale_max_upper_bound is not None:
                raise ValueError("scale_max_upper_bound is invalid for fixed qparams ops: %s" % dtype_config)
        new_dtype_configs.append(dc)
    return new_dtype_configs

def _get_fixed_qparams_op_configs(dtype_configs: List[DTypeConfig]) -> List[BackendPatternConfig]:
    fixed_qparams_op_configs = []
    for fixed_qparam_op, obs_ctr in _FIXED_QPARAMS_OP_TO_OBSERVER.items():
        # Specify fixed qparams constraints through the dtype configs
        obs = obs_ctr()
        assert isinstance(obs, FixedQParamsObserver)
        new_dtype_configs = _add_fixed_qparams_to_dtype_configs(dtype_configs, obs)
        fixed_qparams_op_configs.append(
            BackendPatternConfig(fixed_qparam_op)
                .set_observation_type(ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT)  # noqa: E131
                .set_dtype_configs(new_dtype_configs))
    return fixed_qparams_op_configs

def _get_share_qparams_op_configs(dtype_configs):
    """ Get the operator config for the operators that works for both float and quantized input
    if input is quantized, the output Tensor shares the same quantization parameter
    with input.
    Example operator: avgpool2d, reshape, transpose, maxpool2d
    Example observed operator:
    observer_0 - avgpool2d - observer_0 (same observer instance as input)
    """

    def _get_share_qprams_op_backend_config(op):
        return BackendPatternConfig(op) \
            .set_observation_type(ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT) \
            .set_dtype_configs(dtype_configs)

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

def _get_bn_configs(dtype_configs: List[DTypeConfig]) -> List[BackendPatternConfig]:
    """ Get configs related to batchnorm. """
    bn_configs = []
    bn_to_fused_bn = {
        torch.nn.BatchNorm2d: nni.BNReLU2d,
        torch.nn.BatchNorm3d: nni.BNReLU3d,
    }
    for bn in bn_to_fused_bn.keys():
        fused_bn = bn_to_fused_bn[bn]
        # bn module + relu module fusion config
        bn_configs.append(
            BackendPatternConfig((torch.nn.ReLU, bn))
                .set_dtype_configs(dtype_configs)  # noqa: E131
                .set_fuser_method(reverse_sequential_wrapper2(fused_bn))
                .set_fused_module(fused_bn))
        # bn module + F.relu fusion config
        bn_configs.append(
            BackendPatternConfig((torch.nn.functional.relu, bn))
                .set_dtype_configs(dtype_configs)  # noqa: E131
                .set_fuser_method(reverse_sequential_wrapper2(bn_to_fused_bn[bn]))
                .set_fused_module(fused_bn))
        bn_configs.append(
            BackendPatternConfig(bn)
                .set_observation_type(ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT)  # noqa: E131
                .set_dtype_configs(dtype_configs))

    # fused bn configs
    for fused_bn in bn_to_fused_bn.values():
        bn_configs.append(
            BackendPatternConfig(fused_bn)
                .set_observation_type(ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT)  # noqa: E131
                .set_dtype_configs(dtype_configs))
    return bn_configs

def _get_rnn_op_configs(dtype_configs: List[DTypeConfig]) -> List[BackendPatternConfig]:
    rnn_op_configs = []
    for rnn_op, ref_rnn_op in [
            (nn.GRUCell, nnqr.GRUCell),
            (nn.LSTMCell, nnqr.LSTMCell),
            (nn.RNNCell, nnqr.RNNCell),
            (nn.LSTM, nnqr.LSTM)
    ]:
        rnn_op_configs.append(
            BackendPatternConfig(rnn_op)
                .set_observation_type(ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT)  # noqa: E131
                .set_dtype_configs(dtype_configs)
                .set_root_module(rnn_op)
                .set_reference_quantized_module(ref_rnn_op))
    return rnn_op_configs

def _get_embedding_op_configs(dtype_configs: List[DTypeConfig]) -> List[BackendPatternConfig]:
    embedding_op_configs = []
    for embedding_op, qat_embedding_op, ref_embedding_op in [
            (nn.Embedding, nnqat.Embedding, nnqr.Embedding),
            (nn.EmbeddingBag, nnqat.EmbeddingBag, nnqr.EmbeddingBag),
    ]:
        embedding_op_configs.append(
            BackendPatternConfig(embedding_op)
                .set_observation_type(ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT)  # noqa: E131
                .set_dtype_configs(dtype_configs)
                .set_qat_module(qat_embedding_op)
                .set_root_module(embedding_op)
                .set_reference_quantized_module(ref_embedding_op)
                ._set_input_output_observed(False))  # This is temporary, and will be removed soon
        # config for qat op
        embedding_op_configs.append(
            BackendPatternConfig(qat_embedding_op)
                .set_observation_type(ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT)  # noqa: E131
                .set_dtype_configs(dtype_configs)
                .set_root_module(embedding_op)
                .set_reference_quantized_module(ref_embedding_op)
                ._set_input_output_observed(False))  # This is temporary, and will be removed soon
    return embedding_op_configs

__all__ = [
    "_get_binary_op_configs",
    "_get_linear_configs",
    "_get_conv_configs",
    "_get_share_qparams_op_configs",
]
