import operator
from typing import List
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.qat as nnqat
import torch.nn.quantized._reference as nnqr
from .backend_config import BackendConfig, BackendPatternConfig, DTypeConfig, ObservationType
from ._common_operator_config_utils import _Conv2dMetadata
from ..fuser_method_mappings import reverse_sequential_wrapper2


__all__ = [
    "get_executorch_backend_config",
]


# ===================
# |  DTYPE CONFIGS  |
# ===================

executorch_weighted_op_int8_dtype_config = DTypeConfig(
    input_dtype=torch.quint8,
    output_dtype=torch.quint8,
    weight_dtype=torch.qint8,
    bias_dtype=torch.float,
)

executorch_default_op_quint8_dtype_config = DTypeConfig(
    input_dtype=torch.quint8,
    output_dtype=torch.quint8,
)

executorch_default_dynamic_int8_dtype_config = DTypeConfig(
    input_dtype=torch.quint8,
    output_dtype=torch.float,
    weight_dtype=torch.qint8,
    bias_dtype=torch.float,
    is_dynamic=True,
)

executorch_default_dynamic_float16_dtype_config = DTypeConfig(
    input_dtype=torch.float16,
    output_dtype=torch.float,
    weight_dtype=torch.float16,
    bias_dtype=torch.float,
    is_dynamic=True,
)


# =============================
# |  BACKEND PATTERN CONFIGS  |
# =============================

def _get_linear_configs() -> List[BackendPatternConfig]:
    """
    Return all configs related to linear modules and ops.
    """
    observation_type = ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT
    dtype_configs = [
        executorch_weighted_op_int8_dtype_config,
        executorch_default_dynamic_int8_dtype_config,
        executorch_default_dynamic_float16_dtype_config,
    ]
    linear_configs: List[BackendPatternConfig] = []
    # linear module
    linear_configs.append(
        BackendPatternConfig(torch.nn.Linear)
            .set_observation_type(observation_type)  # noqa: E131
            .set_dtype_configs(dtype_configs)
            .set_root_module(torch.nn.Linear)
            .set_reference_quantized_module(nnqr.Linear)
            .set_qat_module(nnqat.Linear))
    # functional linear
    linear_configs.append(
        BackendPatternConfig(torch.nn.functional.linear)
            .set_observation_type(observation_type)  # noqa: E131
            .set_dtype_configs(dtype_configs)
            ._set_input_type_to_index({"weight": 1, "bias": 2}))
    return linear_configs

def _get_conv_configs() -> List[BackendPatternConfig]:
    """
    Return all configs related to conv modules and ops.
    """
    observation_type = ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT
    dtype_configs = [executorch_weighted_op_int8_dtype_config]
    conv_configs = []
    for convs in [_Conv2dMetadata]:
        # conv module
        conv_configs.append(
            BackendPatternConfig(convs.root)
                .set_observation_type(observation_type)  # noqa: E131
                .set_dtype_configs(dtype_configs)
                .set_root_module(convs.root)
                .set_reference_quantized_module(convs.reference)
                .set_qat_module(convs.qat))
        # functional conv
        conv_configs.append(
            BackendPatternConfig(convs.func)
                .set_observation_type(observation_type)  # noqa: E131
                .set_dtype_configs(dtype_configs)
                ._set_input_type_to_index({"weight": 1, "bias": 2}))
        # conv module + relu module
        conv_configs.append(
            BackendPatternConfig((torch.nn.ReLU, convs.root))
                .set_dtype_configs(dtype_configs)  # noqa: E131
                .set_fuser_method(reverse_sequential_wrapper2(convs.fused_conv_relu))
                .set_fused_module(convs.fused_conv_relu))
        # conv module + functional relu
        conv_configs.append(
            BackendPatternConfig((F.relu, convs.root))
                .set_dtype_configs(dtype_configs)  # noqa: E131
                .set_fuser_method(reverse_sequential_wrapper2(convs.fused_conv_relu))
                .set_fused_module(convs.fused_conv_relu))
        # fused conv relu module
        conv_configs.append(
            BackendPatternConfig(convs.fused_conv_relu)
                .set_observation_type(observation_type)  # noqa: E131
                .set_dtype_configs(dtype_configs)
                .set_root_module(convs.root)
                .set_reference_quantized_module(convs.reference)
                .set_qat_module(convs.relu_qat))
        # functional conv + relu module
        conv_configs.append(
            BackendPatternConfig((torch.nn.ReLU, convs.func))
                .set_observation_type(observation_type)  # noqa: E131
                .set_dtype_configs(dtype_configs))
        # functional conv + functional relu
        conv_configs.append(
            BackendPatternConfig((F.relu, convs.func))
                .set_observation_type(observation_type)  # noqa: E131
                .set_dtype_configs(dtype_configs))
    return conv_configs

def _get_binary_ops_configs() -> List[BackendPatternConfig]:
    """
    Return all configs related to binary ops.
    """
    dtype_configs = [executorch_weighted_op_int8_dtype_config]
    num_tensor_args_to_observation_type_mapping = {
        # TODO: this is not used right now since we have extra check in prepare
        # will need to change this to NO_OBSERVER later after we implemented
        # Tensor dtype inference properly
        0: ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
        1: ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT,
        2: ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
    }
    binary_op_configs: List[BackendPatternConfig] = []
    for op in [operator.add, torch.add]:
        binary_op_configs.append(
            BackendPatternConfig(op)
                .set_dtype_configs(dtype_configs)  # noqa: E131
                ._set_num_tensor_args_to_observation_type(num_tensor_args_to_observation_type_mapping))
    return binary_op_configs

def _get_share_qparams_ops_configs() -> List[BackendPatternConfig]:
    """
    Return the operator configs for the operators that works for both float and quantized
    input if input is quantized, the output Tensor shares the same quantization parameter
    with input.

    Example operator: avgpool2d, reshape, transpose, maxpool2d
    Example observed operator:
    observer_0 - avgpool2d - observer_0 (same observer instance as input)
    """
    observation_type = ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT
    dtype_configs = [executorch_default_op_quint8_dtype_config]
    share_qparams_ops = [
        F.adaptive_avg_pool2d,
        F.relu,
        F.relu6,
        torch.nn.AdaptiveAvgPool2d,
        torch.squeeze,
        "permute",
        "reshape",
        "relu",
        "relu_",
        "squeeze",
        "squeeze_",
    ]
    share_qparams_op_configs: List[BackendPatternConfig] = []
    for op in share_qparams_ops:
        share_qparams_op_configs.append(
            BackendPatternConfig(op)
                .set_observation_type(observation_type)  # noqa: E131
                .set_dtype_configs(dtype_configs))
    return share_qparams_op_configs

def _get_bn_configs() -> List[BackendPatternConfig]:
    """
    Return all configs related to batchnorm.
    """
    observation_type = ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT
    dtype_configs = [executorch_default_op_quint8_dtype_config]
    bn_configs = []
    bn_configs.append(
        BackendPatternConfig(nn.BatchNorm2d)
            .set_observation_type(observation_type)  # noqa: E131
            .set_dtype_configs(dtype_configs))
    return bn_configs


# =====================
# |  BACKEND CONFIGS  |
# =====================

def get_executorch_backend_config() -> BackendConfig:
    """
    Return the `BackendConfig` for backends PyTorch lowers to through the Executorch stack.
    """
    return BackendConfig("executorch") \
        .set_backend_pattern_configs(_get_linear_configs()) \
        .set_backend_pattern_configs(_get_conv_configs()) \
        .set_backend_pattern_configs(_get_binary_ops_configs()) \
        .set_backend_pattern_configs(_get_share_qparams_ops_configs()) \
        .set_backend_pattern_configs(_get_bn_configs())
