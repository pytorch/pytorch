# TODO: rename executorch to qnnpack_executorch since executorch is a general runtime
# not a specific backend

import operator

import torch
import torch.ao.nn.qat as nnqat
import torch.ao.nn.quantized.reference as nnqr
import torch.nn as nn
import torch.nn.functional as F
from torch.ao.quantization.fuser_method_mappings import (
    _sequential_wrapper2,
    fuse_conv_bn,
    fuse_conv_bn_relu,
)

from ._common_operator_config_utils import _Conv2dMetadata
from .backend_config import (
    BackendConfig,
    BackendPatternConfig,
    DTypeConfig,
    DTypeWithConstraints,
    ObservationType,
)
from .qnnpack import (
    qnnpack_default_op_qint8_symmetric_dtype_config,
    qnnpack_weighted_op_qint8_symmetric_dtype_config,
)


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

executorch_default_dynamic_quint8_dtype_config = DTypeConfig(
    input_dtype=torch.quint8,
    output_dtype=torch.float,
    weight_dtype=torch.qint8,
    bias_dtype=torch.float,
    is_dynamic=True,
)

executorch_act_qint8_scale_min_2_neg_12 = DTypeWithConstraints(
    dtype=torch.qint8,
    scale_min_lower_bound=2**-12,
)

executorch_weight_qint8_neg_127_to_127_scale_min_2_neg_12 = DTypeWithConstraints(
    dtype=torch.qint8,
    quant_min_lower_bound=-127,
    quant_max_upper_bound=127,
    scale_min_lower_bound=2**-12,
)

executorch_default_dynamic_qint8_dtype_config = DTypeConfig(
    input_dtype=executorch_act_qint8_scale_min_2_neg_12,
    output_dtype=torch.float,
    weight_dtype=executorch_weight_qint8_neg_127_to_127_scale_min_2_neg_12,
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

executorch_weight_only_quint8_dtype_config = DTypeConfig(
    input_dtype=torch.float,
    output_dtype=torch.float,
    weight_dtype=torch.quint8,
)


# =============================
# |  BACKEND PATTERN CONFIGS  |
# =============================


def _get_linear_configs() -> list[BackendPatternConfig]:
    """
    Return all configs related to linear modules and ops.
    """
    observation_type = ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT
    dtype_configs = [
        qnnpack_weighted_op_qint8_symmetric_dtype_config,
        executorch_weighted_op_int8_dtype_config,
        executorch_default_dynamic_quint8_dtype_config,
        executorch_default_dynamic_qint8_dtype_config,
        executorch_default_dynamic_float16_dtype_config,
    ]
    linear_configs: list[BackendPatternConfig] = []
    # linear module
    linear_configs.append(
        BackendPatternConfig(torch.nn.Linear)
        .set_observation_type(observation_type)  # noqa: E131
        .set_dtype_configs(dtype_configs)
        .set_root_module(torch.nn.Linear)
        .set_reference_quantized_module(nnqr.Linear)
        .set_qat_module(nnqat.Linear)
    )
    # linear qat module
    linear_configs.append(
        BackendPatternConfig(nnqat.Linear)
        .set_observation_type(observation_type)  # noqa: E131
        .set_dtype_configs(dtype_configs)
        .set_root_module(torch.nn.Linear)
        .set_reference_quantized_module(nnqr.Linear)
    )
    # functional linear
    linear_configs.append(
        BackendPatternConfig(torch.nn.functional.linear)
        .set_observation_type(observation_type)  # noqa: E131
        .set_dtype_configs(dtype_configs)
        ._set_input_type_to_index({"weight": 1, "bias": 2})
    )
    return linear_configs


def _get_conv_configs() -> list[BackendPatternConfig]:
    """
    Return all configs related to conv modules and ops.
    """
    observation_type = ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT
    dtype_configs = [
        qnnpack_weighted_op_qint8_symmetric_dtype_config,
        executorch_weighted_op_int8_dtype_config,
    ]
    conv_configs = []
    for convs in [_Conv2dMetadata]:
        # (1) Single conv modules/functions
        # -----------------------------------
        # conv module
        conv_configs.append(
            BackendPatternConfig(convs.root)
            .set_observation_type(observation_type)  # noqa: E131
            .set_dtype_configs(dtype_configs)
            .set_root_module(convs.root)
            .set_reference_quantized_module(convs.reference)
            .set_qat_module(convs.qat)
        )
        # conv qat module
        conv_configs.append(
            BackendPatternConfig(convs.qat)
            .set_observation_type(observation_type)  # noqa: E131
            .set_dtype_configs(dtype_configs)
            .set_root_module(convs.root)
            .set_reference_quantized_module(convs.reference)
        )
        # functional conv
        conv_configs.append(
            BackendPatternConfig(convs.func)
            .set_observation_type(observation_type)  # noqa: E131
            .set_dtype_configs(dtype_configs)
            ._set_input_type_to_index({"weight": 1, "bias": 2})
        )

        # (2) Conv + relu
        # -----------------------------------
        # conv module + relu module
        conv_configs.append(
            BackendPatternConfig((convs.root, nn.ReLU))
            .set_dtype_configs(dtype_configs)  # noqa: E131
            .set_fuser_method(_sequential_wrapper2(convs.fused_conv_relu))
            .set_fused_module(convs.fused_conv_relu)
        )
        # conv module + functional relu
        conv_configs.append(
            BackendPatternConfig((convs.root, F.relu))
            .set_dtype_configs(dtype_configs)  # noqa: E131
            .set_fuser_method(_sequential_wrapper2(convs.fused_conv_relu))
            .set_fused_module(convs.fused_conv_relu)
        )
        # fused conv relu module
        conv_configs.append(
            BackendPatternConfig(convs.fused_conv_relu)
            .set_observation_type(observation_type)  # noqa: E131
            .set_dtype_configs(dtype_configs)
            .set_root_module(convs.root)
            .set_reference_quantized_module(convs.reference)
            .set_qat_module(convs.relu_qat)
        )
        # conv relu, qat fused module
        conv_configs.append(
            BackendPatternConfig(convs.relu_qat)
            .set_observation_type(observation_type)  # noqa: E131
            .set_dtype_configs(dtype_configs)
            .set_root_module(convs.root)
            .set_reference_quantized_module(convs.reference)
        )
        # functional conv + relu module
        conv_configs.append(
            BackendPatternConfig((convs.func, nn.ReLU))
            .set_observation_type(observation_type)  # noqa: E131
            .set_dtype_configs(dtype_configs)
        )
        # functional conv + functional relu
        conv_configs.append(
            BackendPatternConfig((convs.func, F.relu))
            .set_observation_type(observation_type)  # noqa: E131
            .set_dtype_configs(dtype_configs)
        )
        # fused conv relu
        conv_configs.append(
            BackendPatternConfig(convs.fused_conv_relu)
            .set_dtype_configs(dtype_configs)  # noqa: E131
            .set_qat_module(convs.relu_qat)
        )

        conv_configs.append(
            BackendPatternConfig(convs.relu_qat)
            .set_dtype_configs(dtype_configs)  # noqa: E131
            .set_root_module(convs.root)
            .set_reference_quantized_module(convs.reference)
        )

        # (3) Conv + batchnorm (+ relu)
        # -------------------------------
        # conv + batchnorm (+ relu)
        conv_configs.append(
            BackendPatternConfig((convs.root, convs.bn))
            .set_dtype_configs(dtype_configs)  # noqa: E131
            .set_fuser_method(fuse_conv_bn)
            .set_fused_module(convs.fused_conv_bn)
        )
        # conv + bn + relu module fusion
        conv_configs.append(
            BackendPatternConfig((convs.root, convs.bn, nn.ReLU))
            .set_dtype_configs(dtype_configs)  # noqa: E131
            .set_fuser_method(fuse_conv_bn_relu)
            .set_fused_module(convs.fused_conv_bn_relu)
        )
        # conv + bn + relu functional fusion
        conv_configs.append(
            BackendPatternConfig((convs.root, convs.bn, F.relu))
            .set_dtype_configs(dtype_configs)  # noqa: E131
            .set_root_module(convs.root)
            .set_fuser_method(fuse_conv_bn_relu)
            .set_fused_module(convs.fused_conv_bn_relu)
        )
        # TODO: we can add fusion for torch.relu as well
        # 3.2 conv + bn (+ relu) fused module configs
        # fused conv bn
        conv_configs.append(
            BackendPatternConfig(convs.fused_conv_bn)
            .set_dtype_configs(dtype_configs)  # noqa: E131
            .set_qat_module(convs.bn_qat)
        )

        # fused conv bn relu
        conv_configs.append(
            BackendPatternConfig(convs.fused_conv_bn_relu)
            .set_dtype_configs(dtype_configs)  # noqa: E131
            .set_qat_module(convs.bn_relu_qat)
        )

        # conv bn, qat fused module
        conv_configs.append(
            BackendPatternConfig(convs.bn_qat)
            .set_observation_type(observation_type)  # noqa: E131
            .set_dtype_configs(dtype_configs)
            .set_root_module(convs.root)
            .set_reference_quantized_module(convs.reference)
        )
        # conv bn relu, qat fused module
        conv_configs.append(
            BackendPatternConfig(convs.bn_relu_qat)
            .set_observation_type(observation_type)  # noqa: E131
            .set_dtype_configs(dtype_configs)
            .set_root_module(convs.root)
            .set_reference_quantized_module(convs.reference)
        )
    return conv_configs


def _get_binary_ops_configs() -> list[BackendPatternConfig]:
    """
    Return all configs related to binary ops.
    """
    dtype_configs = [
        qnnpack_default_op_qint8_symmetric_dtype_config,
        executorch_weighted_op_int8_dtype_config,
    ]
    num_tensor_args_to_observation_type_mapping = {
        # TODO: this is not used right now since we have extra check in prepare
        # will need to change this to NO_OBSERVER later after we implemented
        # Tensor dtype inference properly
        0: ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
        1: ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT,
        2: ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
    }
    binary_op_configs: list[BackendPatternConfig] = []
    for op in [
        operator.add,
        torch.add,
        operator.sub,
        torch.sub,
        operator.mul,
        torch.mul,
    ]:
        bop_patterns = [
            (op, torch.nn.ReLU),
            (op, torch.nn.functional.relu),
            (op, torch.relu),
            op,
        ]
        binary_op_configs.extend(
            BackendPatternConfig(bop_pattern)
            .set_dtype_configs(dtype_configs)  # noqa: E131
            ._set_num_tensor_args_to_observation_type(
                num_tensor_args_to_observation_type_mapping
            )
            for bop_pattern in bop_patterns
        )
    return binary_op_configs


def _get_share_qparams_ops_configs() -> list[BackendPatternConfig]:
    """
    Return the operator configs for the operators that works for both float and quantized
    input if input is quantized, the output Tensor shares the same quantization parameter
    with input.

    Example operator: avgpool2d, reshape, transpose, maxpool2d
    Example observed operator:
    observer_0 - avgpool2d - observer_0 (same observer instance as input)
    """
    observation_type = ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT
    dtype_configs = [
        qnnpack_default_op_qint8_symmetric_dtype_config,
        executorch_default_op_quint8_dtype_config,
    ]
    share_qparams_ops = [
        torch.nn.Flatten,
        F.adaptive_avg_pool2d,
        F.elu,
        F.hardtanh,
        F.max_pool2d,
        F.pad,
        F.relu,
        F.relu6,
        F.leaky_relu,
        F.leaky_relu_,
        torch.nn.AdaptiveAvgPool2d,
        torch.nn.ConstantPad2d,
        torch.nn.ELU,
        torch.nn.MaxPool2d,
        torch.nn.ReLU6,
        torch.nn.Hardtanh,
        torch.nn.LeakyReLU,
        torch.clamp,
        torch.flatten,
        torch.mean,
        torch.permute,
        torch.permute_copy,
        torch.squeeze,
        "clamp",
        "mean",
        "permute",
        "reshape",
        "relu",
        "relu_",
        "squeeze",
        "squeeze_",
        "leaky_relu",
    ]
    share_qparams_op_configs: list[BackendPatternConfig] = [
        BackendPatternConfig(op)
        .set_observation_type(observation_type)  # noqa: E131
        .set_dtype_configs(dtype_configs)
        for op in share_qparams_ops
    ]
    return share_qparams_op_configs


def _get_bn_configs() -> list[BackendPatternConfig]:
    """
    Return all configs related to batchnorm.
    """
    observation_type = ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT
    dtype_configs = [
        qnnpack_default_op_qint8_symmetric_dtype_config,
        executorch_default_op_quint8_dtype_config,
    ]
    bn_configs = []
    bn_configs.append(
        BackendPatternConfig(nn.BatchNorm2d)
        .set_observation_type(observation_type)  # noqa: E131
        .set_dtype_configs(dtype_configs)
    )
    return bn_configs


def _get_cat_configs() -> list[BackendPatternConfig]:
    dtype_configs = [
        qnnpack_default_op_qint8_symmetric_dtype_config,
        executorch_default_op_quint8_dtype_config,
    ]
    cat_configs = []
    cat_configs.append(
        BackendPatternConfig(torch.cat)
        .set_observation_type(ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT)
        .set_dtype_configs(dtype_configs)
    )
    cat_configs.append(
        BackendPatternConfig(torch.concat)
        .set_observation_type(ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT)
        .set_dtype_configs(dtype_configs)
    )
    cat_configs.append(
        BackendPatternConfig(torch.concatenate)
        .set_observation_type(ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT)
        .set_dtype_configs(dtype_configs)
    )
    return cat_configs


def _get_embedding_op_configs() -> list[BackendPatternConfig]:
    dtype_configs = [
        executorch_weight_only_quint8_dtype_config,
    ]
    embedding_op_configs = []
    for embedding_op, qat_embedding_op, ref_embedding_op in [
        (nn.Embedding, nnqat.Embedding, nnqr.Embedding),
        (nn.EmbeddingBag, nnqat.EmbeddingBag, nnqr.EmbeddingBag),
    ]:
        embedding_op_configs.append(
            BackendPatternConfig(embedding_op)
            .set_observation_type(
                ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT
            )  # noqa: E131
            .set_dtype_configs(dtype_configs)
            .set_qat_module(qat_embedding_op)
            .set_root_module(embedding_op)
            .set_reference_quantized_module(ref_embedding_op)
        )
        # config for qat op
        embedding_op_configs.append(
            BackendPatternConfig(qat_embedding_op)
            .set_observation_type(
                ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT
            )  # noqa: E131
            .set_dtype_configs(dtype_configs)
            .set_root_module(embedding_op)
            .set_reference_quantized_module(ref_embedding_op)
        )

        # config for functional embedding
        embedding_op_configs.append(
            BackendPatternConfig(torch.nn.functional.embedding)
            .set_observation_type(
                ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT
            )  # noqa: E131
            .set_dtype_configs(dtype_configs)
            ._set_input_type_to_index({"weight": 1})
        )
    return embedding_op_configs


# =====================
# |  BACKEND CONFIGS  |
# =====================


def get_executorch_backend_config() -> BackendConfig:
    """
    Return the `BackendConfig` for backends PyTorch lowers to through the Executorch stack.
    """
    return (
        BackendConfig("executorch")
        .set_backend_pattern_configs(_get_linear_configs())
        .set_backend_pattern_configs(_get_conv_configs())
        .set_backend_pattern_configs(_get_binary_ops_configs())
        .set_backend_pattern_configs(_get_share_qparams_ops_configs())
        .set_backend_pattern_configs(_get_bn_configs())
        .set_backend_pattern_configs(_get_cat_configs())
        .set_backend_pattern_configs(_get_embedding_op_configs())
    )
