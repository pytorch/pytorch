import torch
from typing import List
from ._common_operator_config_utils import (
    _get_binary_op_configs,
    _get_bn_configs,
    _get_cat_config,
    _get_conv_configs,
    _get_default_op_configs,
    _get_embedding_op_configs,
    _get_fixed_qparams_op_configs,
    _get_linear_configs,
    _get_rnn_op_configs,
    _get_share_qparams_op_configs,
)
from .backend_config import BackendConfig, BackendPatternConfig, DTypeConfig, ObservationType


# ===================
# |  DTYPE CONFIGS  |
# ===================

# TODO: For now, these DTypeConfigs are identical to the ones defined in native.py
# In the future, once we support specifying quant_min/quant_max and scale_min/scale_max,
# these will diverge. In particular, for FBGEMM, we will restrict the activation quantized
# values to within [0, 127].

fbgemm_weighted_op_int8_dtype_config = DTypeConfig(
    input_dtype=torch.quint8,
    output_dtype=torch.quint8,
    weight_dtype=torch.qint8,
    bias_dtype=torch.float,
)

fbgemm_default_op_quint8_dtype_config = DTypeConfig(
    input_dtype=torch.quint8,
    output_dtype=torch.quint8,
)

fbgemm_default_op_fp16_dtype_config = DTypeConfig(
    input_dtype=torch.float16,
    output_dtype=torch.float16,
    weight_dtype=torch.float16,
    bias_dtype=torch.float16,
)

fbgemm_default_dynamic_int8_dtype_config = DTypeConfig(
    input_dtype=torch.quint8,
    output_dtype=torch.float,
    weight_dtype=torch.qint8,
    bias_dtype=torch.float,
    is_dynamic=True,
)

fbgemm_default_dynamic_float16_dtype_config = DTypeConfig(
    input_dtype=torch.float16,
    output_dtype=torch.float,
    weight_dtype=torch.float16,
    bias_dtype=torch.float,
    is_dynamic=True,
)

fbgemm_weight_only_quint8_dtype_config = DTypeConfig(
    input_dtype=torch.float,
    output_dtype=torch.float,
    weight_dtype=torch.quint8,
)

fbgemm_weight_only_quint4x2_dtype_config = DTypeConfig(
    input_dtype=torch.float,
    output_dtype=torch.float,
    weight_dtype=torch.quint4x2,
)


# ===================
# | FBGEMM OP CONFIGS |
# ===================


def _get_fbgemm_op_configs(dtype_configs: List[DTypeConfig]) -> List[BackendPatternConfig]:
    configs = []
    fbgemm_ops = [
        torch.nn.InstanceNorm1d,
        torch.nn.InstanceNorm2d,
        torch.nn.InstanceNorm3d,
        torch.nn.LayerNorm,
    ]

    for op in fbgemm_ops:
        configs.append(
            BackendPatternConfig(op)
            .set_observation_type(ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT)  # noqa: E131
            .set_dtype_configs(dtype_configs)
        )

    configs.append(
        BackendPatternConfig(torch.nn.functional.layer_norm)
        .set_observation_type(ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT)  # noqa: E131
        .set_dtype_configs(dtype_configs)
        ._set_input_type_to_index({"weight": 2, "bias": 3})
    )

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

# =====================
# |  BACKEND CONFIGS  |
# =====================

def get_fbgemm_backend_config() -> BackendConfig:
    """
    Return the `BackendConfig` for PyTorch's native FBGEMM backend.
    """
    conv_dtype_configs = [fbgemm_weighted_op_int8_dtype_config]
    linear_dtype_configs = [
        fbgemm_weighted_op_int8_dtype_config,
        fbgemm_default_dynamic_int8_dtype_config,
        fbgemm_default_dynamic_float16_dtype_config,
    ]
    binary_op_dtype_configs = [fbgemm_weighted_op_int8_dtype_config]
    default_op_dtype_configs = [fbgemm_default_op_quint8_dtype_config]
    fbgemm_op_dtype_configs = [fbgemm_default_op_quint8_dtype_config]
    fixed_qparams_op_dtype_configs = [fbgemm_weighted_op_int8_dtype_config]
    share_qparams_op_dtype_configs = [fbgemm_default_op_quint8_dtype_config]
    rnn_op_dtype_configs = [
        fbgemm_default_dynamic_int8_dtype_config,
        fbgemm_default_dynamic_float16_dtype_config,
    ]
    embedding_op_dtype_configs = [
        fbgemm_weight_only_quint8_dtype_config,
        fbgemm_weight_only_quint4x2_dtype_config,
    ]
    return BackendConfig("fbgemm") \
        .set_backend_pattern_configs(_get_fbgemm_op_configs(fbgemm_op_dtype_configs)) \
        .set_backend_pattern_configs(_get_conv_configs(conv_dtype_configs)) \
        .set_backend_pattern_configs(_get_linear_configs(linear_dtype_configs)) \
        .set_backend_pattern_configs(_get_binary_op_configs(binary_op_dtype_configs)) \
        .set_backend_pattern_config(_get_cat_config(default_op_dtype_configs)) \
        .set_backend_pattern_configs(_get_default_op_configs(default_op_dtype_configs)) \
        .set_backend_pattern_configs(_get_fixed_qparams_op_configs(fixed_qparams_op_dtype_configs)) \
        .set_backend_pattern_configs(_get_share_qparams_op_configs(share_qparams_op_dtype_configs)) \
        .set_backend_pattern_configs(_get_bn_configs(default_op_dtype_configs)) \
        .set_backend_pattern_configs(_get_rnn_op_configs(rnn_op_dtype_configs)) \
        .set_backend_pattern_configs(_get_embedding_op_configs(embedding_op_dtype_configs))

__all__ = [
    "get_fbgemm_backend_config",
]
