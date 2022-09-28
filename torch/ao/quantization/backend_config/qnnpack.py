import torch
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
from .backend_config import BackendConfig, DTypeConfig, DTypeWithConstraints


# ===================
# |  DTYPE CONFIGS  |
# ===================

def _get_activation_dtype_with_constraints(dtype: torch.dtype):
    return DTypeWithConstraints(dtype=dtype, scale_min_lower_bound=2 ** -12)

def _get_weight_dtype_with_constraints(dtype: torch.dtype):
    return DTypeWithConstraints(
        dtype=dtype,
        quant_min_lower_bound=-127,
        quant_max_upper_bound=127,
        scale_min_lower_bound=2 ** -12
    )

act_qint8_with_constraints = _get_activation_dtype_with_constraints(torch.qint8)
act_quint8_with_constraints = _get_activation_dtype_with_constraints(torch.quint8)
act_fp16_with_constraints = _get_activation_dtype_with_constraints(torch.float16)
act_fp_with_constraints = _get_activation_dtype_with_constraints(torch.float)
weight_qint8_with_constraints = _get_weight_dtype_with_constraints(torch.qint8)
weight_quint8_with_constraints = _get_weight_dtype_with_constraints(torch.quint8)
weight_quint4x2_with_constraints = _get_weight_dtype_with_constraints(torch.quint4x2)
weight_fp16_with_constraints = _get_weight_dtype_with_constraints(torch.float16)

# weighted op

qnnpack_weighted_op_qint8_dtype_config = DTypeConfig(
    input_dtype=act_qint8_with_constraints,
    output_dtype=act_qint8_with_constraints,
    weight_dtype=weight_qint8_with_constraints,
    bias_dtype=torch.float,
)

qnnpack_weighted_op_quint8_dtype_config = DTypeConfig(
    input_dtype=act_quint8_with_constraints,
    output_dtype=act_quint8_with_constraints,
    weight_dtype=weight_qint8_with_constraints,
    bias_dtype=torch.float,
)

# default op

qnnpack_default_op_qint8_dtype_config = DTypeConfig(
    input_dtype=act_qint8_with_constraints,
    output_dtype=act_qint8_with_constraints,
)

qnnpack_default_op_quint8_dtype_config = DTypeConfig(
    input_dtype=act_quint8_with_constraints,
    output_dtype=act_quint8_with_constraints,
)

qnnpack_default_op_fp16_dtype_config = DTypeConfig(
    input_dtype=act_fp16_with_constraints,
    output_dtype=act_fp16_with_constraints,
    weight_dtype=weight_fp16_with_constraints,
    bias_dtype=torch.float16,
)

# dynamic

qnnpack_default_dynamic_qint8_dtype_config = DTypeConfig(
    input_dtype=act_qint8_with_constraints,
    output_dtype=act_fp_with_constraints,
    weight_dtype=weight_qint8_with_constraints,
    bias_dtype=torch.float,
    is_dynamic=True,
)

qnnpack_default_dynamic_quint8_dtype_config = DTypeConfig(
    input_dtype=act_quint8_with_constraints,
    output_dtype=act_fp_with_constraints,
    weight_dtype=weight_qint8_with_constraints,
    bias_dtype=torch.float,
    is_dynamic=True,
)

qnnpack_default_dynamic_fp16_dtype_config = DTypeConfig(
    input_dtype=act_fp16_with_constraints,
    output_dtype=act_fp_with_constraints,
    weight_dtype=weight_fp16_with_constraints,
    bias_dtype=torch.float,
    is_dynamic=True,
)

# weight only

qnnpack_weight_only_qint8_dtype_config = DTypeConfig(
    input_dtype=act_fp_with_constraints,
    output_dtype=act_fp_with_constraints,
    weight_dtype=weight_qint8_with_constraints,
)

qnnpack_weight_only_quint8_dtype_config = DTypeConfig(
    input_dtype=act_fp_with_constraints,
    output_dtype=act_fp_with_constraints,
    weight_dtype=weight_quint8_with_constraints,
)

qnnpack_weight_only_quint4x2_dtype_config = DTypeConfig(
    input_dtype=act_fp_with_constraints,
    output_dtype=act_fp_with_constraints,
    weight_dtype=weight_quint4x2_with_constraints,
)


# =====================
# |  BACKEND CONFIGS  |
# =====================

def get_qnnpack_backend_config() -> BackendConfig:
    """
    Return the `BackendConfig` for PyTorch's native QNNPACK backend.
    """
    conv_dtype_configs = [
        qnnpack_weighted_op_qint8_dtype_config,
        qnnpack_weighted_op_quint8_dtype_config,
    ]
    linear_dtype_configs = [
        qnnpack_weighted_op_qint8_dtype_config,
        qnnpack_weighted_op_quint8_dtype_config,
        qnnpack_default_dynamic_qint8_dtype_config,
        qnnpack_default_dynamic_quint8_dtype_config,
        qnnpack_default_dynamic_fp16_dtype_config,
    ]
    binary_op_dtype_configs = [
        qnnpack_weighted_op_qint8_dtype_config,
        qnnpack_weighted_op_quint8_dtype_config,
    ]
    default_op_dtype_configs = [
        qnnpack_default_op_qint8_dtype_config,
        qnnpack_default_op_quint8_dtype_config,
    ]
    fixed_qparams_op_dtype_configs = [
        qnnpack_weighted_op_qint8_dtype_config,
        qnnpack_weighted_op_quint8_dtype_config,
    ]
    share_qparams_op_dtype_configs = [
        qnnpack_default_op_qint8_dtype_config,
        qnnpack_default_op_quint8_dtype_config,
    ]
    rnn_op_dtype_configs = [
        qnnpack_default_dynamic_qint8_dtype_config,
        qnnpack_default_dynamic_quint8_dtype_config,
        qnnpack_default_dynamic_fp16_dtype_config,
    ]
    embedding_op_dtype_configs = [
        qnnpack_weight_only_qint8_dtype_config,
        qnnpack_weight_only_quint8_dtype_config,
        qnnpack_weight_only_quint4x2_dtype_config,
    ]
    return BackendConfig("qnnpack") \
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
    "get_qnnpack_backend_config",
]
