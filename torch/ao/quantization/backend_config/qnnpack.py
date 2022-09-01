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
from .backend_config import BackendConfig, DTypeConfig


# ===================
# |  DTYPE CONFIGS  |
# ===================

# TODO: For now, these DTypeConfigs are identical to the ones defined in native.py
# In the future, once we support specifying quant_min/quant_max and scale_min/scale_max,
# these will diverge. In particular, for QNNPACK, we will restrict the weight quantized
# values to within [-127, 127] and set the min scale value to 2 ** -12.

qnnpack_weighted_op_int8_dtype_config = DTypeConfig(
    input_dtype=torch.quint8,
    output_dtype=torch.quint8,
    weight_dtype=torch.qint8,
    bias_dtype=torch.float,
)

qnnpack_default_op_quint8_dtype_config = DTypeConfig(
    input_dtype=torch.quint8,
    output_dtype=torch.quint8,
)

qnnpack_default_op_fp16_dtype_config = DTypeConfig(
    input_dtype=torch.float16,
    output_dtype=torch.float16,
    weight_dtype=torch.float16,
    bias_dtype=torch.float16,
)

qnnpack_default_dynamic_int8_dtype_config = DTypeConfig(
    input_dtype=torch.quint8,
    output_dtype=torch.float,
    weight_dtype=torch.qint8,
    bias_dtype=torch.float,
    is_dynamic=True,
)

qnnpack_default_dynamic_float16_dtype_config = DTypeConfig(
    input_dtype=torch.float16,
    output_dtype=torch.float,
    weight_dtype=torch.float16,
    bias_dtype=torch.float,
    is_dynamic=True,
)

qnnpack_weight_only_quint8_dtype_config = DTypeConfig(
    input_dtype=torch.float,
    output_dtype=torch.float,
    weight_dtype=torch.quint8,
)

qnnpack_weight_only_quint4x2_dtype_config = DTypeConfig(
    input_dtype=torch.float,
    output_dtype=torch.float,
    weight_dtype=torch.quint4x2,
)


# =====================
# |  BACKEND CONFIGS  |
# =====================

def get_qnnpack_backend_config() -> BackendConfig:
    """
    Return the `BackendConfig` for PyTorch's native QNNPACK backend.
    """
    conv_dtype_configs = [qnnpack_weighted_op_int8_dtype_config]
    linear_dtype_configs = [
        qnnpack_weighted_op_int8_dtype_config,
        qnnpack_default_dynamic_int8_dtype_config,
        qnnpack_default_dynamic_float16_dtype_config,
    ]
    binary_op_dtype_configs = [qnnpack_weighted_op_int8_dtype_config]
    default_op_dtype_configs = [qnnpack_default_op_quint8_dtype_config]
    fixed_qparams_op_dtype_configs = [qnnpack_weighted_op_int8_dtype_config]
    share_qparams_op_dtype_configs = [qnnpack_default_op_quint8_dtype_config]
    rnn_op_dtype_configs = [
        qnnpack_default_dynamic_int8_dtype_config,
        qnnpack_default_dynamic_float16_dtype_config,
    ]
    embedding_op_dtype_configs = [
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
