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


__all__ = [
    "get_qnnpack_backend_config",
]

# ===================
# |  DTYPE CONFIGS  |
# ===================

qnnpack_weighted_op_quint8_dtype_config = DTypeConfig(
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

# xnnpack compatible dtype configs

# We restrict scale values to be 2 ** -12 to ensure the
# requantization scale never falls below the xnnpack lower
# threshold. Additionally, for qint8 weight, we restrict
# the quantization values to [-127, +127], excluding -128.
# For more detail, refer to the description of
# `default_symmetric_qnnpack_qconfig`.

# TODO: add additional restriction on qscheme to ensure it
# is either per_tensor_symmetric or per_channel_symmetric

qnnpack_act_qint8_scale_min_2_neg_12 = DTypeWithConstraints(
    dtype=torch.qint8,
    scale_min_lower_bound=2**-12,
)

qnnpack_weight_qint8_neg_127_to_127_scale_min_2_neg_12 = DTypeWithConstraints(
    dtype=torch.qint8,
    quant_min_lower_bound=-127,
    quant_max_upper_bound=127,
    scale_min_lower_bound=2**-12,
)

qnnpack_weighted_op_qint8_symmetric_dtype_config = DTypeConfig(
    input_dtype=qnnpack_act_qint8_scale_min_2_neg_12,
    output_dtype=qnnpack_act_qint8_scale_min_2_neg_12,
    weight_dtype=qnnpack_weight_qint8_neg_127_to_127_scale_min_2_neg_12,
    bias_dtype=torch.float,
)

qnnpack_default_op_qint8_symmetric_dtype_config = DTypeConfig(
    input_dtype=qnnpack_act_qint8_scale_min_2_neg_12,
    output_dtype=qnnpack_act_qint8_scale_min_2_neg_12,
)


# =====================
# |  BACKEND CONFIGS  |
# =====================


def get_qnnpack_backend_config() -> BackendConfig:
    """
    Return the `BackendConfig` for PyTorch's native QNNPACK backend.
    """
    conv_dtype_configs = [
        qnnpack_weighted_op_qint8_symmetric_dtype_config,
        qnnpack_weighted_op_quint8_dtype_config,
    ]
    linear_dtype_configs = [
        qnnpack_weighted_op_qint8_symmetric_dtype_config,
        qnnpack_weighted_op_quint8_dtype_config,
        qnnpack_default_dynamic_int8_dtype_config,
        qnnpack_default_dynamic_float16_dtype_config,
    ]
    binary_op_dtype_configs = [
        qnnpack_default_op_qint8_symmetric_dtype_config,
        qnnpack_default_op_quint8_dtype_config,
    ]
    default_op_dtype_configs = [
        qnnpack_default_op_qint8_symmetric_dtype_config,
        qnnpack_default_op_quint8_dtype_config,
    ]
    fixed_qparams_op_dtype_configs = [
        qnnpack_default_op_qint8_symmetric_dtype_config,
        qnnpack_default_op_quint8_dtype_config,
    ]
    share_qparams_op_dtype_configs = [
        qnnpack_default_op_qint8_symmetric_dtype_config,
        qnnpack_default_op_quint8_dtype_config,
    ]
    rnn_op_dtype_configs = [
        qnnpack_default_dynamic_int8_dtype_config,
        qnnpack_default_dynamic_float16_dtype_config,
    ]
    embedding_op_dtype_configs = [
        qnnpack_weight_only_quint8_dtype_config,
        qnnpack_weight_only_quint4x2_dtype_config,
    ]
    return (
        BackendConfig("qnnpack")
        .set_backend_pattern_configs(_get_conv_configs(conv_dtype_configs))
        .set_backend_pattern_configs(_get_linear_configs(linear_dtype_configs))
        .set_backend_pattern_configs(_get_binary_op_configs(binary_op_dtype_configs))
        .set_backend_pattern_config(_get_cat_config(default_op_dtype_configs))
        .set_backend_pattern_configs(_get_default_op_configs(default_op_dtype_configs))
        .set_backend_pattern_configs(
            _get_fixed_qparams_op_configs(fixed_qparams_op_dtype_configs)
        )
        .set_backend_pattern_configs(
            _get_share_qparams_op_configs(share_qparams_op_dtype_configs)
        )
        .set_backend_pattern_configs(_get_bn_configs(default_op_dtype_configs))
        .set_backend_pattern_configs(_get_rnn_op_configs(rnn_op_dtype_configs))
        .set_backend_pattern_configs(
            _get_embedding_op_configs(embedding_op_dtype_configs)
        )
    )
