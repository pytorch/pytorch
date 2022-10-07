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
    _get_ln_configs,
    _get_rnn_op_configs,
    _get_share_qparams_op_configs,
)
from .backend_config import BackendConfig, DTypeConfig


# ===================
# |  DTYPE CONFIGS  |
# ===================

# weighted op int8 dtype config
# this is config for ops that has quantized weights, like linear, conv
weighted_op_quint8_dtype_config = DTypeConfig(
    input_dtype=torch.quint8,
    output_dtype=torch.quint8,
    weight_dtype=torch.qint8,
    bias_dtype=torch.float,
)

default_op_quint8_dtype_config = DTypeConfig(
    input_dtype=torch.quint8,
    output_dtype=torch.quint8,
)

default_op_fp16_dtype_config = DTypeConfig(
    input_dtype=torch.float16,
    output_dtype=torch.float16,
    weight_dtype=torch.float16,
    bias_dtype=torch.float16,
)

default_dynamic_int8_dtype_config = DTypeConfig(
    input_dtype=torch.quint8,
    output_dtype=torch.float,
    weight_dtype=torch.qint8,
    bias_dtype=torch.float,
    # currently the dtype check is not yet enabled, so we provided the dtype_configs but
    # it is not really used yet,
    # we will enable it a bit later after we moved everything to backend_config_dict
    is_dynamic=True,
)

default_dynamic_float16_dtype_config = DTypeConfig(
    input_dtype=torch.float16,
    output_dtype=torch.float,
    weight_dtype=torch.float16,
    bias_dtype=torch.float,
    # currently the dtype check is not yet enabled, so we provided the dtype_configs but
    # it is not really used yet,
    # we will enable it a bit later after we moved everything to backend_config_dict
    is_dynamic=True,
)

# Needed for LayerNorm and f.layer_norm, since currently the kernel only supports float weights
input_output_only_quint8_dtype_config = DTypeConfig(
    input_dtype=torch.quint8,
    output_dtype=torch.quint8,
    weight_dtype=torch.float,
    bias_dtype=torch.float,
)

weight_only_quint8_dtype_config = DTypeConfig(
    input_dtype=torch.float,
    output_dtype=torch.float,
    weight_dtype=torch.quint8,
)

weight_only_quint4x2_dtype_config = DTypeConfig(
    input_dtype=torch.float,
    output_dtype=torch.float,
    weight_dtype=torch.quint4x2,
)


# =====================
# |  BACKEND CONFIGS  |
# =====================

def get_test_only_legacy_native_backend_config() -> BackendConfig:
    """
    Return the `BackendConfig` for PyTorch Native backend (fbgemm/qnnpack) with various additional fp16 ops.
    """
    conv_dtype_configs = [weighted_op_quint8_dtype_config]
    linear_dtype_configs = [
        weighted_op_quint8_dtype_config,
        default_dynamic_int8_dtype_config,
        default_dynamic_float16_dtype_config,
        default_op_fp16_dtype_config,
    ]
    binary_op_dtype_configs = [
        default_op_quint8_dtype_config,
        default_op_fp16_dtype_config,
    ]
    default_op_dtype_configs = [default_op_quint8_dtype_config]
    fixed_qparams_op_dtype_configs = [
        default_op_quint8_dtype_config,
        default_op_fp16_dtype_config,
    ]
    share_qparams_op_dtype_configs = [
        default_op_quint8_dtype_config,
        default_op_fp16_dtype_config
    ]
    rnn_op_dtype_configs = [
        default_dynamic_int8_dtype_config,
        default_dynamic_float16_dtype_config,
    ]
    embedding_op_dtype_configs = [
        weight_only_quint8_dtype_config,
        weight_only_quint4x2_dtype_config,
    ]
    layer_norm_op_dtype_configs = [input_output_only_quint8_dtype_config]
    return BackendConfig("_native_and_fp16") \
        .set_backend_pattern_configs(_get_conv_configs(conv_dtype_configs)) \
        .set_backend_pattern_configs(_get_linear_configs(linear_dtype_configs)) \
        .set_backend_pattern_configs(_get_binary_op_configs(binary_op_dtype_configs)) \
        .set_backend_pattern_config(_get_cat_config(default_op_dtype_configs)) \
        .set_backend_pattern_configs(_get_default_op_configs(default_op_dtype_configs)) \
        .set_backend_pattern_configs(_get_fixed_qparams_op_configs(fixed_qparams_op_dtype_configs)) \
        .set_backend_pattern_configs(_get_share_qparams_op_configs(share_qparams_op_dtype_configs)) \
        .set_backend_pattern_configs(_get_bn_configs(default_op_dtype_configs)) \
        .set_backend_pattern_configs(_get_ln_configs(layer_norm_op_dtype_configs)) \
        .set_backend_pattern_configs(_get_rnn_op_configs(rnn_op_dtype_configs)) \
        .set_backend_pattern_configs(_get_embedding_op_configs(embedding_op_dtype_configs))

def get_native_backend_config() -> BackendConfig:
    """
    Return the `BackendConfig` for PyTorch Native backend (fbgemm/qnnpack).
    """
    # TODO: express this BackendConfig as a union of the FBGEMM and QNNPACK BackendConfigs
    conv_dtype_configs = [weighted_op_quint8_dtype_config]
    linear_dtype_configs = [
        weighted_op_quint8_dtype_config,
        default_dynamic_int8_dtype_config,
        default_dynamic_float16_dtype_config,
    ]
    binary_op_dtype_configs = [default_op_quint8_dtype_config]
    default_op_dtype_configs = [default_op_quint8_dtype_config]
    fixed_qparams_op_dtype_configs = [default_op_quint8_dtype_config]
    share_qparams_op_dtype_configs = [default_op_quint8_dtype_config]
    rnn_op_dtype_configs = [
        default_dynamic_int8_dtype_config,
        default_dynamic_float16_dtype_config,
    ]
    embedding_op_dtype_configs = [
        weight_only_quint8_dtype_config,
        weight_only_quint4x2_dtype_config,
    ]
    layer_norm_op_dtype_configs = [input_output_only_quint8_dtype_config]
    return BackendConfig("native") \
        .set_backend_pattern_configs(_get_conv_configs(conv_dtype_configs)) \
        .set_backend_pattern_configs(_get_linear_configs(linear_dtype_configs)) \
        .set_backend_pattern_configs(_get_binary_op_configs(binary_op_dtype_configs)) \
        .set_backend_pattern_config(_get_cat_config(default_op_dtype_configs)) \
        .set_backend_pattern_configs(_get_default_op_configs(default_op_dtype_configs)) \
        .set_backend_pattern_configs(_get_fixed_qparams_op_configs(fixed_qparams_op_dtype_configs)) \
        .set_backend_pattern_configs(_get_share_qparams_op_configs(share_qparams_op_dtype_configs)) \
        .set_backend_pattern_configs(_get_bn_configs(default_op_dtype_configs)) \
        .set_backend_pattern_configs(_get_ln_configs(layer_norm_op_dtype_configs)) \
        .set_backend_pattern_configs(_get_rnn_op_configs(rnn_op_dtype_configs)) \
        .set_backend_pattern_configs(_get_embedding_op_configs(embedding_op_dtype_configs))

def get_native_backend_config_dict():
    """
    Return the `BackendConfig` for PyTorch Native backend (fbgemm/qnnpack) in dictionary form.
    """
    return get_native_backend_config().to_dict()

def get_test_only_legacy_native_backend_config_dict():
    """
    Return the `BackendConfig` for PyTorch Native backend (fbgemm/qnnpack) with various additional
    fp16 ops in dictionary form.
    """
    return get_test_only_legacy_native_backend_config().to_dict()

__all__ = [
    "get_test_only_legacy_native_backend_config",
    "default_op_quint8_dtype_config",
    "default_op_fp16_dtype_config",
    "default_dynamic_int8_dtype_config",
    "default_dynamic_float16_dtype_config",
    "input_output_only_quint8_dtype_config",
    "weight_only_quint8_dtype_config",
    "weight_only_quint4x2_dtype_config",
    "get_native_backend_config",
    "get_native_backend_config_dict",
    "get_test_only_legacy_native_backend_config_dict",
]
