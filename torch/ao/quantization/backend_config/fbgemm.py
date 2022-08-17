import torch
from .native import _get_native_backend_config_helper
from .backend_config import BackendConfig, DTypeConfig


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
    return _get_native_backend_config_helper(
        "fbgemm",
        conv_dtype_configs,
        linear_dtype_configs,
        binary_op_dtype_configs,
        default_op_dtype_configs,
        fixed_qparams_op_dtype_configs,
        share_qparams_op_dtype_configs,
        rnn_op_dtype_configs,
        embedding_op_dtype_configs)

__all__ = [
    "get_fbgemm_backend_config",
]
