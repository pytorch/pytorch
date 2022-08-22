from typing import List
import torch
import torch.nn as nn
import torch.nn.intrinsic as nni
import torch.ao.nn.qat as nnqat
import torch.ao.nn.quantized._reference as nnqr
from ._common_operator_config_utils import (
    _get_binary_op_configs,
    _get_linear_configs,
    _get_conv_configs,
    _get_share_qparams_op_configs,
)
from .backend_config import (
    BackendConfig,
    BackendPatternConfig,
    DTypeConfig,
    ObservationType
)
from ..fake_quantize import FixedQParamsFakeQuantize
from ..fuser_method_mappings import (
    reverse_sequential_wrapper2,
)
from ..qconfig_mapping import _FIXED_QPARAMS_OP_TO_OBSERVER
from ..utils import Pattern

# ===================
# |  DTYPE CONFIGS  |
# ===================

# weighted op int8 dtype config
# this is config for ops that has quantized weights, like linear, conv
weighted_op_int8_dtype_config = DTypeConfig(
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

# ======================
# |  OPERATOR CONFIGS  |
# ======================

def _get_default_op_backend_config(op: Pattern, dtype_configs: List[DTypeConfig]) -> BackendPatternConfig:
    return BackendPatternConfig(op) \
        .set_observation_type(ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT) \
        .set_dtype_configs(dtype_configs)

_DEFAULT_OP_INT8_CONFIGS: List[BackendPatternConfig] = [
    _get_default_op_backend_config(op, [default_op_quint8_dtype_config]) for op in [
        torch.nn.ELU,
        torch.nn.LeakyReLU,
        torch.nn.Hardswish,
        torch.nn.InstanceNorm1d,
        torch.nn.InstanceNorm2d,
        torch.nn.InstanceNorm3d,
        torch.nn.LayerNorm,
        torch.nn.Dropout,
        torch.nn.PReLU,
        torch.nn.functional.elu,
        torch.nn.functional.hardswish,
        torch.nn.functional.instance_norm,
        torch.nn.functional.leaky_relu,
        torch.nn.functional.dropout,
        torch.nn.functional.layer_norm,
    ]]

def _get_fixed_qparams_op_configs(dtype_configs: List[DTypeConfig]) -> List[BackendPatternConfig]:
    fixed_qparams_op_configs = []
    for fixed_qparam_op, output_observer in _FIXED_QPARAMS_OP_TO_OBSERVER.items():
        fixed_qparams_op_configs.append(
            # TODO: The _overwrite_output keys are temporary, since we don't want to put observer
            # in the configs we expect that it's provided by user
            # What we want to put here is the requirement on observers, in this case dtype,
            # quant_min, quant_max etc., but we need to first move all configs to
            # backend_config_dict to do that, we'll remove these keys after we fully migrated
            # everything to use backend_config_dict
            BackendPatternConfig(fixed_qparam_op)
                .set_observation_type(ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT)  # noqa: E131
                .set_dtype_configs(dtype_configs)
                ._set_overwrite_output_fake_quantize(FixedQParamsFakeQuantize.with_args(observer=output_observer))
                ._set_overwrite_output_observer(output_observer))
    return fixed_qparams_op_configs

_CAT_CONFIG = BackendPatternConfig(torch.cat) \
    .set_observation_type(ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT) \
    .add_dtype_config(default_op_quint8_dtype_config)

def _get_bn_configs() -> List[BackendPatternConfig]:
    """ Get configs related to batchnorm
    """
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
                .add_dtype_config(default_op_quint8_dtype_config)  # noqa: E131
                .set_fuser_method(reverse_sequential_wrapper2(fused_bn))
                .set_fused_module(fused_bn))
        # bn module + F.relu fusion config
        bn_configs.append(
            BackendPatternConfig((torch.nn.functional.relu, bn))
                .add_dtype_config(default_op_quint8_dtype_config)  # noqa: E131
                .set_fuser_method(reverse_sequential_wrapper2(bn_to_fused_bn[bn]))
                .set_fused_module(fused_bn))
        bn_configs.append(
            BackendPatternConfig(bn)
                .set_observation_type(ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT)  # noqa: E131
                .add_dtype_config(default_op_quint8_dtype_config))

    # fused bn configs
    for fused_bn in bn_to_fused_bn.values():
        bn_configs.append(
            BackendPatternConfig(fused_bn)
                .set_observation_type(ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT)  # noqa: E131
                .add_dtype_config(default_op_quint8_dtype_config))
    return bn_configs

def _get_rnn_op_configs() -> List[BackendPatternConfig]:
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
                .add_dtype_config(default_dynamic_int8_dtype_config)
                .add_dtype_config(default_dynamic_float16_dtype_config)
                .set_root_module(rnn_op)
                .set_reference_quantized_module(ref_rnn_op))
    return rnn_op_configs

def _get_embedding_op_configs() -> List[BackendPatternConfig]:
    embedding_op_configs = []
    for embedding_op, qat_embedding_op, ref_embedding_op in [
            (nn.Embedding, nnqat.Embedding, nnqr.Embedding),
            (nn.EmbeddingBag, nnqat.EmbeddingBag, nnqr.EmbeddingBag),
    ]:
        embedding_op_configs.append(
            BackendPatternConfig(embedding_op)
                .set_observation_type(ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT)  # noqa: E131
                .add_dtype_config(weight_only_quint8_dtype_config)
                .add_dtype_config(weight_only_quint4x2_dtype_config)
                .set_qat_module(qat_embedding_op)
                .set_root_module(embedding_op)
                .set_reference_quantized_module(ref_embedding_op)
                ._set_input_output_observed(False))  # This is temporary, and will be removed soon
        # config for qat op
        embedding_op_configs.append(
            BackendPatternConfig(qat_embedding_op)
                .set_observation_type(ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT)  # noqa: E131
                .add_dtype_config(weight_only_quint8_dtype_config)
                .add_dtype_config(weight_only_quint4x2_dtype_config)
                .set_root_module(embedding_op)
                .set_reference_quantized_module(ref_embedding_op)
                ._set_input_output_observed(False))  # This is temporary, and will be removed soon
    return embedding_op_configs

def get_test_only_legacy_native_backend_config() -> BackendConfig:
    """
    Return the `BackendConfig` for PyTorch Native backend (fbgemm/qnnpack) with various additional fp16 ops.
    """
    conv_dtype_configs = [weighted_op_int8_dtype_config]
    linear_dtype_configs = [
        weighted_op_int8_dtype_config,
        default_dynamic_int8_dtype_config,
        default_dynamic_float16_dtype_config,
        default_op_fp16_dtype_config,
    ]
    binary_op_dtype_configs = [
        weighted_op_int8_dtype_config,
        default_op_fp16_dtype_config,
    ]
    share_qparams_op_dtype_configs = [
        default_op_quint8_dtype_config,
        default_op_fp16_dtype_config
    ]
    fixed_qparams_op_dtype_configs = [
        weighted_op_int8_dtype_config,
        default_op_fp16_dtype_config,
    ]
    return BackendConfig("_native_and_fp16") \
        .set_backend_pattern_configs(_DEFAULT_OP_INT8_CONFIGS) \
        .set_backend_pattern_configs(_get_linear_configs(linear_dtype_configs)) \
        .set_backend_pattern_configs(_get_conv_configs(conv_dtype_configs)) \
        .set_backend_pattern_configs(_get_binary_op_configs(binary_op_dtype_configs)) \
        .set_backend_pattern_configs(_get_fixed_qparams_op_configs(fixed_qparams_op_dtype_configs)) \
        .set_backend_pattern_config(_CAT_CONFIG) \
        .set_backend_pattern_configs(_get_bn_configs()) \
        .set_backend_pattern_configs(_get_share_qparams_op_configs(share_qparams_op_dtype_configs)) \
        .set_backend_pattern_configs(_get_rnn_op_configs()) \
        .set_backend_pattern_configs(_get_embedding_op_configs())

def get_native_backend_config() -> BackendConfig:
    """
    Return the `BackendConfig` for PyTorch Native backend (fbgemm/qnnpack).
    """
    conv_dtype_configs = [weighted_op_int8_dtype_config]
    linear_dtype_configs = [
        weighted_op_int8_dtype_config,
        default_dynamic_int8_dtype_config,
        default_dynamic_float16_dtype_config,
    ]
    binary_op_dtype_configs = [
        weighted_op_int8_dtype_config,
    ]
    share_qparams_op_dtype_configs = [
        default_op_quint8_dtype_config,
    ]
    fixed_qparams_op_dtype_configs = [
        weighted_op_int8_dtype_config,
    ]
    return BackendConfig("native") \
        .set_backend_pattern_configs(_DEFAULT_OP_INT8_CONFIGS) \
        .set_backend_pattern_configs(_get_linear_configs(linear_dtype_configs)) \
        .set_backend_pattern_configs(_get_conv_configs(conv_dtype_configs)) \
        .set_backend_pattern_configs(_get_binary_op_configs(binary_op_dtype_configs)) \
        .set_backend_pattern_configs(_get_fixed_qparams_op_configs(fixed_qparams_op_dtype_configs)) \
        .set_backend_pattern_config(_CAT_CONFIG) \
        .set_backend_pattern_configs(_get_bn_configs()) \
        .set_backend_pattern_configs(_get_share_qparams_op_configs(share_qparams_op_dtype_configs)) \
        .set_backend_pattern_configs(_get_rnn_op_configs()) \
        .set_backend_pattern_configs(_get_embedding_op_configs())

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
    "get_test_only_legacy_native_backend_config_dict",
    "get_native_backend_config",
    "get_native_backend_config_dict",
]
