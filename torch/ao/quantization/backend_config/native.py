import torch
import torch.nn as nn
import torch.nn.intrinsic as nni
import torch.nn.qat as nnqat
import torch.nn.quantized._reference as nnqr
from ._common_operator_config_utils import (
    _get_binary_op_configs,
    _get_linear_configs,
    _get_conv_configs,
    _get_share_qparams_op_configs,
)
from .observation_type import ObservationType
from ..fake_quantize import FixedQParamsFakeQuantize
from ..fuser_method_mappings import (
    reverse_sequential_wrapper2,
)
from ..qconfig_mapping import _FIXED_QPARAMS_OP_TO_OBSERVER

# ===================
# |  DTYPE CONFIGS  |
# ===================

# weighted op int8 dtype config
# this is config for ops that has quantized weights, like linear, conv
weighted_op_int8_dtype_config = {
    # optional, input activation dtype
    "input_dtype": torch.quint8,
    # optional, weight dtype
    "weight_dtype": torch.qint8,
    # optional, bias dtype
    "bias_dtype": torch.float,
    # optional, output activation dtype
    "output_dtype": torch.quint8
}

default_op_quint8_dtype_config = {
    # optional, input activation dtype
    "input_dtype": torch.quint8,
    # optional, output activation dtype
    "output_dtype": torch.quint8,
}

default_op_fp16_dtype_config = {
    # optional, input activation dtype
    "input_dtype": torch.float16,
    # optional, weight dtype
    "weight_dtype": torch.float16,
    # optional, bias dtype
    "bias_dtype": torch.float16,
    # optional, output activation dtype
    "output_dtype": torch.float16,
}

default_dynamic_int8_dtype_config = {
    "input_dtype": torch.quint8,
    "weight_dtype": torch.qint8,
    "bias_dtype": torch.float,
    "output_dtype": torch.float,
    # currently the dtype check is not yet enabled, so we provided the dtype_configs but
    # it is not really used yet,
    # we will enable it a bit later after we moved everything to backend_config_dict
    "is_dynamic": True,
}

default_dynamic_float16_dtype_config = {
    "input_dtype": torch.float16,
    "weight_dtype": torch.float16,
    "bias_dtype": torch.float,
    "output_dtype": torch.float,
    # currently the dtype check is not yet enabled, so we provided the dtype_configs but
    # it is not really used yet,
    # we will enable it a bit later after we moved everything to backend_config_dict
    "is_dynamic": True,
}

weight_only_quint8_dtype_config = {
    "input_dtype": torch.float,
    "weight_dtype": torch.quint8,
    "output_dtype": torch.float,
}

weight_only_quint4x2_dtype_config = {
    "input_dtype": torch.float,
    "weight_dtype": torch.quint4x2,
    "output_dtype": torch.float,
}

# ======================
# |  OPERATOR CONFIGS  |
# ======================

def _get_default_op_backend_config(op, dtype_configs):
    return {
        "pattern": op,
        "observation_type": ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
        "dtype_configs": dtype_configs,
    }

_DEFAULT_OP_INT8_CONFIGS = [
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

def _get_fixed_qparams_op_configs(dtype_configs):
    fixed_qparams_op_configs = []
    for fixed_qparam_op, output_observer in _FIXED_QPARAMS_OP_TO_OBSERVER.items():
        fixed_qparams_op_configs.append({
            "pattern": fixed_qparam_op,
            "observation_type": ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
            # TODO: The following two keys are temporary, since we don't want to put observer in the configs
            # we expect that it's provided by user
            # What we want to put here is the requirement on observers, in this case dtype,
            # quant_min, quant_max etc., but we need to first move all configs to
            # backend_config_dict to do that, we'll remove these keys after we fully migrated
            # everything to use backend_config_dict
            "_overwrite_output_fake_quantizer": FixedQParamsFakeQuantize.with_args(observer=output_observer),
            "_overwrite_output_observer": output_observer,
            "dtype_configs": dtype_configs,
        })
    return fixed_qparams_op_configs

_CAT_CONFIG = {
    "pattern": torch.cat,
    "observation_type": ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT,
    "dtype_configs": [
        default_op_quint8_dtype_config,
    ]
}

def _get_bn_configs():
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
        bn_configs.append({
            "pattern": (torch.nn.ReLU, bn),
            "dtype_configs": [default_op_quint8_dtype_config],
            "fuser_method": reverse_sequential_wrapper2(fused_bn),
            "fused_module": fused_bn,
        })
        # bn module + F.relu fusion config
        bn_configs.append({
            "pattern": (torch.nn.functional.relu, bn),
            "dtype_configs": [default_op_quint8_dtype_config],
            "fuser_method": reverse_sequential_wrapper2(bn_to_fused_bn[bn]),
            "fused_module": fused_bn,
        })
        bn_configs.append({
            "pattern": bn,
            "observation_type": ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
            "dtype_configs": [default_op_quint8_dtype_config],
        })

    # fused bn configs
    for fused_bn in bn_to_fused_bn.values():
        bn_configs.append({
            "pattern": fused_bn,
            "observation_type": ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
            "dtype_configs": [default_op_quint8_dtype_config],
        })
    return bn_configs

def _get_rnn_op_configs():
    rnn_op_configs = []
    for rnn_op, ref_rnn_op in [
            (nn.GRUCell, nnqr.GRUCell),
            (nn.LSTMCell, nnqr.LSTMCell),
            (nn.RNNCell, nnqr.RNNCell),
            (nn.LSTM, nnqr.LSTM)
    ]:
        rnn_op_configs.append({
            "pattern": rnn_op,
            "observation_type": ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
            "dtype_configs": [default_dynamic_int8_dtype_config, default_dynamic_float16_dtype_config],
            "root_module": rnn_op,
            "reference_quantized_module_for_root": ref_rnn_op,
        })
    return rnn_op_configs

def _get_embedding_op_configs():
    embedding_op_configs = []
    for embedding_op, qat_embedding_op, ref_embedding_op in [
            (nn.Embedding, nnqat.Embedding, nnqr.Embedding),
            (nn.EmbeddingBag, nnqat.EmbeddingBag, nnqr.EmbeddingBag),
    ]:
        embedding_op_configs.append({
            "pattern": embedding_op,
            "observation_type": ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
            "dtype_configs": [
                weight_only_quint8_dtype_config,
                weight_only_quint4x2_dtype_config
            ],
            "qat_module": qat_embedding_op,
            "root_module": embedding_op,
            "reference_quantized_module_for_root": ref_embedding_op,
            # This is temporary, and will be removed soon
            "_input_output_observed": False
        })
        # config for qat op
        embedding_op_configs.append({
            "pattern": qat_embedding_op,
            "observation_type": ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
            "dtype_configs": [
                weight_only_quint8_dtype_config,
                weight_only_quint4x2_dtype_config
            ],
            "root_module": embedding_op,
            "reference_quantized_module_for_root": ref_embedding_op,
            # This is temporary, and will be removed soon
            "_input_output_observed": False
        })
    return embedding_op_configs

def get_test_only_legacy_native_backend_config_dict():
    """
    This is a backend configuration for the union of fbgemm/qnnpack
    and various additional fp16 ops.
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
    return {
        # optional
        "name": "_native_and_fp16",
        "configs": [
            *_DEFAULT_OP_INT8_CONFIGS,
            *_get_linear_configs(linear_dtype_configs),
            *_get_conv_configs(conv_dtype_configs),
            *_get_binary_op_configs(binary_op_dtype_configs),
            *_get_fixed_qparams_op_configs(fixed_qparams_op_dtype_configs),
            _CAT_CONFIG,
            *_get_bn_configs(),
            *_get_share_qparams_op_configs(share_qparams_op_dtype_configs),
            *_get_rnn_op_configs(),
            *_get_embedding_op_configs(),
        ],
    }

def get_native_backend_config_dict():
    """ Get backend_config_dict for PyTorch Native backend (fbgemm/qnnpack). """
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
    return {
        # optional
        "name": "native",
        "configs": [
            *_DEFAULT_OP_INT8_CONFIGS,
            *_get_linear_configs(linear_dtype_configs),
            *_get_conv_configs(conv_dtype_configs),
            *_get_binary_op_configs(binary_op_dtype_configs),
            *_get_fixed_qparams_op_configs(fixed_qparams_op_dtype_configs),
            _CAT_CONFIG,
            *_get_bn_configs(),
            *_get_share_qparams_op_configs(share_qparams_op_dtype_configs),
            *_get_rnn_op_configs(),
            *_get_embedding_op_configs(),
        ],
    }

__all__ = [
    "get_test_only_legacy_native_backend_config_dict",
    "get_native_backend_config_dict",
]
