import torch
import torch.nn.qat as nnqat
import operator
from .observation_type import ObservationType

def _get_default_op_backend_config(op, dtype_configs):
    return {
        "pattern": op,
        "observation_type": ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
        "dtype_configs": dtype_configs,
    }

# START dtype configs

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
    # optional, output activation dtype
    "output_dtype": torch.float16,
}
# END dtype configs

# operator (module/functional/torch ops) configs
_LINEAR_MODULE_CONFIG = {
    # Please see README under this folder for pattern format
    "pattern": torch.nn.Linear,
    "observation_type": ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
    "dtype_configs": [
        weighted_op_int8_dtype_config,
    ],
    # the root module for the pattern, used to query the reference quantized module
    # e.g. for a (torch.nn.ReLU, torch.nn.Linear) pattern, the root will be torch.nn.Linear
    "root_module": torch.nn.Linear,
    # the corresponding reference quantized module for the root module
    "reference_quantized_module_for_root": torch.nn.quantized._reference.Linear,
    "qat_module": nnqat.Linear,
}

_DEFAULT_OP_INT8_CONFIGS = [
    _get_default_op_backend_config(op, [default_op_quint8_dtype_config]) for op in [
        torch.nn.ConvTranspose1d,
        torch.nn.ConvTranspose2d,
        torch.nn.ELU,
        torch.nn.LeakyReLU,
        torch.nn.Hardswish,
        torch.nn.InstanceNorm1d,
        torch.nn.InstanceNorm2d,
        torch.nn.InstanceNorm3d,
        torch.nn.LayerNorm,
        torch.nn.Dropout,
        torch.nn.functional.elu,
        torch.nn.functional.hardswish,
        torch.nn.functional.instance_norm,
        torch.nn.functional.leaky_relu,
        torch.nn.functional.dropout,
        torch.nn.functional.layer_norm,
    ]]

_ADD_CONFIG = {
    "pattern": operator.add,
    "num_tensor_args_to_observation_type": {
        # TODO: maybe change this to NO_OBSERVER
        0: ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
        1: ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT,
        2: ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
    },
    "dtype_configs": [
        weighted_op_int8_dtype_config,
    ],
}


def get_native_backend_config_dict():
    """ Get backend for PyTorch Native backend_config_dict (fbgemm/qnnpack)
    """
    return {
        # optional
        "name": "native",
        "configs": [
            _LINEAR_MODULE_CONFIG,
            *_DEFAULT_OP_INT8_CONFIGS,
            _ADD_CONFIG,
        ],
    }
