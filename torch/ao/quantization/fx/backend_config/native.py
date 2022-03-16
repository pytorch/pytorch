import torch
from .observation_type import ObservationType
import torch.nn.qat as nnqat

def get_native_backend_config_dict():
    """ Get backend for PyTorch Native backend_config_dict (fbgemm/qnnpack)
    """
    # dtype configs

    # weighted op int8 config
    # activation: quint8, weight: qint8, bias: float
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
    # operator (module/functional/torch ops) configs
    linear_module_config = {
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

    return {
        # optional
        "name": "native",
        "configs": [
            linear_module_config,
        ],
    }
