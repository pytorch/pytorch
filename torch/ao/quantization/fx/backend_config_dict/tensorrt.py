import torch
from .observation_type import ObservationType

def get_tensorrt_backend_config_dict():
    """ Get the backend config dictionary for tensorrt backend
    NOTE: Current api will change in the future, it's just to unblock experimentation for
    new backends, please don't use it right now.
    """
    weighted_op_qint8_dtype_config = {
        # optional, input activation dtype
        "input_dtype": torch.qint8,
        # optional, weight dtype
        "weight_dtype": torch.qint8,
        # optional, bias dtype
        "bias_dtype": torch.float,
        # optional, output activation dtype
        "output_dtype": torch.qint8
    }
    linear_module_config = {
        # Please see README under this folder for pattern format
        "pattern": torch.nn.Linear,
        "observation_type": ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
        "dtype_configs": [
            weighted_op_qint8_dtype_config,
        ]
    }
    conv_module_config = {
        "pattern": torch.nn.Conv2d,
        "observation_type": ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
        "dtype_configs": [
            weighted_op_qint8_dtype_config,
        ]
    }
    return {
        # optional
        "name": "tensorrt",
        "configs": [
            linear_module_config,
            conv_module_config,
        ]
    }
