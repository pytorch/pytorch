import torch
from .observation_type import ObservationType
from ._common_operator_config_utils import (
    _get_binary_op_configs,
    _get_linear_configs,
    _get_conv_configs,
    _get_share_qparams_op_configs,
)

def get_tensorrt_backend_config_dict():
    """ Get the backend config dictionary for tensorrt backend
    NOTE: Current api will change in the future, it's just to unblock experimentation for
    new backends, please don't use it right now.
    TODO: add a README when it's more stable
    """
    # dtype configs
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
    non_weighted_op_qint8_dtype_config = {
        # optional, input activation dtype
        "input_dtype": torch.qint8,
        # optional, output activation dtype
        "output_dtype": torch.qint8,
    }

    addmm_config = {
        "pattern": torch.addmm,
        "observation_type": ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
        "dtype_configs": [
            weighted_op_qint8_dtype_config,
        ],
        # a map from input type to input index
        "input_type_to_index": {
            "bias": 0,
            "input": 1,
            "weight": 2,
        }
    }
    cat_config = {
        "pattern": torch.cat,
        "observation_type": ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT,
        "dtype_configs": [
            non_weighted_op_qint8_dtype_config,
        ]
    }
    conv_dtype_configs = [
        weighted_op_qint8_dtype_config,
    ]
    linear_dtype_configs = [
        weighted_op_qint8_dtype_config,
    ]
    binary_op_dtype_configs = [
        weighted_op_qint8_dtype_config,
    ]
    share_qparams_op_dtype_configs = [
        non_weighted_op_qint8_dtype_config,
    ]
    return {
        # optional
        "name": "tensorrt",
        "configs": [
            # there might be things not supported in fx2trt, but it will error out
            # during fx2trt conversion and can support them after that
            *_get_conv_configs(conv_dtype_configs),
            addmm_config,
            cat_config,
            *_get_linear_configs(linear_dtype_configs),
            *_get_binary_op_configs(binary_op_dtype_configs),
            *_get_share_qparams_op_configs(share_qparams_op_dtype_configs),
        ]
    }

__all__ = [
    "get_tensorrt_backend_config_dict",
]
