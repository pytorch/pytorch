import torch
from .observation_type import ObservationType
import torch.nn.intrinsic as nni
from ...fuser_method_mappings import reverse2

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

    # operator (module/functional/torch ops) configs
    linear_module_config = {
        # Please see README under this folder for pattern format
        "pattern": torch.nn.Linear,
        "observation_type": ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
        "dtype_configs": [
            weighted_op_qint8_dtype_config,
        ],
        # the root module for the pattern, used to query the reference quantized module
        # e.g. for a (torch.nn.ReLU, torch.nn.Linear) pattern, the root will be torch.nn.Linear
        "root_module": torch.nn.Linear,
        # the corresponding reference quantized module for the root module
        "reference_quantized_module_for_root": torch.nn.quantized._reference.Linear,
    }
    # TODO: maybe make "pattern" to be a list of patterns
    # TODO: current patterns are the ones after fusion, we will want to expose fusion
    # here as well in the future, maybe we need to
    linear_relu_mm_config = {
        "pattern": (torch.nn.ReLU, torch.nn.Linear),
        "observation_type": ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
        "dtype_configs": [
            weighted_op_qint8_dtype_config,
        ],
        "fuser_method": reverse2(nni.LinearReLU),
        "root_module": torch.nn.Linear,
        "reference_quantized_module_for_root": torch.nn.quantized._reference.Linear,
    }
    linear_relu_mf_config = {
        "pattern": (torch.nn.functional.relu, torch.nn.Linear),
        "observation_type": ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
        "dtype_configs": [
            weighted_op_qint8_dtype_config,
        ],
        "fuser_method": reverse2(nni.LinearReLU),
        "root_module": torch.nn.Linear,
        "reference_quantized_module_for_root": torch.nn.quantized._reference.Linear,
    }

    linear_relu_fused_config = {
        "pattern": nni.LinearReLU,
        "observation_type": ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
        "dtype_configs": [
            weighted_op_qint8_dtype_config,
        ],
        "root_module": torch.nn.Linear,
        "reference_quantized_module_for_root": torch.nn.quantized._reference.Linear,
    }
    conv_module_config = {
        "pattern": torch.nn.Conv2d,
        "observation_type": ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
        "dtype_configs": [
            weighted_op_qint8_dtype_config,
        ],
        "root_module": torch.nn.Conv2d,
        "reference_quantized_module_for_root": torch.nn.quantized._reference.Conv2d,
    }
    conv_relu_1d_fused_config = {
        "pattern": torch.nn.intrinsic.ConvReLU1d,
        "observation_type": ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
        "dtype_configs": [
            weighted_op_qint8_dtype_config,
        ],
        "root_module": torch.nn.Conv1d,
        "reference_quantized_module_for_root": torch.nn.quantized._reference.Conv1d,
    }
    conv_relu_2d_fused_config = {
        "pattern": torch.nn.intrinsic.ConvReLU2d,
        "observation_type": ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
        "dtype_configs": [
            weighted_op_qint8_dtype_config,
        ],
        "root_module": torch.nn.Conv2d,
        "reference_quantized_module_for_root": torch.nn.quantized._reference.Conv2d,
    }
    conv_relu_3d_fused_config = {
        "pattern": torch.nn.intrinsic.ConvReLU3d,
        "observation_type": ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT,
        "dtype_configs": [
            weighted_op_qint8_dtype_config,
        ],
        "root_module": torch.nn.Conv3d,
        "reference_quantized_module_for_root": torch.nn.quantized._reference.Conv3d,
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
    identity_config = {
        "pattern": torch.nn.Identity,
        "observation_type": ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT,
        "dtype_configs": [
            non_weighted_op_qint8_dtype_config,
        ]
    }
    return {
        # optional
        "name": "tensorrt",
        "configs": [
            linear_module_config,
            linear_relu_fused_config,
            linear_relu_mm_config,
            linear_relu_mf_config,
            conv_module_config,
            # conv1d is not supported in fx2trt
            # conv_relu_1d_fused_config,
            conv_relu_2d_fused_config,
            # conv3d is not supported in fx2trt
            # conv_relu_3d_fused_config,
            addmm_config,
            cat_config,
            identity_config,
        ]
    }
