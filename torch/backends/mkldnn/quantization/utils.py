from torch.quantization import qconfig
import torch
from torch import nn
from enum import Enum

class QuantizedOperatorType(Enum):
    NEED_OBSERVER_FOR_BOTH_INPUTS_AND_OUTPUTS = 0
    OUTPUT_IS_SHARING_OBSERVER_WITH_INPUT = 1

# Customized config for MKLDNN backend.
# Note: This is not used now since the PyTorch API is not ready
mkldnn_backend_config_dict = {
    # optional
    "name": "MKLDNN",
    # quantized operator config is a map from
    # module/functional/torch ops to their configurations
    "operator": {
    }
}

# define a function to return the backend config dict
def get_mkldnn_backend_config_dict():
    return mkldnn_backend_config_dict
