import torch
from ..quantization_patterns import ConvReLUQuantizeHandlerNew, LinearReLUQuantizeHandlerNew

def get_tensorrt_backend_config_dict():
    """ Get the backend config dictionary for tensorrt backend
    NOTE: Current api will change in the future, it's just to unblock experimentation for
    new backends, please don't use it right now.
    """
    quant_patterns = {
        torch.nn.Conv2d: ConvReLUQuantizeHandlerNew,
        torch.nn.Linear: LinearReLUQuantizeHandlerNew
    }
    return {
        "quant_patterns": quant_patterns
    }
