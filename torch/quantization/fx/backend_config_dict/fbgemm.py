import torch
from ..quantization_patterns import *
from ..pattern_utils import get_default_quant_patterns

def get_fbgemm_backend_config_dict():
    """ Get the backend config dictionary for fbgemm backend
    NOTE: Current api is not final, it's just to unblock experimentation for new backends
    """
    # TODO: add output_activation_post_process_map
    return {
        "quant_patterns": get_default_quant_patterns()
    }
