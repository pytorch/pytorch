from ..pattern_utils import get_default_quant_patterns

def get_fbgemm_backend_config_dict():
    """ Get the backend config dictionary for fbgemm backend
    NOTE: Current api will change in the future, it's just to unblock experimentation for
    new backends, please don't use it right now.
    """
    # TODO: add output_activation_post_process_map
    return {
        "quant_patterns": get_default_quant_patterns()
    }
