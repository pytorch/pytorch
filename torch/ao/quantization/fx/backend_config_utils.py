from torch.ao.quantization.quantization_types import Pattern, QuantizerCls
from .quantize_handler import get_quantize_handler_cls
from .fuse_handler import get_fuse_handler_cls
from torch.ao.quantization.fx.pattern_utils import get_default_quant_patterns, sorted_patterns_dict
from torch.ao.quantization.backend_config import get_native_backend_config_dict
from typing import Dict, Any, Callable
from torch.ao.quantization.utils import get_combined_dict

def get_pattern_to_quantize_handlers(
        backend_config_dict: Dict[str, Any]) -> Dict[Pattern, QuantizerCls]:
    """
    Note: Quantize handler is just a holder for some check methods like
    (should_insert_observer_for_output), maybe this can be a enum as well,
    we can refactor this after we convert the path for fbgemm/qnnpack fully to the
    new path, this is not exposed to backend developers
    """
    pattern_to_quantize_handlers = dict()
    for config in backend_config_dict.get("configs", []):
        pattern = config["pattern"]
        observation_type = config.get("observation_type", None)
        dtype_configs = config["dtype_configs"]
        num_tensor_args_to_observation_type = config.get("num_tensor_args_to_observation_type", {})
        overwrite_fake_quantizer = config.get("_overwrite_output_fake_quantizer", None)
        overwrite_observer = config.get("_overwrite_output_observer", None)
        input_output_observed = config.get("_input_output_observed", True)
        pattern_to_quantize_handlers[pattern] = \
            get_quantize_handler_cls(
                observation_type,
                dtype_configs,
                num_tensor_args_to_observation_type,
                overwrite_fake_quantizer,
                overwrite_observer,
                input_output_observed)

    return pattern_to_quantize_handlers

def get_fusion_pattern_to_fuse_handler_cls(
        backend_config_dict: Dict[str, Any]) -> Dict[Pattern, Callable]:
    fusion_pattern_to_fuse_handlers = dict()
    for config in backend_config_dict.get("configs", []):
        if "fuser_method" in config:
            pattern = config["pattern"]
            fusion_pattern_to_fuse_handlers[pattern] = \
                get_fuse_handler_cls()

    return fusion_pattern_to_fuse_handlers

# TODO: remove when all uses are changed to backend_config_dict
def get_native_quant_patterns(additional_quant_patterns: Dict[Pattern, QuantizerCls] = None) -> Dict[Pattern, QuantizerCls]:
    """
    Return a map from pattern to quantize handlers based on the default patterns and the native backend_config_dict.
    The returned map is sorted such that longer patterns will be encountered first when iterating through it.
    """
    patterns = get_default_quant_patterns()
    if additional_quant_patterns is not None:
        patterns = get_combined_dict(patterns, additional_quant_patterns)
    # TODO: currently we just extend the quantize handlers generated from
    # `get_native_backend_config_dict`
    # in the future we can just assign backend_config_dict when everything is defined
    for pattern, quantize_handler in get_pattern_to_quantize_handlers(get_native_backend_config_dict()).items():
        patterns[pattern] = quantize_handler
    return sorted_patterns_dict(patterns)
