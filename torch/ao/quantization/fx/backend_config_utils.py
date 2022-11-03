import torch
from torch.ao.quantization.fx.pattern_utils import get_default_quant_patterns, sorted_patterns_dict
from torch.ao.quantization.backend_config import (
    get_native_backend_config,
    ObservationType,
)
from torch.ao.quantization.utils import (
    _activation_dtype,
    _get_combined_dict,
    Pattern,
    NodePattern,
    QuantizerCls,
)

from ..backend_config import BackendConfig
from .quantization_patterns import QuantizeHandler
from .fusion_patterns import DefaultFuseHandler

from typing import Dict, Any, Callable, Optional

def get_quantize_handler_cls(
        observation_type,
        dtype_configs,
        num_tensor_args_to_observation_type,
        overwrite_output_fake_quantizer,
        overwrite_output_observer,
        input_output_observed):

    class ConfigurableQuantizeHandler(QuantizeHandler):
        def __init__(
                self,
                node_pattern: NodePattern,
                modules: Dict[str, torch.nn.Module],
                root_node_getter: Callable = None):
            super().__init__(node_pattern, modules, root_node_getter)
            if num_tensor_args_to_observation_type:
                assert self.num_tensor_args in num_tensor_args_to_observation_type, \
                    f"Must provide observation_type config for tensor number {self.num_tensor_args}" \
                    f" in num_tensor_args_to_observation_type for {node_pattern}"
                self.observation_type = num_tensor_args_to_observation_type[self.num_tensor_args]
            else:
                self.observation_type = observation_type
            self.dtype_configs = dtype_configs
            self.overwrite_output_fake_quantizer = overwrite_output_fake_quantizer
            self.overwrite_output_observer = overwrite_output_observer
            self.input_output_observed_ = input_output_observed

        def is_general_tensor_value_op(self) -> bool:
            return self.observation_type == ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT

        # TODO: change this to output activation
        def get_activation_ctr(
                self,
                qconfig: Any,
                pattern: Pattern,
                is_training: bool,
        ) -> Optional[Callable]:
            """
            Returns the constructor for the activation observer which should be
            used for the pattern matched to this handler. Some handlers override
            this to a different value than what is specified in the qconfig.
            """
            act_dtype = _activation_dtype(qconfig)
            # TODO: change to is_qat
            if is_training:
                if act_dtype == torch.quint8 and self.overwrite_output_fake_quantizer is not None:
                    return self.overwrite_output_fake_quantizer
            else:
                if act_dtype == torch.quint8 and self.overwrite_output_observer is not None:
                    return self.overwrite_output_observer
            return qconfig.activation

        # This is temporary, and will be removed soon
        def input_output_observed(self):
            return self.input_output_observed_


    return ConfigurableQuantizeHandler

def get_pattern_to_quantize_handlers(backend_config: BackendConfig) -> Dict[Pattern, QuantizerCls]:
    """
    Note: Quantize handler is just a holder for some check methods like
    (should_insert_observer_for_output), maybe this can be a enum as well,
    we can refactor this after we convert the path for fbgemm/qnnpack fully to the
    new path, this is not exposed to backend developers
    """
    pattern_to_quantize_handlers = {}
    for pattern, config in backend_config.configs.items():
        observation_type = config.observation_type
        dtype_configs = config.dtype_configs
        num_tensor_args_to_observation_type = config._num_tensor_args_to_observation_type
        overwrite_fake_quantizer = config._overwrite_output_fake_quantize
        overwrite_observer = config._overwrite_output_observer
        input_output_observed = config._input_output_observed
        if input_output_observed is None:
            input_output_observed = True
        pattern_to_quantize_handlers[pattern] = \
            get_quantize_handler_cls(
                observation_type,
                dtype_configs,
                num_tensor_args_to_observation_type,
                overwrite_fake_quantizer,
                overwrite_observer,
                input_output_observed)

    return pattern_to_quantize_handlers

# TODO: move this to torch/ao/quantization/backend_config/utils.py
def get_fusion_pattern_to_fuse_handler_cls(
        backend_config: BackendConfig) -> Dict[Pattern, Callable]:
    fusion_pattern_to_fuse_handlers: Dict[Pattern, Callable] = {}
    for pattern, config in backend_config.configs.items():
        if config.fuser_method is not None:
            # TODO: is this logic right?
            fusion_pattern_to_fuse_handlers[pattern] = DefaultFuseHandler

    return fusion_pattern_to_fuse_handlers

# TODO: remove when all uses are changed to backend_config
def get_native_quant_patterns(additional_quant_patterns: Dict[Pattern, QuantizerCls] = None) -> Dict[Pattern, QuantizerCls]:
    """
    Return a map from pattern to quantize handlers based on the default patterns and the native backend_config.
    The returned map is sorted such that longer patterns will be encountered first when iterating through it.
    """
    patterns = get_default_quant_patterns()
    if additional_quant_patterns is not None:
        patterns = _get_combined_dict(patterns, additional_quant_patterns)
    # TODO: currently we just extend the quantize handlers generated from
    # `get_native_backend_config`
    # in the future we can just assign backend_config when everything is defined
    for pattern, quantize_handler in get_pattern_to_quantize_handlers(get_native_backend_config()).items():
        patterns[pattern] = quantize_handler
    return sorted_patterns_dict(patterns)

get_fusion_pattern_to_fuse_handler_cls.__module__ = "torch.ao.quantization.fx.backend_config_utils"
get_native_quant_patterns.__module__ = "torch.ao.quantization.fx.backend_config_utils"
get_pattern_to_quantize_handlers.__module__ = "torch.ao.quantization.fx.backend_config_utils"

__all__ = [
    "get_quantize_handler_cls",
    "get_fusion_pattern_to_fuse_handler_cls",
    "get_native_quant_patterns",
    "get_pattern_to_quantize_handlers",
]
