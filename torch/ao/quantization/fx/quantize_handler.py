import torch
from torch.fx.graph import (
    Node,
)

from .utils import (
    all_node_args_have_no_tensors,
)
from torch.ao.quantization.backend_config import (
    BackendConfig,
    DTypeConfig,
    ObservationType,
)
from torch.ao.quantization.utils import (
    NodePattern,
    Pattern,
    QuantizerCls,
)

from abc import ABC
from typing import Callable, Dict, List, Type

__all__ = [
    "QuantizeHandler",
    "BinaryOpQuantizeHandler",
    "CatQuantizeHandler",
    "ConvReluQuantizeHandler",
    "LinearReLUQuantizeHandler",
    "BatchNormQuantizeHandler",
    "EmbeddingQuantizeHandler",
    "RNNDynamicQuantizeHandler",
    "DefaultNodeQuantizeHandler",
    "FixedQParamsOpQuantizeHandler",
    "CopyNodeQuantizeHandler",
    "GeneralTensorShapeOpQuantizeHandler",
    "CustomModuleQuantizeHandler",
    "StandaloneModuleQuantizeHandler",
]

def _default_root_node_getter(node_pattern):
    if node_pattern is None:
        return node_pattern
    while not isinstance(node_pattern, Node):
        node_pattern = node_pattern[-1]
    return node_pattern

# Base Pattern Handler
class QuantizeHandler(ABC):
    """ Base handler class for the quantizer patterns
    """
    def __init__(
            self,
            node_pattern: NodePattern,
            modules: Dict[str, torch.nn.Module],
            root_node_getter: Callable = None,
            is_custom_module=False,
            is_standalone_module=False):
        """ Records pattern information in __init__, which will be used
        in convert
        """
        self.node_pattern = node_pattern
        self.modules = modules
        if root_node_getter is None:
            root_node_getter = _default_root_node_getter
        self.root_node = root_node_getter(node_pattern)
        self.is_custom_module_ = is_custom_module
        self.is_standalone_module_ = is_standalone_module
        self.num_tensor_args = 0
        # determine how many of the first two args are Tensors (versus scalars)
        # this distinguishes things like "x + y" from "x + 2" or "2 + x"
        if isinstance(self.root_node, Node):
            cache_for_no_tensor_check: Dict[Node, bool] = {}
            for arg_idx in range(len(self.root_node.args)):
                arg = self.root_node.args[arg_idx]
                if isinstance(arg, Node) and (
                        not all_node_args_have_no_tensors(
                            arg, self.modules, cache_for_no_tensor_check)):
                    self.num_tensor_args += 1

    def is_general_tensor_value_op(self) -> bool:
        """
        Returns True if the operator works for both floating point and
        quantized input, and does some computation based on the input Tensor,
        or the ops that only re-arranges the Tensor values or query some metadata
        about the Tensor
        so we need to insert observer/fake_quant for the output of the
        operator (same observer instance as input)
        since the distribution of values is different for input and output
        Tensors (for HistogramObserver) while they share the same quantization
        parameters
        Example operator: avgpool2d, reshape, transpose, maxpool2d
        Example observed operator:
        observer_0 - avgpool2d - observer_0 (same observer instance as input)
        """
        return False

    def is_custom_module(self):
        return self.is_custom_module_

    def is_standalone_module(self):
        return self.is_standalone_module_

def _get_quantize_handler_cls(
        observation_type: ObservationType,
        dtype_configs: List[DTypeConfig],
        num_tensor_args_to_observation_type: Dict[int, ObservationType]) -> Type[QuantizeHandler]:
    """
    Return a configurable QuantizeHandler that matches the given specifications from the backend.
    """

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

        def is_general_tensor_value_op(self) -> bool:
            return self.observation_type == ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT

    return ConfigurableQuantizeHandler

def _get_pattern_to_quantize_handlers(backend_config: BackendConfig) -> Dict[Pattern, QuantizerCls]:
    """
    Note: Quantize handler is just a holder for some check methods like
    (should_insert_observer_for_output), maybe this can be a enum as well,
    we can refactor this after we convert the path for fbgemm/qnnpack fully to the
    new path, this is not exposed to backend developers
    """
    pattern_to_quantize_handlers = {}
    for pattern, config in backend_config._pattern_complex_format_to_config.items():
        observation_type = config.observation_type
        dtype_configs = config.dtype_configs
        num_tensor_args_to_observation_type = config._num_tensor_args_to_observation_type
        pattern_to_quantize_handlers[pattern] = \
            _get_quantize_handler_cls(
                observation_type,
                dtype_configs,
                num_tensor_args_to_observation_type)
    return pattern_to_quantize_handlers

# TODO: remove this class, this is still exposed in torch.ao.quantization
# but we should be able to break bc
class BinaryOpQuantizeHandler(QuantizeHandler):
    pass

class CatQuantizeHandler(QuantizeHandler):
    pass

# TODO: remove this class
class ConvReluQuantizeHandler(QuantizeHandler):
    pass

# TODO: remove this class
class LinearReLUQuantizeHandler(QuantizeHandler):
    pass

# TODO: remove this class
class BatchNormQuantizeHandler(QuantizeHandler):
    pass

# TODO: remove this class
class EmbeddingQuantizeHandler(QuantizeHandler):
    pass

# TODO: remove this class
class RNNDynamicQuantizeHandler(QuantizeHandler):
    pass

# TODO: remove this class
class DefaultNodeQuantizeHandler(QuantizeHandler):
    """ Common quantized op, first input and first output will be quantized
    """
    pass

# TODO: remove this class
class FixedQParamsOpQuantizeHandler(QuantizeHandler):
    pass

# TODO: remove
class CopyNodeQuantizeHandler(QuantizeHandler):
    pass

# TODO: remove
class GeneralTensorShapeOpQuantizeHandler(QuantizeHandler):
    pass

# TODO: not used, can be removed after torch.ao.quantization namespace is deprecated
class CustomModuleQuantizeHandler(QuantizeHandler):
    pass

# TODO: not used, can be removed after torch.ao.quantization namespace is deprecated
class StandaloneModuleQuantizeHandler(QuantizeHandler):
    pass
