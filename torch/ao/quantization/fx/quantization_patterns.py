import torch
from torch.fx.graph import (
    Node,
)

from .utils import (
    all_node_args_have_no_tensors,
)
from torch.ao.quantization.utils import (
    Pattern,
    NodePattern,
)

from abc import ABC
from typing import Any, Callable, Dict, Optional

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

# TODO: move to backend_config_utils.py
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

    # TODO: can remove after the is_dynamic flag is defined, so that we can
    # move embedding op to backend_config_dict
    def input_output_observed(self) -> bool:
        """
        Returns True if the pattern matched to this qhandler could be
        be observed, and False it it should not be observed.
        """
        return True

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
        return qconfig.activation

    def is_custom_module(self):
        return self.is_custom_module_

    def is_standalone_module(self):
        return self.is_standalone_module_

# TODO: remove this class, this is still exposed in torch.quantization
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

# TODO: not used, can be removed after torch.quantization namespace is deprecated
class CustomModuleQuantizeHandler(QuantizeHandler):
    pass

# TODO: not used, can be removed after torch.quantization namespace is deprecated
class StandaloneModuleQuantizeHandler(QuantizeHandler):
    pass
