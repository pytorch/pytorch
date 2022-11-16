from itertools import chain
import torch
import torch.nn.functional as F
from torch import nn
from torch.fx import symbolic_trace
from torch.nn.utils import parametrize
from typing import Type, Set, Dict, Callable, Tuple, Any

from torch.ao.pruning import BaseSparsifier
from .parametrization import FakeStructuredSparsity, BiasHook
from .match_utils import apply_match
from .convert_functions import (
    convert_linear,
    convert_linear_linear,
    convert_linear_activation_linear,
    convert_conv2d,
    convert_conv2d_conv2d,
    convert_conv2d_activation_conv2d,
    convert_conv2d_activation_pool_conv2d,
    convert_conv2d_pool_activation_conv2d,
    convert_conv2d_pool_flatten_linear,
)

__all__ = ["BaseStructuredSparsifier"]

SUPPORTED_STRUCTURED_PRUNING_MODULES = {  # added to config if None given
    nn.Linear,
    nn.Conv2d,
}

SUPPORTED_ACTIVATION_FUNCTIONS = {
    F.relu,
    F.rrelu,
    F.hardtanh,
    F.relu6,
    F.sigmoid,
    F.hardsigmoid,
    F.tanh,
    F.silu,
    F.mish,
    F.hardswish,
    F.elu,
    F.celu,
    F.selu,
    F.hardshrink,
    F.leaky_relu,
    F.logsigmoid,
    F.softplus,
    F.prelu,
    F.softsign,
    F.tanhshrink,
}

SUPPORTED_ACTIVATION_MODULES = {
    nn.ReLU,
    nn.RReLU,
    nn.Hardtanh,
    nn.ReLU6,
    nn.Sigmoid,
    nn.Hardsigmoid,
    nn.Tanh,
    nn.SiLU,
    nn.Mish,
    nn.Hardswish,
    nn.ELU,
    nn.CELU,
    nn.SELU,
    nn.Hardshrink,
    nn.LeakyReLU,
    nn.LogSigmoid,
    nn.Softplus,
    nn.PReLU,
    nn.Softsign,
    nn.Tanhshrink,
}

def get_default_structured_pruning_patterns():
    """
    Returns the patterns for conv2d / linear conversion for each element in the activation functions/modules defined above.
    """
    patterns: Dict[Tuple[Any, ...], Callable] = {
        # linear -> linear
        (nn.Linear, "output"): convert_linear,
        (nn.Linear, nn.Linear): convert_linear_linear,
        # conv2d -> conv2d
        (nn.Conv2d, "output"): convert_conv2d,
        (nn.Conv2d, nn.Conv2d): convert_conv2d_conv2d,
    }

    for activation in chain(SUPPORTED_ACTIVATION_MODULES, SUPPORTED_ACTIVATION_FUNCTIONS):
        patterns.update({
            # linear -> activation -> linear
            (nn.Linear, activation, nn.Linear): convert_linear_activation_linear,
            # conv2d -> activation -> conv2d
            (nn.Conv2d, activation, nn.Conv2d): convert_conv2d_activation_conv2d,
            # conv2d -> activation -> pool -> conv2d
            (nn.Conv2d, activation, nn.AvgPool2d, nn.Conv2d): convert_conv2d_activation_pool_conv2d,
            (nn.Conv2d, activation, F.avg_pool2d, nn.Conv2d): convert_conv2d_activation_pool_conv2d,
            (nn.Conv2d, activation, nn.MaxPool2d, nn.Conv2d): convert_conv2d_activation_pool_conv2d,
            (nn.Conv2d, activation, F.max_pool2d, nn.Conv2d): convert_conv2d_activation_pool_conv2d,
            # conv2d -> pool -> activation -> conv2d
            (nn.Conv2d, nn.AvgPool2d, activation, nn.Conv2d): convert_conv2d_pool_activation_conv2d,
            (nn.Conv2d, F.avg_pool2d, activation, nn.Conv2d): convert_conv2d_pool_activation_conv2d,
            (nn.Conv2d, nn.MaxPool2d, activation, nn.Conv2d): convert_conv2d_pool_activation_conv2d,
            (nn.Conv2d, F.max_pool2d, activation, nn.Conv2d): convert_conv2d_pool_activation_conv2d,
            # conv2d -> adaptive pool -> flatten -> linear
            (nn.Conv2d, nn.AdaptiveAvgPool2d, nn.Flatten, nn.Linear): convert_conv2d_pool_flatten_linear,
            (nn.Conv2d, nn.AdaptiveAvgPool2d, torch.flatten, nn.Linear): convert_conv2d_pool_flatten_linear,
            (nn.Conv2d, nn.AdaptiveMaxPool2d, nn.Flatten, nn.Linear): convert_conv2d_pool_flatten_linear,
            (nn.Conv2d, nn.AdaptiveMaxPool2d, torch.flatten, nn.Linear): convert_conv2d_pool_flatten_linear,
        })
    return patterns

SUPPORTED_STRUCTURED_PRUNING_PATTERNS = get_default_structured_pruning_patterns()


class BaseStructuredSparsifier(BaseSparsifier):
    r"""Base class for structured pruning.

    Abstract methods that need to be implemented:
        - update_mask: Function to compute a new mask for all keys in the
            `groups` attribute.

    Args:
        - defaults [dict]: default configurations will be attached to the
            configuration. Only the keys that don't exist in the `config` will
            be updated.
    """

    def __init__(self, defaults, patterns=SUPPORTED_STRUCTURED_PRUNING_PATTERNS):
        super().__init__(defaults)
        self.patterns = patterns

    def make_config_from_model(
        self,
        model: nn.Module,
        SUPPORTED_MODULES: Set[Type] = SUPPORTED_STRUCTURED_PRUNING_MODULES,
    ) -> None:
        super().make_config_from_model(
            model, SUPPORTED_MODULES=SUPPORTED_STRUCTURED_PRUNING_MODULES
        )

    def _prepare(self, *args, **kwargs) -> None:
        r"""This function will attach the FakeStructuredSparsity parameterizations
        and BiasHooks at the appropriate points in the model.
        """
        self.bias_handles = []

        for config in self.groups:
            module = config["module"]
            tensor_name = config["tensor_name"]
            parametrization = config.get("parametrization", FakeStructuredSparsity)

            mask = config.get(
                "mask",
                torch.ones(getattr(module, tensor_name).shape[0], dtype=torch.bool),
            )
            self.state[config["tensor_fqn"]]["mask"] = mask
            parametrize.register_parametrization(
                module, tensor_name, parametrization(mask)
            )
            prune_bias = config.get("prune_bias", True)
            if module.bias is not None:
                module.register_parameter("_bias", nn.Parameter(module.bias.detach()))
                module.bias = None
                module.prune_bias = prune_bias
            self.bias_handles.append(
                module.register_forward_hook(
                    BiasHook(module.parametrizations.weight[0], prune_bias)
                )
            )

    def convert(self):
        r"""
        This function will FX symbolically trace the model and then find instances of the patterns
        defined in self.patterns (by default SUPPORTED_STRUCTURED_PRUNING_PATTERNS ).

        For each pattern, it will apply to corresponding conversion function, which will modify the output
        and input size expected by the modules within the pattern
        """

        self.traced = symbolic_trace(self.model)
        modules = dict(self.traced.named_modules())

        # Right now we check for matches simply by iterating across all the patterns
        # if this is slow we can store patterns in a trie-structure and modify this code for faster lookup
        for node in self.traced.graph.nodes:
            for pattern, convert_fn in self.patterns.items():
                matched = apply_match(modules, pattern, node, [])
                if matched is not None:
                    first_module = modules.get(node.target)

                    # check if first module exists and has a FakeStructuredSparsity parameterization, otherwise skip
                    if (
                        first_module is not None
                        and parametrize.is_parametrized(first_module)
                        and isinstance(
                            first_module.parametrizations["weight"][0],
                            FakeStructuredSparsity,
                        )
                    ):
                        convert_block = []
                        for node in matched:
                            if node.op == "call_module":
                                convert_block.append(modules.get(node.target))
                            elif node.op == "call_function":
                                convert_block.append(node.target)
                        convert_fn(*convert_block)

        # remove bias hooks, since biases are propogated during module conversion.
        for handle in self.bias_handles:
            handle.remove()

        self.traced.graph.lint()
        self.traced.recompile()
        return self.traced
