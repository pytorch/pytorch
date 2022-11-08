import torch
import torch.nn.functional as F
from torch import nn
from torch.fx import Node, symbolic_trace
from torch.nn.utils import parametrize
from torch.ao.quantization.fx.match_utils import (
    MatchAllNode,
)
from typing import Any, Dict, List, Callable, Optional, Tuple, Type, Union, Set

from torch.ao.pruning import BaseSparsifier
from .utils import FakeStructuredSparsity, BiasHook
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

    def __init__(self, defaults):
        super().__init__(defaults)

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

        patterns = {
            (nn.Conv2d, F.max_pool2d, nn.ReLU, nn.Conv2d): convert_conv2d_pool_activation_conv2d,
            (nn.Conv2d, nn.ReLU, nn.Conv2d): convert_conv2d_activation_conv2d,
            (nn.Conv2d, "output"): convert_conv2d,
            (nn.Conv2d, nn.Tanh, nn.AvgPool2d, nn.Conv2d): convert_conv2d_activation_pool_conv2d,
            (nn.Conv2d, nn.MaxPool2d, nn.ReLU, nn.Conv2d): convert_conv2d_pool_activation_conv2d,
            (nn.Conv2d, nn.Conv2d): convert_conv2d_conv2d,
            (nn.Conv2d, nn.Tanh, nn.Conv2d): convert_conv2d_activation_conv2d,
            (nn.Conv2d, F.relu, nn.Conv2d): convert_conv2d_activation_conv2d,
            (nn.Conv2d, nn.AdaptiveAvgPool2d, torch.flatten, nn.Linear): convert_conv2d_pool_flatten_linear,
            (nn.Conv2d, nn.AdaptiveAvgPool2d, nn.Flatten, nn.Linear): convert_conv2d_pool_flatten_linear,
            (nn.Linear, "output"): convert_linear,
            (nn.Linear, nn.Linear): convert_linear_linear,
            (nn.Linear, nn.ReLU, nn.Linear): convert_linear_activation_linear,
            (nn.Linear, nn.Tanh, nn.Linear): convert_linear_activation_linear,
            (nn.Linear, F.relu, nn.Linear): convert_linear_activation_linear,
        }

        self.traced = symbolic_trace(self.model)
        modules = dict(self.traced.named_modules())

        def match(node: Node, current: nn.Module) -> bool:
            if isinstance(current, type) and issubclass(current, MatchAllNode):
                return True
            if not isinstance(node, Node):
                return False
            if isinstance(current, type) and issubclass(current, torch.nn.Module):
                return (
                    node.op == "call_module"
                    and parametrize.type_before_parametrizations(modules[node.target])
                    == current
                )
            elif callable(current):
                return node.op == "call_function" and node.target is current
            elif isinstance(current, str):
                return node.target == current
            else:
                return False

        def apply_match(
            pattern: Union[Tuple[nn.Module], nn.Module],
            node: Node,
            matched_node_pattern: List[Node],
        ) -> Optional[List[Node]]:
            if isinstance(pattern, tuple):
                if len(pattern) == 1:
                    if match(node, pattern[0]):
                        if isinstance(pattern[0], str):
                            return matched_node_pattern
                        return matched_node_pattern + [node]

                s, *args = pattern
                if match(node, s):
                    if args is None:
                        return matched_node_pattern + [node]

                    for user in node.users:
                        return apply_match(
                            tuple(args), user, matched_node_pattern + [node]
                        )
            elif match(node, pattern):
                return [node]

        for node in self.traced.graph.nodes:
            for pattern, convert_fn in patterns.items():
                matched = apply_match(pattern, node, [])
                if matched is not None and parametrize.is_parametrized(
                    modules.get(node.target)
                ):
                    convert_fn(
                        *[modules.get(node.target, node.target) for node in matched]
                    )

        for handle in self.bias_handles:
            handle.remove()

        self.traced.graph.lint()
        self.traced.recompile()
        return self.traced
