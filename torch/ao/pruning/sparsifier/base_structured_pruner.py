from typing import Set, Type
import torch
from torch import nn
from torch.nn.utils import parametrize

from torch.ao.pruning import BaseSparsifier
from .utils import FakeStructuredSparsity, BiasHook

__all__ = ["BaseStructuredPruner"]

SUPPORTED_STRUCTURED_PRUNING_MODULES = {  # added to config if None given
    nn.Linear,
    nn.Conv2d,
}

class BaseStructuredPruner(BaseSparsifier):
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
        super().make_config_from_model(model, SUPPORTED_MODULES=SUPPORTED_STRUCTURED_PRUNING_MODULES)


    def _prepare(self, *args, **kwargs) -> None:
        r"""Adds mask parametrization to the layer weight
        """
        self.bias_handles = []

        for config in self.groups:
            module = config['module']
            tensor_name = config['tensor_name']
            parametrization = config.get('parametrization', FakeStructuredSparsity)

            mask = config.get('mask', torch.ones(getattr(module, tensor_name).shape[0], dtype=torch.bool))
            self.state[config['tensor_fqn']]['mask'] = mask
            parametrize.register_parametrization(module, tensor_name, parametrization(mask), unsafe=True)

            prune_bias = config.get('prune_bias', True)
            if prune_bias and module.bias is not None:
                module.register_parameter('_bias', nn.Parameter(module.bias.detach()))
                module.bias = None
            self.bias_handles.append(module.register_forward_hook(BiasHook(module.parametrizations.weight[0], prune_bias)))

    def convert(self):
        pass
