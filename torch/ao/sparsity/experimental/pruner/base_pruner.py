
import abc

import torch
from torch import nn
from torch.nn.utils import parametrize

from torch.nn.modules.container import ModuleDict, ModuleList

from .parametrization import PruningParametrization, ActivationReconstruction, BiasHook

from torch.ao.sparsity import BaseSparsifier, fqn_to_module

SUPPORTED_MODULES = {
    nn.Linear,
    nn.Conv2d
}


class BasePruner(BaseSparsifier):
    r"""Base class for all pruners.

    Abstract methods that need to be implemented:

    - update_mask: Function to compute a new mask for all keys in the
        `module_groups`.

    Args:
        - defaults [dict]: default configurations will be attached to the
            configuration. Only the keys that don't exist in the `config` will
            be updated.
        - also_prune_bias [bool]: whether to prune bias in addition to weights (to prune full output channel)
            or not; default=True.

    """
    def __init__(self, defaults, also_prune_bias=True):
        super().__init__(defaults)
        self.prune_bias = also_prune_bias

    def _prepare(self, use_path=False, *args, **kwargs):
        r"""Adds mask parametrization to the layer weight
        """
        self.activation_handles = []  # store removable hook handles
        self.bias_handles = []

        for config in self.module_groups:
            if use_path:
                module = fqn_to_module(self.model, config['fqn'])
            else:
                module = config['module']

            if getattr(module, 'mask', None) is None:
                module.register_buffer('mask', torch.tensor(module.weight.shape[0]))
            param = config.get('parametrization', PruningParametrization)
            parametrize.register_parametrization(module, 'weight',
                                                 param(module.mask),
                                                 unsafe=True)

            assert isinstance(module.parametrizations, ModuleDict)  # make mypy happy
            assert isinstance(module.parametrizations.weight, ModuleList)
            if isinstance(module, tuple(SUPPORTED_MODULES)):
                self.activation_handles.append(module.register_forward_hook(
                    ActivationReconstruction(module.parametrizations.weight[0])
                ))
            else:
                raise NotImplementedError("This module type is not supported yet.")

            if module.bias is not None:
                module.register_parameter('_bias', nn.Parameter(module.bias.detach()))
                module.bias = None
            self.bias_handles.append(module.register_forward_hook(BiasHook(module.parametrizations.weight[0], self.prune_bias)))

    def squash_mask(self, use_path=False, *args, **kwargs):
        for config in self.module_groups:
            if use_path:
                module = fqn_to_module(self.model, config['fqn'])
            else:
                module = config['module']
            parametrize.remove_parametrizations(module, 'weight',
                                                leave_parametrized=True)
            if getattr(module._parameters, 'mask', None):
                del module._parameters['mask']
            elif getattr(module._buffers, 'mask', None):
                del module._buffers['mask']
            delattr(module, 'mask')

    @abc.abstractmethod
    def update_mask(self, layer, **kwargs):
        pass
