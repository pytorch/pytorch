
import abc
import copy

import torch
from torch import nn
from torch.nn.utils import parametrize

from .parametrization import PruningParametrization, ActivationReconstruction

SUPPORTED_MODULES = {
    nn.Linear
}

def _module_to_path(model, layer, prefix=''):
    for name, child in model.named_children():
        new_name = prefix + '.' + name
        if child is layer:
            return new_name
        child_path = _module_to_path(child, layer, prefix=new_name)
        if child_path is not None:
            return child_path
    return None

def _path_to_module(model, path):
    path = path.split('.')
    for name in path:
        model = getattr(model, name, None)
        if model is None:
            return None
    return model


class BasePruner(abc.ABC):
    r"""Base class for all pruners.

    Abstract methods that need to be implemented:

    - update_mask: Function to compute a new mask for all keys in the
        `module_groups`.

    Args:
        - model [nn.Module]: model to configure. The model itself is not saved
            but used for the state_dict saving / loading.
        - config [list]: configuration elements could either be instances of
            nn.Module or dict maps. The dicts must have a key 'module' with the
            value being an instance of a nn.Module.
        - defaults [dict]: default configurations will be attached to the
            configuration. Only the keys that don't exist in the `config` will
            be updated.

    """
    def __init__(self, model, config, defaults):
        super().__init__()
        self.config = config
        self.defaults = defaults
        if self.defaults is None:
            self.defaults = dict()

        self.module_groups = []
        self.enable_mask_update = False
        self.activation_handles = []
        self.bias_handles = []

        self.model = model
        # If no config -- try getting all the supported layers
        if self.config is None:
            # Add all models to the config
            self.config = []
            stack = [model]
            while stack:
                module = stack.pop()
                for name, child in module.named_children():
                    if type(child) in SUPPORTED_MODULES:
                        self.config.append(child)
                    else:
                        stack.append(child)

        for module_config in self.config:
            if isinstance(module_config, nn.Module):
                module_config = {'module': module_config}
            local_args = copy.deepcopy(self.defaults)
            local_args.update(module_config)
            module = local_args['module']
            module_path = _module_to_path(self.model, module)
            if module_path and module_path[0] == '.':
                module_path = module_path[1:]
            local_args['path'] = module_path
            self.module_groups.append(local_args)

    def __getstate__(self):
        return {
            'defaults': self.defaults,
            'module_groups': self.module_groups,
        }

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __repr__(self):
        format_string = self.__class__.__name__ + ' ('
        for i, sparse_args in enumerate(self.module_groups):
            module = sparse_args['module']
            format_string += '\n'
            format_string += f'\tModule Group {i}\n'
            format_string += f'\t    module: {module}\n'
            for key in sorted(sparse_args.keys()):
                if key == 'module':
                    continue
                format_string += f'\t    {key}: {sparse_args[key]}\n'
        format_string += ')'
        return format_string

    def bias_hook(self, module, input, output):
        if getattr(module, '_bias', None) is not None:
            output += module._bias
        return output

    def prepare(self, use_path=False, *args, **kwargs):
        r"""Adds mask parametrization to the layer weight
        """
        for config in self.module_groups:
            if use_path:
                module = _path_to_module(self.model, config['path'])
            else:
                module = config['module']

            if getattr(module, 'mask', None) is None:
                module.register_buffer('mask', torch.tensor(module.weight.shape[0]))
            param = config.get('parametrization', PruningParametrization)
            parametrize.register_parametrization(module, 'weight',
                                                 param(module.mask),
                                                 unsafe=True)

            self.activation_handles.append(module.register_forward_hook(
                ActivationReconstruction(module.parametrizations.weight[0])
            ))

            if module.bias is not None:
                module.register_parameter('_bias', nn.Parameter(module.bias.detach()))
                module.bias = None
            self.bias_handles.append(module.register_forward_hook(self.bias_hook))

    def convert(self, use_path=False, *args, **kwargs):
        for config in self.module_groups:
            if use_path:
                module = _path_to_module(self.model, config['path'])
            else:
                module = config['module']
            parametrize.remove_parametrizations(module, 'weight',
                                                leave_parametrized=True)
            if getattr(module._parameters, 'mask', None):
                del module._parameters['mask']
            elif getattr(module._buffers, 'mask', None):
                del module._buffers['mask']
            delattr(module, 'mask')

    def step(self, use_path=True):
        if not self.enable_mask_update:
            return
        with torch.no_grad():
            for config in self.module_groups:
                if use_path:
                    module = _path_to_module(self.model, config['path'])
                else:
                    module = config['module']
                self.update_mask(module, **config)

    @abc.abstractmethod
    def update_mask(self, layer, **kwargs):
        pass
