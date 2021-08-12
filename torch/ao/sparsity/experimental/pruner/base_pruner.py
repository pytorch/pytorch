
import abc
import copy

import torch
from torch import nn
from torch.nn.utils import parametrize

from torch.nn.modules.container import ModuleDict, ModuleList

from .parametrization import PruningParametrization, ActivationReconstruction

SUPPORTED_MODULES = {
    nn.Linear,
    nn.Conv2d
}

def _module_to_fqn(model, layer, prefix=''):
    for name, child in model.named_children():
        new_name = prefix + '.' + name
        if child is layer:
            return new_name
        child_path = _module_to_fqn(child, layer, prefix=new_name)
        if child_path is not None:
            return child_path
    return None

def _fqn_to_module(model, path):
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
    def __init__(self, defaults):
        super().__init__()
        self.defaults = defaults
        if self.defaults is None:
            self.defaults = dict()

        self.module_groups = []
        self.enable_mask_update = True

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
            idx = [1] * len(output.shape)
            idx[1] = output.shape[1]
            bias = module._bias.reshape(idx)
            output += bias
        return output

    def _prepare(self, use_path=False, *args, **kwargs):
        r"""Adds mask parametrization to the layer weight
        """
        self.activation_handles = []  # store removable hook handles
        self.bias_handles = []

        for config in self.module_groups:
            if use_path:
                module = _fqn_to_module(self.model, config['fqn'])
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
            self.bias_handles.append(module.register_forward_hook(self.bias_hook))

    def prepare(self, model, config):
        r"""Prepares a model, by adding the parametrizations.
        Note::
            The model is modified inplace. If you need to preserve the original
            model, use copy.deepcopy.
        """
        self.model = model  # TODO: Need to figure out how to load without this.
        self.config = config

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
            module_fqn = _module_to_fqn(model, module)
            if module_fqn and module_fqn[0] == '.':
                module_fqn = module_fqn[1:]
            local_args['fqn'] = module_fqn
            self.module_groups.append(local_args)

        self._prepare()

    def squash_mask(self, use_path=False, *args, **kwargs):
        for config in self.module_groups:
            if use_path:
                module = _fqn_to_module(self.model, config['fqn'])
            else:
                module = config['module']
            parametrize.remove_parametrizations(module, 'weight',
                                                leave_parametrized=True)
            if getattr(module._parameters, 'mask', None):
                del module._parameters['mask']
            elif getattr(module._buffers, 'mask', None):
                del module._buffers['mask']
            delattr(module, 'mask')

    def convert(self):
        # TODO: Call the torch.ao.utils.convert in here
        raise NotImplementedError('`convert` is not implemented. Please, use '
                                  '`torch.ao.utils.convert` instead.')

    def step(self, use_path=True):
        if not self.enable_mask_update:
            return
        with torch.no_grad():
            for config in self.module_groups:
                if use_path:
                    module = _fqn_to_module(self.model, config['fqn'])
                else:
                    module = config['module']
                self.update_mask(module, **config)

    @abc.abstractmethod
    def update_mask(self, layer, **kwargs):
        pass
