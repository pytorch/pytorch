
import abc
import copy
from collections import defaultdict
from typing import Any, Dict, Tuple, Union

import torch
from torch import nn
from torch.nn.utils import parametrize

from .utils import FakeSparsity, module_to_fqn, fqn_to_module

SUPPORTED_MODULES = {
    nn.Linear
}

_SPARSE_PARAM_TYPE = Union[Tuple[str, ...], Dict[str, Tuple[str, ...]]]


class BaseSparsifier(abc.ABC):
    r"""Base class for all sparsifiers.

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

    Example::

        >>> config = [model.layer1, {'module': model.linear2, 'sparsity_level': 0.5}]
        >>> defaults = {'sparsity_level': 0.7}
        >>> # model.layer1 will have `sparsity_level` = 0.7 (getting default)
        >>> sparsifier = BaseSparsifier(config, defaults)
    """
    def __init__(self, defaults):
        super().__init__()
        self.defaults = defaults
        if self.defaults is None:
            self.defaults = dict()

        self.state: Dict[str, Dict] = defaultdict(dict)
        self.module_groups = []
        self.enable_mask_update = True

    def __getstate__(self):
        return {
            'defaults': self.defaults,
            'state': self.state,
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

    def state_dict(self):
        r"""Returns the state of the optimizer as a :class:`dict`.

        It contains:
        * state - current state of the sparsification.
        * module_groups - a list containing all sparsity configuration groups
            with the key 'fqn' specifying the layer path within a model

        TODO: Need a clean way of loading the state of the "preapred" module
        """
        module_groups = [
            dict(filter(lambda key_value: key_value[0] != 'module', mg.items()))
            for mg in self.module_groups
        ]

        return {
            'state': self.state,
            'module_groups': module_groups,
        }

    def load_state_dict(self, state_dict, strict=True):
        module_groups = copy.deepcopy(state_dict['module_groups'])
        states = state_dict['state']
        for fqn, s in states.items():
            layer = fqn_to_module(self.model, fqn)
            if strict and layer is None:
                raise RuntimeError(f'Error loading {fqn} into the model')

            found = False
            for p in layer.parametrizations['weight']:
                if isinstance(p, FakeSparsity):
                    found = True
                    break
            if not found:
                p = FakeSparsity(torch.ones(layer.weight.shape))
                parametrize.register_parametrization(layer, 'weight', p)
            if s.get('mask', None) is not None:
                mask = s.pop('mask')
                p.mask = mask

            for mg in module_groups:
                if mg['fqn'] == fqn:
                    mg['module'] = layer
        self.__setstate__({'state': states, 'module_groups': module_groups})

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

        # TODO: Remove the configuration by reference ('module')
        for module_config in self.config:
            if isinstance(module_config, nn.Module):
                module_config = {'module': module_config}
            local_args = copy.deepcopy(self.defaults)
            local_args.update(module_config)
            # Make sure there is at least one way of handling the model
            module = local_args.get('module', None)
            module_fqn = local_args.get('fqn', None)
            if module is None and module_fqn is None:
                # No module given for this group
                raise ValueError('Either `module` or `fqn` must be specified!')
            elif module is None:
                # FQN is given
                module = fqn_to_module(model, module_fqn)
            elif module_fqn is None:
                # Module is given
                module_fqn = module_to_fqn(model, module)
            else:
                # Both Module and FQN are given
                module_from_fqn = fqn_to_module(model, module_fqn)
                assert module is module_from_fqn, \
                    'Given both `module` and `fqn`, it is expected them to ' \
                    'refer to the same thing!'
            if module_fqn and module_fqn[0] == '.':
                module_fqn = module_fqn[1:]
            local_args['fqn'] = module_fqn
            local_args['module'] = module
            self.module_groups.append(local_args)

        self._prepare()

    def _prepare(self, *args, **kwargs):
        r"""Adds mask parametrization to the layer weight
        """
        for config in self.module_groups:
            module = config['module']
            param = config.get('parametrization', FakeSparsity)
            mask = config.get('mask', torch.ones(module.weight.shape))
            self.state[config['fqn']]['mask'] = mask
            parametrize.register_parametrization(module, 'weight', param(mask))

    def squash_mask(self, keep_sparse_params: _SPARSE_PARAM_TYPE = None,
                    *args, **kwargs):
        r"""Squashes the sparse masks into the appropriate tensors.

        If the `keep_sparse_params` is set, the module will have a
        `sparse_params` dict attached to it.

        Args:
            keep_sparse_params: List of keys to save in the module or a dict
                                representing the modules and keys that will have
                                sparsity parameters saved

        Examples:
            >>> # Don't save any sparse params
            >>> sparsifier.squash_mask()
            >>> hasattr(model.submodule1, 'sparse_params')
            False

            >>> # Keep sparse params per layer
            >>> sparsifier.squash_mask(
            ...     keep_sparse_params={
            ...         'submodule1.linear1': ('foo', 'bar'),
            ...         'submodule2.linear42': ('baz')
            ...     })
            >>> print(model.submodule1.linear1.sparse_params)
            {'foo': 42, 'bar': 24}
            >>> print(model.submodule2.linear42.sparse_params)
            {'baz': 0.1}

            >>> # Keep sparse params for all layers
            >>> sparsifier.squash_mask(keep_sparse_params=('foo', 'bar'))
            >>> print(model.submodule1.linear1.sparse_params)
            {'foo': 42, 'bar': 24}
            >>> print(model.submodule2.linear42.sparse_params)
            {'foo': 42, 'bar': 24}

        """
        for config in self.module_groups:
            module = config['module']
            parametrize.remove_parametrizations(module, 'weight',
                                                leave_parametrized=True)
            if keep_sparse_params is not None:
                if isinstance(keep_sparse_params, dict):
                    # Case 1: Dict[str, Tuple[str, ...]]
                    sparse_params = keep_sparse_params.get(config['fqn'], None)
                    if sparse_params is not None:
                        sparse_params = {k: config[k] for k in sparse_params}
                elif isinstance(keep_sparse_params, (tuple, list)):
                    # Case 2: Tuple[str, ...]
                    sparse_params = {k: config[k] for k in keep_sparse_params}
                else:
                    raise ValueError(
                        "'keep_sparse_params' should either be a list, a tuple, or a dict")
                module.sparse_params = sparse_params


    def convert(self):
        # TODO: Call the torch.ao.utils.convert in here
        raise NotImplementedError('`convert` is not implemented. Please, use '
                                  '`torch.ao.utils.convert` instead.')

    def step(self, use_path=True):
        if not self.enable_mask_update:
            return
        with torch.no_grad():
            for config in self.module_groups:
                module = config['module']
                self.update_mask(module, **config)

    @abc.abstractmethod
    def update_mask(self, layer, **kwargs):
        pass
