
import abc
from collections import defaultdict
import copy
import inspect

from torch.quantization.quantize import prepare as qprepare
from torch.quantization.quantize import convert as qconvert

from . import _variables

class BaseSparsifier(abc.ABC):
    r"""Base class for all sparsifiers.

    TODO: Docstring

    config keys:
        - 'module'
    """

    def __init__(self, config, defaults):
        self.config = config
        self.defaults = defaults

        self.state = defaultdict(dict)
        self.module_groups = {}
        self.enable_mask_update = False

        modules_to_sparsify = []

        for module_config in self.config:
            local_args = copy.deepcopy(defaults)
            local_args.update(module_config)
            local_key = local_args.pop('module')
            self.module_groups[local_key] = local_args

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
        for i, (layer, sparse_args) in enumerate(self.module_groups.items()):
                format_string += '\n'
                format_string += f'\tModule Group {i}\n'
                format_string += f'\t    module: {layer}\n'
                for key in sorted(sparse_args.keys()):
                    format_string += f'\t    {key}: {sparse_args[key]}\n'
        format_string += ')'
        return format_string

    def state_dict(self):
        raise NotImplementedError("This is WIP")

    def load_state_dict(self, state_dict):
        raise NotImplementedError("This is WIP")

    def prepare(self, model, mapping=None):
        r"""Prepares the model by inserting the fake sparsiers into the forward
        """
        if mapping is None:
            mapping = _variables.get_sparse_mapping()
        def new_child_fn(child, mapping):
            new_child = mapping[type(child)].from_dense(child)
            new_child.load_state_dict(child.state_dict())
            return new_child
        self._swap_modules(model, new_child_fn, mapping=mapping,
                           update_config=True)

        # In case there is qconfig, we would like to call the quantization
        # prepare, so that the observers are properly inserted.
        allow_list = mapping.values()
        qprepare(model, inplace=True, allow_list=allow_list)

        return model

    def convert(self, model, mapping=None):
        if mapping is None:
            mapping = _variables.get_static_sparse_quantized_mapping()
        def new_child_fn(child, mapping, **kwargs):
            new_child = mapping[type(child)].from_float(child, **kwargs)
            return new_child
        self._swap_modules(model, new_child_fn, mapping=mapping,
                           from_float_args=from_float_args)


    @abc.abstractmethod
    def step(self):
        return

    def _swap_modules(self, module, new_child_fn, mapping, update_config=False):
        # Recursively replace the layers in the module
        def swap(module, prefix=''):
            reassign = {}
            for name, child in module._modules.items():
                if child in self.module_groups and type(child) in mapping:
                    new_module = new_child_fn(child, mapping)
                    reassign[name] = new_module
                    if update_config:
                        self.module_groups[new_module] = self.module_groups.pop(child)
                elif child is not None:
                    swap(child, prefix + name + '.')
            for key, value in reassign.items():
                module._modules[key] = value
        swap(module)
