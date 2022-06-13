import abc
import torch
from typing import Optional, Tuple, List, Any, Dict
from ...sparsifier import base_sparsifier
from collections import defaultdict
from torch import nn
import warnings
import copy
from ...sparsifier import utils
from torch.nn.utils import parametrize

__all__ = ['BaseDataSparsifier']

SUPPORTED_TYPES = {
    torch.Tensor
}


class _Container(nn.Module):
    def __init__(self):
        super().__init__()


class BaseDataSparsifier(base_sparsifier.BaseSparsifier):
    r"""
    Base Data Sparsifier class for all Data sparsifiers.
    The abstract class accepts raw torch tensors / embedding / embedding bags (refer to SUPPORTED_TYPES above)
    to prepare for sparsification.
    In this case, mask (and parametrizations) is owned by the class and not by the user.
    Specifically, the container object inside the class maintains the mask and parametrizations of the input data

    Args:
        data_list (list of tuples)
            list of (name, data) tuples to sparsify. Lookup SUPPORTED_TYPES
            for type of data. Internally, a container module handles the data sparsification.

        defaults (dict)
            default configurations will be attached to the
            configuration. Only the keys that don't exist in the `config` will
            be updated.
    Example::

        >>> data_list = [('tensor_1', torch.randn(3,3)), ('tensor_2', torch.randn(4,4))]
        >>> defaults = {'sparsity_level': 0.7}
        >>> sparsifier = DerivedDataSparsifier(data_list = data_list, **defaults) # Some sparsifier that inherits BaseDataSparsifier
        >>> new_tensor_to_add = {'name': 'tensor_3', 'data': torch.randn(5,5), 'sparsity_level': 0.3}
        >>> sparsifier.add_data(**new_tensor_to_add)
        >>> # tensor_1 and tensor_2 will have sparsity_level of 0.7 but tensor_3 will have sparsity_level=0.3
    """
    def __init__(self, data_list: Optional[List[Tuple[str, Any]]] = None, **defaults):
        super().__init__(defaults=defaults)

        self._container = _Container()

        self.data_groups: Dict[str, Dict] = defaultdict(dict)  # name -> {**config}
        if data_list is not None:
            # add data with default config here
            [self.add_data(name, data, **self.defaults) for name, data in data_list]

    def prepare(self):
        raise NotImplementedError("this function is undefined for this class")

    def _extract_weight(self, data):
        if isinstance(data, torch.Tensor):
            return data

    def add_data(self, name: str, data, **config):
        r""" Configures and parametrizes the internal container model with name and data
        """
        local_args = copy.deepcopy(self.defaults)
        local_args.update(config)
        self.data_groups[name] = local_args

        weight = self._extract_weight(data)

        # Bookkeeping in the container class
        mask = local_args.get('mask', torch.ones_like(weight))
        param_class = local_args.get('parametrization', utils.FakeSparsity)  # change once public_api for utils is fixed!
        param = nn.Parameter(weight, requires_grad=False)

        if name in self.state:
            # If the named data already exists - replace
            warnings.warn("Replacing existing data of the same name. - Did you mean a different name?")
            # check if parametrized
            if parametrize.is_parametrized(self._container, name):
                # If parametrized, squash mask
                self.squash_mask(names=[name], leave_parametrized=False)
            self._container.get_parameter(name).data = weight  # overwrite the data
        else:
            setattr(self._container, name, param)
        parametrize.register_parametrization(self._container, name, param_class(mask))
        self.state[name]['mask'] = mask
        return getattr(self._container, name)

    def get_data(self, name: str, return_original: bool = True):
        r"""Returns weight tensor (or data)
        Args:
            - name: name of the data to be returned
            - return_original returns weight tensor without applying parametrization if True
                else - returns the sparsified version (parametrized)
        """
        if name not in self.data_groups:
            raise ValueError("data with specified name does not exist")

        if return_original:
            if not parametrize.is_parametrized(self._container, name):
                raise ValueError("mask squashed - original mask value does not exist")
            data = getattr(self._container.parametrizations, name).original
            return data
        else:
            return getattr(self._container, name)

    def __repr__(self):
        format_string = self.__class__.__name__ + ' ('
        for name, sparse_args in self.data_groups.items():
            format_string += '\n'
            format_string += '\tData Group\n'
            format_string += f'\t    name: {name}\n'
            for key in sorted(sparse_args.keys()):
                if key == 'data':
                    continue
                format_string += f'\t    {key}: {sparse_args[key]}\n'
        format_string += ')'
        return format_string

    def get_mask(self, name: str):
        if name not in self.state:
            raise ValueError("data with specified name does not exist")
        return self.state[name]['mask']

    def squash_mask(self, *args, leave_parametrized=True, names=None, **kwargs):
        r"""Squashes the sparse masks into the appropriate tensors. Also, accepts list of strings
        to squash mask for. If none, squashes mask for all the keys
        kwargs:
            * names: list of strings to squash mask for
            * sparsified: if true - applies the mask before squashing
                          if false - does not apply the mask before squashing
        """
        if names is None:
            names = list(self.data_groups.keys())
        for name in names:
            parametrize.remove_parametrizations(self._container, name, leave_parametrized=leave_parametrized)

    def step(self):
        if not self.enable_mask_update:
            return
        with torch.no_grad():
            for name, config in self.data_groups.items():
                # get non-sparsified data
                data = self.get_data(name)
                # need name for the mask otherwise can directly pass mask?
                self.update_mask(name, data, **config)

    @abc.abstractmethod
    def update_mask(self, name, data, **kwargs):
        pass
