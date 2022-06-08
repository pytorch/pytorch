import abc
import copy
from collections import defaultdict
from typing import Dict, Optional, Tuple, List, Any

import torch
from torch import nn
from torch.nn.utils import parametrize
import warnings
from torch.ao.sparsity.sparsifier.utils import FakeSparsity


NON_TENSOR_TYPES = {
    nn.Embedding,
    nn.EmbeddingBag,
}

SUPPORTED_TYPES = {
    *NON_TENSOR_TYPES,
    torch.Tensor,
    nn.Parameter
}


class Container(nn.Module):
    def __init__(self):
        super().__init__()


class BaseDataSparsifier(abc.ABC):
    r"""
    Base Data Sparsifier class for all Data sparsifiers.
    The abstract class accepts raw torch tensors / embedding / embedding bags (refer to SUPPORTED_TYPES above)
    to prepare for sparsification.
    In this case, mask (and parametrizations) is owned by the class and not by the user. 
    Specifically, the container object inside the class maintains the mask and parametrizations of the input data

    Abstract methods that need to be implemented:
    - update_mask: Function to compute a new mask for all keys in the
        `data_groups`.
        Should contain at least 2 arguments - name, data.

    Args:
        - data_list [List[Tuple[str, Any]]]: list of (name, data) tuples to sparsify.
        Currently - operates on embeddings and torch tensors, parameters
        Internally, a container module handles the data sparsification. 

        - defaults [dict]: default configurations will be attached to the
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
        super().__init__()
        self.defaults = defaults
        if self.defaults is None:
            self.defaults = dict()
        self._container = Container()
        self.state: Dict[str, Dict] = defaultdict(dict)  # name -> {mask}
        self.data_groups = {}   # name -> {**config}
        self.enable_mask_update = True
        if data_list is not None:
            # add data with default config here
            [self.add_data(name, data, **self.defaults) for name, data in data_list]

    def add_data(self, name: str, data: SUPPORTED_TYPES, **config):
        r""" Configures and parametrizes the internal container model with name and data

        Note: The container model is private to the BaseDataSparsifier class
        """
        assert type(data) in SUPPORTED_TYPES, \
            f'data type unsupported for {name}'

        if name in self.state:
            # If the named data already exists - replace
            warnings.warn("Replacing existing data of the same name. - Did you mean a different name?")
        local_args = copy.deepcopy(self.defaults)
        local_args.update(config)
        self.data_groups[name] = local_args

        # Bookkeeping in the container class
        if type(data) in NON_TENSOR_TYPES:
            weight = data.weight
        else:
            weight = data

        mask = local_args.get('mask', torch.ones_like(weight))
        param_class = local_args.get('parametrization', FakeSparsity)
        param = nn.Parameter(weight, requires_grad=False)
        setattr(self._container, name, param)
        parametrize.register_parametrization(self._container, name, param_class(mask))
        self.state[name]['mask'] = mask
        return getattr(self._container, name)

    def __getstate__(self):
        return {
            'defaults': self.defaults,
            'state': self.state,
            'data_groups': self.data_groups,
            'container_state': self._container.state_dict()
        }

    def get_data(self, name: str, return_sparsified: bool = True):
        r"""Returns weight tensor (or data)
        Args:
            - name: name of the data to be returned
            - return_sparsified: returns weight tensor after applying parametrization if True
                else - returns the original version (non-parametrized)
        """
        if name not in self.data_groups:
            raise ValueError("data with specified name does not exist")

        if not return_sparsified:
            if not parametrize.is_parametrized(self._container, name):
                raise ValueError("mask squashed - origina mask value does not exist")
            data = getattr(self._container.parametrizations, name).original
            return data
        else:
            return getattr(self._container, name)

    def get_mask(self, name):
        if name not in self.state:
            raise ValueError("data with specified name does not exist")
        return self.state[name]['mask']

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __repr__(self):
        format_string = self.__class__.__name__ + ' ('
        for name, sparse_args in self.data_groups.items():
            format_string += '\n'
            format_string += f'\Data Group\n'
            format_string += f'\t    name: {name}\n'
            for key in sorted(sparse_args.keys()):
                if key == 'data':
                    continue
                format_string += f'\t    {key}: {sparse_args[key]}\n'
        format_string += ')'
        return format_string

    def state_dict(self):
        r"""Returns the state of the optimizer as a :class:`dict`.

        It contains:
        * state - contains name -> mask mapping.
        * data_groups - a list containing all sparsity configuration groups
            with the key name specifying the name of the data
        * container_state_dict - the state dictionary of the internal
            container model used for sparsification
        """
        return {
            'state': self.state,
            'data_groups': self.data_groups,
            'container_state_dict': self._container.state_dict()
        }

    def load_state_dict(self, state_dict, strict=True):
        r"""The load_state_dict() restores the state of the sparsifier based on the state_dict

        Args:
        * state_dict - the dictionary that to which the current sparsifier needs to be restored to
        * strict - If True - the sparsifier is reset and is restored exactly to the state in state_dict.
            If False - the current sparsifier is not reset before loading the state_dict i.e. data added
            before loading the state_dict is not erased. 
        """
        states = state_dict['state']
        data_groups = copy.deepcopy(state_dict['data_groups'])
        container_state_dict = state_dict['container_state_dict']

        if strict:
            # if strict load -> then reset container
            self._container = Container()

        for name, state in states.items():
            config_name = data_groups.get(name, None)
            if config_name is None:
                raise RuntimeError(f"Error loading {name}")

            parametrized_name = f'parametrizations.{name}.original'
          
            parametrized = False
            data = container_state_dict.get(name, None)
            if name in container_state_dict:
                # the parametrization was probably removed for this
                data = container_state_dict.get(name)

            elif parametrized_name in container_state_dict:
                # so the weight was parametrized
                data = container_state_dict.get(parametrized_name)
                parametrized = True

            if data is None:
                raise RuntimeError(f"Error loading {name}")

            param = nn.Parameter(data, requires_grad=False)
            setattr(self._container, name, param)

            if parametrized:
                # register parameter if parametrized
                mask = state.get('mask', torch.ones_like(data))
                param_class = data_groups.get('parametrization', FakeSparsity)
                parametrize.register_parametrization(self._container, name, param_class(mask))

        if not strict:
            states.update(self.state)
            data_groups.update(self.data_groups)
          
        self.__setstate__({'state': states, 'data_groups': data_groups,
                        'container_state_dict': self._container.state_dict()})

    def squash_mask(self, *args, **kwargs):
        r"""Squashes the sparse masks into the appropriate tensors.
        """
        for name in self.data_groups.keys():
            parametrize.remove_parametrizations(self._container, name, leave_parametrized=True)
       
    def convert(self):
        # TODO: Call the torch.ao.utils.convert in here
        raise NotImplementedError('`convert` is not implemented. Please, use '
                                  '`torch.ao.utils.convert` instead.')

    def step(self, use_path=True):
        if not self.enable_mask_update:
            return
        with torch.no_grad():
            for name, config in self.data_groups.items():
                # get non-sparsified data
                data = self.get_data(name, return_sparsified=False)
                # need name for the mask otherwise can directly pass mask?
                self.update_mask(name, data, **config)

    @abc.abstractmethod
    def update_mask(self, name, data, **kwargs):
        pass


__all__ = [Dict, Any, FakeSparsity, List, Optional, Tuple, defaultdict, 'BaseDataSparsifier', 'Container']
