import abc
from typing import Optional, Tuple, List, Any, Dict
from ...sparsifier import base_sparsifier
from collections import defaultdict
from torch import nn
__all__ = ['BaseDataSparsifier']


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

    def add_data(self, name: str, data, **config):
        r""" Configures and parametrizes the internal container model with name and data
        """
        pass

    def get_data(self, name: str):
        r"""Returns weight tensor (or data) based on the input name.
        """
        pass

    def __repr__(self):
        r"""String representation of an object when printed
        """
        pass

    def get_mask(self, name: str):
        r"""Returns the mask currently associated with the named tensor.
        """
        pass

    def squash_mask(self, *args, **kwargs):
        r"""Squashes the sparse masks into the appropriate tensors.
        """
        pass

    def step(self):
        r"""Updates the mask for all the named data.
        """
        pass

    @abc.abstractmethod
    def update_mask(self, name, data, **kwargs):
        pass
