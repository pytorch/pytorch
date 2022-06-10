from typing import Optional, Tuple, List, Any, Dict
from ...sparsifier import base_sparsifier
from collections import defaultdict
from torch import nn


class Container(nn.Module):
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
        - data_list [List[Tuple[str, Any]]]: list of (name, data) tuples to sparsify.
        Currently - operates on embeddings and torch tensors, parameters
        Internally, a container module handles the data sparsification.

        - defaults [dict]: default configurations will be attached to the
            configuration. Only the keys that don't exist in the `config` will
            be updated.
    """
    def __init__(self, data_list: Optional[List[Tuple[str, Any]]] = None, **defaults):
        super().__init__(defaults=defaults)

        self._container = Container()

        self.data_groups: Dict[str, Dict] = defaultdict(dict)  # name -> {**config}
        if data_list is not None:
            # add data with default config here
            [self.add_data(name, data, **self.defaults) for name, data in data_list]

    def prepare(self):
        raise NotImplementedError("this function is undefined for this class")

    def add_data(self, name: str, data, **config):
        r""" Configures and parametrizes the internal container model with name and data

        Note: The container model is private to the BaseDataSparsifier class
        """
        pass

    def get_data(self, name: str):
        pass

    def state_dict(self):
        pass

    def load_state_dict(self, state_dict):
        pass

    def __setstate__(self, state):
        pass

    def __getstate__(self):
        pass

    def __repr__(self):
        pass

    def get_mask(self, name: str):
        pass

    def squash_mask(self, *args, **kwargs):
        pass

    def step(self):
        pass

    def update_mask(self, name, data, **kwargs):
        pass
