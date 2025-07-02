# mypy: allow-untyped-defs
import abc
import copy
import sys
import warnings
from collections import defaultdict
from typing import Any, Optional

import torch
from torch import nn
from torch.ao.pruning.sparsifier import base_sparsifier, utils
from torch.nn.utils import parametrize


if not sys.warnoptions:
    # to suppress repeated warnings when being used in a training loop.
    warnings.simplefilter("once")

__all__ = ["BaseDataSparsifier"]

EMBEDDING_TYPES = {
    nn.Embedding,
    nn.EmbeddingBag,
}

SUPPORTED_TYPES = {
    torch.Tensor,
    nn.Parameter,
    *EMBEDDING_TYPES,
}


class _Container(nn.Module):
    pass


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
        >>> # xdoctest: +SKIP
        >>> data_list = [('tensor_1', torch.randn(3,3)), ('tensor_2', torch.randn(4,4))]
        >>> defaults = {'sparsity_level': 0.7}
        >>> sparsifier = DerivedDataSparsifier(data_list = data_list, **defaults) # Some sparsifier that inherits BaseDataSparsifier
        >>> new_tensor_to_add = {'name': 'tensor_3', 'data': torch.randn(5,5), 'sparsity_level': 0.3}
        >>> sparsifier.add_data(**new_tensor_to_add)
        >>> # tensor_1 and tensor_2 will have sparsity_level of 0.7 but tensor_3 will have sparsity_level=0.3
    """

    def __init__(self, data_list: Optional[list[tuple[str, Any]]] = None, **defaults):
        super().__init__(defaults=defaults)

        self._container = _Container()

        self.data_groups: dict[str, dict] = defaultdict(dict)  # name -> {**config}
        if data_list is not None:
            # add data with default config here
            [self.add_data(name, data, **self.defaults) for name, data in data_list]

    def prepare(self, model, config):
        raise NotImplementedError("this function is undefined for this class")

    def _extract_weight(self, data):
        # extract the weight parameter instead of underlying data
        if type(data) in [torch.Tensor, nn.Parameter]:
            return data
        elif type(data) in EMBEDDING_TYPES:
            return data.weight

    def add_data(self, name: str, data, reuse_mask=True, **config):
        r"""Configures and parametrizes the internal container model with name and data.

        **Note**:
            1. If the data with name already exists, it replaces the data.
            2. While replacing, the old mask is reused when `reuse_mask=True`
            3. If `reuse_mask=True`, then the replacing data needs to have the same shape as that of old data.
            4. By default, the config of the replaced data is used as config for the replacing data, unless something
               is specified in the config dictionary.
        """
        assert type(data) in SUPPORTED_TYPES, (
            "specified data type not supported at the moment"
        )
        local_args = copy.deepcopy(self.defaults)
        local_args.update(config)
        weight = self._extract_weight(data)

        # Bookkeeping in the container class
        mask = local_args.get("mask", torch.ones_like(weight))
        param_class = local_args.get("parametrization", utils.FakeSparsity)

        if name in self.state:
            # If the named data already exists - replace
            warnings.warn(
                "Replacing existing data of the same name. - Did you mean a different name?"
            )

            # reuse old config
            old_args = self.data_groups[name]
            local_args = copy.deepcopy(old_args)
            local_args.update(config)

            if reuse_mask:
                current_data = self.get_data(name=name)
                assert weight.shape == current_data.shape, (
                    "to retain the old mask, the shape of the new data must be the same as the previous one"
                )
                mask = self.get_mask(
                    name=name
                )  # reuse mask instead of creating a new one

            self._delete_data(name=name)

        # parameter creates a deepcopy of the weight inside, so create a buffer
        self._container.register_buffer(name=name, tensor=weight)
        parametrize.register_parametrization(self._container, name, param_class(mask))
        self.state[name]["mask"] = mask
        self.data_groups[name] = local_args
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

    def _convert_mask(self, states, sparse_coo=True):
        r"""Converts the mask to sparse coo or dense tensors depending on the `sparse_coo` argument."""
        states = copy.deepcopy(states)
        for state in states.values():
            if sparse_coo:
                state["mask"] = state["mask"].to_sparse_coo()
            else:
                state["mask"] = state["mask"].to_dense()

        return states

    def state_dict(self):
        r"""Returns the state of the optimizer as a :class:`dict`.

        It contains:
        * state - contains name -> mask mapping.
        * data_groups - a list containing all sparsity configuration groups
            with the key name specifying the name of the data
        * container_state_dict - the state dictionary of the internal
            container model used for sparsification
        """
        state = self._convert_mask(self.state)
        return {
            "state": state,
            "data_groups": self.data_groups,
            "_container": self._container.state_dict(),
        }

    def _load_container_from_state(self, states, data_groups, container_state_dict):
        r"""This restores the state of the container specifically based on the data present in state and data_groups
        If the data was parametrized, then the data would be added to the container and then parametrized,
        else it would just add the attribute the container.
        """
        for name, state in states.items():
            config_name = data_groups.get(name, None)
            if config_name is None:
                raise RuntimeError(f"Error loading {name}")

            # check if the data with such a name was parametrized, if so parametrize
            # otherwise just set the attribute and continue
            parametrized_name = f"parametrizations.{name}.original"
            parametrized = False
            data = container_state_dict.get(name, None)
            if name in container_state_dict:
                # the parametrization was probably removed for this
                data = container_state_dict.get(name)

            elif parametrized_name in container_state_dict:
                # so the weight was parametrized
                data = container_state_dict.get(parametrized_name)
                parametrized = True

            else:
                raise RuntimeError(f"Error loading {name}")

            self._container.register_buffer(name=name, tensor=data)

            if parametrized:
                # register parameter if parametrized
                mask = state.get("mask", torch.ones_like(data))
                param_class = data_groups.get(
                    "parametrization", utils.FakeSparsity
                )  # change once public_api for utils is fixed!
                parametrize.register_parametrization(
                    self._container, name, param_class(mask)
                )

    def load_state_dict(self, state_dict, strict=True):
        r"""The load_state_dict() restores the state of the sparsifier based on the state_dict

        Args:
        * state_dict - the dictionary that to which the current sparsifier needs to be restored to
        * strict - If True - the sparsifier is reset and is restored exactly to the state in state_dict.
            If False - the current sparsifier is not reset before loading the state_dict i.e. data added
            before loading the state_dict is not erased.
        """
        states = copy.deepcopy(state_dict["state"])
        data_groups = copy.deepcopy(state_dict["data_groups"])
        container_state_dict = copy.deepcopy(state_dict["_container"])

        states = self._convert_mask(
            states, sparse_coo=False
        )  # convert sparse coo mask to dense
        if strict:
            # if strict load -> then reset container
            self._container = _Container()

        self._load_container_from_state(states, data_groups, container_state_dict)

        if not strict:
            states.update(self.state)
            data_groups.update(self.data_groups)

        self.__setstate__({"state": states, "data_groups": data_groups})

    def __setstate__(self, state):
        if "_container" in state:  # If container object is in state then load model
            container_dict = state.pop("_container")
            self._container = _Container()
            state["state"] = self._convert_mask(
                state["state"], sparse_coo=False
            )  # convert sparse coo mask to dense
            self._load_container_from_state(
                state["state"], state["data_groups"], container_dict
            )

        self.__dict__.update(state)

    def __getstate__(self):
        state = self._convert_mask(self.state)
        return {
            "defaults": self.defaults,
            "state": state,
            "data_groups": self.data_groups,
            "_container": self._container.state_dict(),
        }

    def __repr__(self):  # type:ignore[override]
        format_string = self.__class__.__name__ + " ("
        for name, sparse_args in self.data_groups.items():
            format_string += "\n"
            format_string += "\tData Group\n"
            format_string += f"\t    name: {name}\n"
            for key in sorted(sparse_args.keys()):
                if key == "data":
                    continue
                format_string += f"\t    {key}: {sparse_args[key]}\n"
        format_string += ")"
        return format_string

    def get_mask(self, name: str):
        if name not in self.state:
            raise ValueError("data with specified name does not exist")
        return self.state[name]["mask"]

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
            parametrize.remove_parametrizations(
                self._container, name, leave_parametrized=leave_parametrized
            )

    def step(self):  # type:ignore[override]
        if not self.enable_mask_update:
            return
        with torch.no_grad():
            for name, config in self.data_groups.items():
                # get non-sparsified data
                data = self.get_data(name)
                # need name for the mask otherwise can directly pass mask?
                self.update_mask(name, data, **config)

    @abc.abstractmethod
    def update_mask(self, name, data, **kwargs):  # type: ignore[override]
        pass

    def _delete_data(self, name):
        """Detaches some data from the sparsifier.

        Args:
            name (str)
                Name of the data to be removed from the sparsifier

        Note:
            Currently private. Kind of used as a helper function when replacing data of the same name
        """
        self.squash_mask(
            names=[name], leave_parametrized=False
        )  # do not apply the mask while deleting
        delattr(self._container, name)
        self.state.pop(name)
        self.data_groups.pop(name)
