# mypy: allow-untyped-defs
import abc
import copy
from collections import defaultdict
from typing import Any, Optional

import torch
from torch import nn
from torch.nn.utils import parametrize
from torch.nn.utils.parametrize import type_before_parametrizations

from .utils import (
    FakeSparsity,
    get_arg_info_from_tensor_fqn,
    module_contains_param,
    module_to_fqn,
    swap_module,
)


__all__ = ["BaseSparsifier"]

SUPPORTED_MODULES = {nn.Linear}

KEYS_NOT_IN_STATE_DICT = ["module", "module_fqn", "tensor_name"]


# TODO update desc with new config args
class BaseSparsifier(abc.ABC):
    r"""Base class for all sparsifiers.

    Abstract methods that need to be implemented:

    - update_mask: Function to compute a new mask for all keys in the
        `groups`.

    Args:
        - model [nn.Module]: model to configure. The model itself is not saved
            but used for the state_dict saving / loading.
        - config [list]: configuration elements should be a dict map that includes
            `tensor_fqn` of tensors to sparsify
        - defaults [dict]: default configurations will be attached to the
            configuration. Only the keys that don't exist in the `config` will
            be updated.

    Example::

        >>> # xdoctest: +SKIP("Can't instantiate abstract class BaseSparsifier with abstract method update_mask")
        >>> config = [{'tensor_fqn': 'layer1.weight', 'tensor_fqn': 'linear2.weight2', 'sparsity_level': 0.5}]
        >>> defaults = {'sparsity_level': 0.7}
        >>> # model.layer1.weight will have `sparsity_level` = 0.7 (getting default)
        >>> sparsifier = BaseSparsifier(config, defaults)
    """

    def __init__(self, defaults: Optional[dict[str, Any]] = None):
        super().__init__()
        self.defaults: dict[str, Any] = defaults or {}

        self.state: dict[str, dict] = defaultdict(dict)
        self.groups: list[dict[str, Any]] = []
        self.enable_mask_update = True

    def __getstate__(self) -> dict[str, Any]:
        return {
            "defaults": self.defaults,
            "state": self.state,
            "groups": self.groups,
        }

    def __setstate__(self, state: dict[str, dict[str, Any]]) -> None:
        self.__dict__.update(state)

    def __repr__(self):
        format_string = self.__class__.__name__ + " ("
        for i, sparse_args in enumerate(self.groups):
            module = sparse_args["module"]
            format_string += "\n"
            format_string += f"\tGroup {i}\n"
            format_string += f"\t    module: {module}\n"
            for key in sorted(sparse_args.keys()):
                if key == "module":
                    continue
                format_string += f"\t    {key}: {sparse_args[key]}\n"
        format_string += ")"
        return format_string

    def state_dict(self) -> dict[str, Any]:
        r"""Returns the state of the optimizer as a :class:`dict`.

        It contains:
        * state - current state of the sparsification.
        * groups - a list containing all sparsity configuration groups
            with the key 'tensor_fqn' specifying the path to the sparsified tensor within a model

        TODO: Need a clean way of loading the state of the "prepared" module
        """

        groups: list[dict[str, Any]] = [
            dict(
                filter(
                    lambda key_value: key_value[0] not in KEYS_NOT_IN_STATE_DICT,
                    mg.items(),
                )
            )
            for mg in self.groups
        ]

        return {
            "state": self.state,
            "groups": groups,
        }

    def load_state_dict(self, state_dict: dict[str, Any], strict: bool = True):
        groups = copy.deepcopy(state_dict["groups"])
        states = state_dict["state"]
        for tensor_fqn, s in states.items():
            arg_info = get_arg_info_from_tensor_fqn(self.model, tensor_fqn)
            module = arg_info["module"]
            tensor_name = arg_info["tensor_name"]
            if strict and module is None:
                raise RuntimeError(f"Error loading {tensor_fqn} into the model")

            found = False
            for p in module.parametrizations[tensor_name]:
                if isinstance(p, FakeSparsity):
                    found = True
                    break
            if not found:
                p = FakeSparsity(torch.ones(getattr(module, tensor_name).shape))
                parametrize.register_parametrization(module, tensor_name, p)
            if s.get("mask", None) is not None:
                mask = s.pop("mask")
                p.mask = mask

            for mg in groups:
                if mg["tensor_fqn"] == tensor_fqn:
                    mg.update(arg_info)
        self.__setstate__({"state": states, "groups": groups})

    def make_config_from_model(
        self,
        model: nn.Module,
        SUPPORTED_MODULES: set[type[nn.Linear]] = SUPPORTED_MODULES,
    ) -> None:
        self.config = []
        stack = [model]
        while stack:
            module = stack.pop()
            for _name, child in module.named_children():
                if type(child) in SUPPORTED_MODULES:
                    module_fqn = module_to_fqn(model, child)
                    if not isinstance(module_fqn, str):
                        raise AssertionError("module_fqn must be a string")
                    self.config.append({"tensor_fqn": module_fqn + ".weight"})
                else:
                    stack.append(child)

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
            self.make_config_from_model(model)

        # TODO: Remove the configuration by reference ('module')
        # pyrefly: ignore [not-iterable]
        for module_config in self.config:
            if not isinstance(module_config, dict):
                raise AssertionError(
                    "config elements should be dicts not modules i.e.:"
                    "[{`tensor_fqn`: `foo.bar.weight`}, {`tensor_fqn`: ... }, ...]"
                )

            if not isinstance(self.defaults, dict):
                raise AssertionError("defaults must be a dict")
            local_args = copy.deepcopy(self.defaults)
            local_args.update(module_config)

            tensor_fqn = local_args.get("tensor_fqn", None)
            if tensor_fqn is None:
                raise AssertionError(
                    "tensor_fqn is a required argument in the sparsity config which"
                    "replaces previous `module` and [module]`fqn` arguments"
                )

            # populate all information from tensor_fqn
            info_from_tensor_fqn = get_arg_info_from_tensor_fqn(model, tensor_fqn)

            # check that whatever was put into local_args agrees with what was obtained
            # from tensor_fqn
            for key in info_from_tensor_fqn:
                if key in local_args:
                    if not (
                        info_from_tensor_fqn[key] == local_args[key]
                        or (
                            key == "tensor_fqn"
                            and "." + info_from_tensor_fqn[key] == local_args[key]
                        )
                        # info_from_tensor_fqn will chop leading '.' from tensor_fqn so ignore that
                    ):
                        raise AssertionError(
                            f"Given both `{key}` and `tensor_fqn` in the config, it is expected them to agree!"
                        )
            local_args.update(info_from_tensor_fqn)
            self.groups.append(local_args)
        self._prepare()

    def _prepare(self, *args, **kwargs):
        r"""Adds mask parametrization to the layer weight"""
        for config in self.groups:
            module = config["module"]
            tensor_name = config["tensor_name"]
            parametrization = config.get("parametrization", FakeSparsity)
            mask = config.get("mask", torch.ones_like(getattr(module, tensor_name)))
            self.state[config["tensor_fqn"]]["mask"] = mask
            parametrize.register_parametrization(
                module, tensor_name, parametrization(mask)
            )

    def squash_mask(
        self,
        params_to_keep: Optional[tuple[str, ...]] = None,
        params_to_keep_per_layer: Optional[dict[str, tuple[str, ...]]] = None,
        *args,
        **kwargs,
    ):
        r"""Squashes the sparse masks into the appropriate tensors.

        If either the `params_to_keep` or `params_to_keep_per_layer` is set,
        the module will have a `sparse_params` dict attached to it.

        Args:
            params_to_keep: List of keys to save in the module or a dict
                            representing the modules and keys that will have
                            sparsity parameters saved
            params_to_keep_per_layer: Dict to specify the params that should be
                            saved for specific layers. The keys in the dict
                            should be the module fqn, while the values should
                            be a list of strings with the names of the variables
                            to save in the `sparse_params`

        Examples:
            >>> # xdoctest: +SKIP("locals are undefined")
            >>> # Don't save any sparse params
            >>> sparsifier.squash_mask()
            >>> hasattr(model.submodule1, "sparse_params")
            False

            >>> # Keep sparse params per layer
            >>> sparsifier.squash_mask(
            ...     params_to_keep_per_layer={
            ...         "submodule1.linear1": ("foo", "bar"),
            ...         "submodule2.linear42": ("baz",),
            ...     }
            ... )
            >>> print(model.submodule1.linear1.sparse_params)
            {'foo': 42, 'bar': 24}
            >>> print(model.submodule2.linear42.sparse_params)
            {'baz': 0.1}

            >>> # Keep sparse params for all layers
            >>> sparsifier.squash_mask(params_to_keep=("foo", "bar"))
            >>> print(model.submodule1.linear1.sparse_params)
            {'foo': 42, 'bar': 24}
            >>> print(model.submodule2.linear42.sparse_params)
            {'foo': 42, 'bar': 24}

            >>> # Keep some sparse params for all layers, and specific ones for
            >>> # some other layers
            >>> sparsifier.squash_mask(
            ...     params_to_keep=("foo", "bar"),
            ...     params_to_keep_per_layer={"submodule2.linear42": ("baz",)},
            ... )
            >>> print(model.submodule1.linear1.sparse_params)
            {'foo': 42, 'bar': 24}
            >>> print(model.submodule2.linear42.sparse_params)
            {'foo': 42, 'bar': 24, 'baz': 0.1}
        """
        for config in self.groups:
            module = config["module"]
            tensor_name = config["tensor_name"]
            parametrize.remove_parametrizations(
                module, tensor_name, leave_parametrized=True
            )
            sparse_params = {}
            if params_to_keep is not None:
                global_params = {k: config[k] for k in params_to_keep}
                sparse_params.update(global_params)
            if params_to_keep_per_layer is not None:
                params = params_to_keep_per_layer.get(config["module_fqn"], None)
                if params is not None:
                    per_layer_params = {k: config[k] for k in params}
                    sparse_params.update(per_layer_params)
            if sparse_params:
                # TODO handle multiple tensor being quantized on a single module, where to store sparse_params?
                module.sparse_params = sparse_params

    def convert(
        self,
        module: nn.Module,
        mapping: Optional[dict[type[nn.Module], type[nn.Module]]] = None,
        inplace: bool = False,
        parameterization: type[nn.Module] = FakeSparsity,
    ):
        r"""Converts submodules in input module to a different module according to `mapping`
        by calling `from_dense` method on the target module class
        Args:
            module: input module
            mapping: a dictionary that maps from source module type to target
                module type, can be overwritten to allow swapping user defined
                Modules
            inplace: carry out model transformations in-place, the original module
                is mutated
        """
        if mapping is None:
            raise NotImplementedError("Need to auto generate mapping ")
        if not inplace:
            module = copy.deepcopy(module)

        reassign = {}
        for name, mod in module.named_children():
            # leaf node
            if (
                module_contains_param(mod, parameterization)
                and type_before_parametrizations(mod) in mapping
            ):
                reassign[name] = swap_module(mod, mapping)
            else:
                # recurse
                reassign[name] = self.convert(
                    mod,
                    mapping=mapping,
                    inplace=True,
                    parameterization=parameterization,
                )

        for key, value in reassign.items():
            module._modules[key] = value

        return module

    def step(self, use_path: bool = True) -> None:
        if not self.enable_mask_update:
            return
        with torch.no_grad():
            for config in self.groups:
                self.update_mask(**config)

    @abc.abstractmethod
    def update_mask(self, module: nn.Module, tensor_name: str, **kwargs):
        pass
