# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) Tongzhou Wang
# Licensed under the MIT License.

from contextlib import contextmanager
from itertools import chain
import typing
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Generator,
    Iterator,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
    cast,
)

import torch
from torch import Tensor
import torch.nn as nn

from .utils import replace_by_prefix_

if TYPE_CHECKING:
    from collections import OrderedDict  # noqa: F401


class FlatParameter(nn.Parameter):
    """ A parameter that is initialized from a list of parameters and can be
        turned into a list of views as needed.
    """

    def __new__(cls, params: Sequence[nn.Parameter], requires_grad: bool = True) -> "FlatParameter":
        """ Make an object using the parent's __new__ function. """

        # A empty of non-list input doesn't make sense.
        if not isinstance(params, (list, tuple)) or len(params) == 0:
            raise ValueError("An non-empty list or tuple argument is needed")

        # Normally, all items are Parameters. But during pickling, we will have a single
        # Tensor as the input and later in __init__, the correct _param_numels and _param_shapes
        # are set.
        if not all(isinstance(p, (nn.Parameter, Tensor)) for p in params):
            raise ValueError("List items need to be Parameter types")

        # Flattening involves (1) making a tensor flat (i.e. single dimensional) and (2) making a module
        # hierarchy flat (using a single tensor to replace a tree of tensors). Therefore,
        # adding back nesting and hierarchy is counter-productive. If nesting is encountered
        # in the future, the reasonable thing to do is likely for the top level FlatParameter to
        # absorb the nested one and keep the result flat, free from hierarchy.
        if any(isinstance(p, FlatParameter) for p in params):
            raise ValueError("Nesting FlatParameter is not supported")

        data = torch.cat([p.detach().reshape(-1) if isinstance(p, nn.Parameter) else p.reshape(-1) for p in params], 0)
        return super(FlatParameter, cls).__new__(cls, data, requires_grad=requires_grad)  # type: ignore[call-arg]

    def __init__(self, params: Sequence[nn.Parameter], requires_grad: bool = True):
        """ Initialize the _param_numels and _param_shapes lists. """
        self._param_numels = [p.numel() for p in params]
        assert self.numel() <= sum(
            self._param_numels
        ), f"Something wrong with __new__ method, {self.numel()} vs. {sum(self._param_numels)}"
        self._param_shapes = [p.size() for p in params]

        # These are set by FPW class below, not by this class itself.
        self._param_infos: List[Tuple[str, nn.Module, str]] = []
        self._shared_param_infos: List[Tuple[str, str, nn.Module, str, nn.Module, str]] = []

    def get_param_views(self, external_data: Optional[Tensor] = None) -> Iterator[Tensor]:
        """ Return a generator of views that map to the original parameters. """
        # Note, self.data could be sharded, so its numel is <= to the sum.
        assert self.data.numel() <= sum(
            self._param_numels
        ), f"Incorrect internal state {self.data.numel()} vs. {sum(self._param_numels)}"
        data = external_data if external_data is not None else self
        if data.numel() != sum(self._param_numels):
            raise ValueError(
                f"Incorrect numel of supplied data: got {data.numel()} but expected {sum(self._param_numels)}"
            )
        return (t.view(s) for (t, s) in zip(data.split(self._param_numels), self._param_shapes))

    def metadata(self) -> Tuple[List[str], List[torch.Size], List[int]]:
        """Return tuple of (names, shapes, numels) metadata for this flat parameter."""
        names = [".".join([m, n]) if m else n for (m, _, n) in self._param_infos]
        return names, self._param_shapes, self._param_numels

    def __setstate__(self, state: Tuple[Any, Any, Any, Any]) -> None:
        """ Use by pickle to set the internal states. """
        (self._param_numels, self._param_shapes, self._param_infos, self._shared_param_infos) = state
        assert self.numel() <= sum(
            self._param_numels
        ), f"Incorrect pickling {self.numel()} vs. {sum(self._param_numels)}"

    def __reduce_ex__(self, proto: int) -> Tuple[Any, Any, Any]:
        """ Support pickling between ranks. """
        return (
            FlatParameter,  # Callable
            # Args to the callable above
            ([self.data], self.requires_grad),
            # Args to __setstate__
            (self._param_numels, self._param_shapes, self._param_infos, self._shared_param_infos),
        )


# Static types.
ParamGroups = Optional[Union[List[List[nn.Parameter]], List[nn.Parameter]]]


class FlattenParamsWrapper(nn.Module):
    """
    A wrapper for transparently flattening a Module's parameters.
    Compared to the original implementation [1], this version:
    - removes tracing
    - supports shared parameters
    - handles state_dict/load_state_dict transparently
    - is renamed to FlattenParamsWrapper
    - refactored to use the FlatParameter class
    - extended to support flattening multiple groups of params (useful
      when different groups of params need different hyperparameters, like
      learning rate or weight decay)
    [1] https://github.com/SsnL/PyTorch-Reparam-Module
    Args:
        module (nn.Module):
            The module to wrap.
        param_list (Optional[List[List[nn.Parameter]]]):
            Only flatten parameters appearing in the given groups.
            If the param_list is an empty list, then no parameters will get flattened.
            Note, if a single param is in one of the list, it still get flattened and the
            original param is removed and replaced with the flatten one.
            Default: None, flatten all parameters (if any)
        flat_param_names (Optional[List[str]]):
            originally, give each flat_param a unique name. Note a "flat_param_"
            prefix will be added to those names.
    """

    def __init__(self, module: nn.Module, param_list: ParamGroups = None, flat_param_names: Optional[List[str]] = None):
        super().__init__()
        self._fpw_module = module
        self.is_flattened = False

        # Handle param_list being None.
        if param_list is None:
            param_list = list(module.parameters())

        # Be backward compatible and turn a single param list into a list of
        # a single list.
        if len(param_list) > 0 and isinstance(param_list[0], nn.Parameter):
            param_list = [cast(List[nn.Parameter], param_list)]

        # Since the parameters will be deleted, let's record the number original
        # parameters managed by this class. This and get_param_views function
        # below are used by fsdp_optim_utils.py to save/restore optimizer state,
        # which mirrors the flatten parameters here.
        self.num_params_managed = 0

        self._param_sets = []
        overall_param_set: Set[nn.Parameter] = set()
        for p_list in param_list:
            # Remove any duplicates from the list.
            p_set: Set[nn.Parameter] = set(cast(List[nn.Parameter], p_list))

            self.num_params_managed += len(p_set)
            overall_param_set = overall_param_set.union(p_set)

            # Convert from list of Parameters to set of (Module, name) tuples,
            # which will survive in case the parameter instances are reset.
            # Also, a shared param will correctly appear under multiple modules
            # as they should.
            new_p_set_with_names = set()
            for m in self.modules():
                for n, p in m.named_parameters(recurse=False):
                    if p in p_set:
                        new_p_set_with_names.add((m, n))
            if new_p_set_with_names:
                self._param_sets.append(new_p_set_with_names)

        if len(overall_param_set) != self.num_params_managed:
            # Each p_list above could have shared params. However, you can't
            # have shared params cross different p_list. That means part of
            # the flattened parameter must be shared, which is impossible to
            # support.
            raise ValueError(f"Incorrect param groups {len(overall_param_set)} vs {self.num_param_managed}")

        self.flat_params: List[FlatParameter] = []

        # Prepare flat param names.
        if flat_param_names is None:
            flat_param_names = [f"{i}" for i, _ in enumerate(self._param_sets)]
        if len(flat_param_names) != len(self._param_sets):
            raise ValueError("Names and number of param lists must be equal")
        if len(flat_param_names) != len(set(flat_param_names)):
            raise ValueError("Each flat param must be given a unique name")
        self.flat_param_names = [f"flat_param_{n}" for n in flat_param_names]

        # Init all flat_params.
        for new_p_set in self._param_sets:
            params, param_infos, shared_param_infos = self._init_flatten_params(new_p_set)
            flat_param = FlatParameter(params, params[0].requires_grad)
            flat_param._param_infos = param_infos
            flat_param._shared_param_infos = shared_param_infos
            self.flat_params.append(flat_param)

        self._flatten_params(self.flat_params)

        # Register hook to be called after state_dict() to remove the
        # "_fpw_module." prefix and before load_state_dict() to add it back.
        self._register_state_dict_hook(_post_state_dict_hook)
        self._register_load_state_dict_pre_hook(_pre_load_state_dict_hook)

        # Flag to indicate whether state_dict() should automatically unflatten
        # params. This defaults to True, but may be set to False if the user
        # explicitly requests a flat state dict via flat_state_dict().
        self._auto_unflatten_state_dict = True

    @property
    def module(self) -> Any:
        """ Support fpw.module in case we are immitating DDP, which has .module
            property to the underlying module.
        """
        return self._fpw_module

    @property
    def flat_param(self) -> nn.Parameter:
        """ We used to support only a single flat_param. This allows us to
            be backward compatible.
        """
        assert len(self.flat_params) == 1, "Incorrect access to flat_param"
        return self.flat_params[0]

    def _init_flatten_params(
        self, p_set: Set[Tuple[nn.Module, str]]
    ) -> Tuple[
        List[nn.Parameter], List[Tuple[str, nn.Module, str]], List[Tuple[str, str, nn.Module, str, nn.Module, str]]
    ]:
        """ Build metadata for need-to-be-flatten parameters and returns a list
            contains the need-to-be-flatten parameters.
            This also returns param_infos and shared_param_infos, which
            will be attached to the flat parameter object.
        Args:
            p_set (set):
                A set of (module, param_name) for a set of params that needed
                to be flattened. There could be shared params in this set.
        """
        param_infos = []
        shared_param_memo: Dict[nn.Parameter, Tuple[str, nn.Module, str]] = {}
        shared_param_infos = []
        params = []
        for module_name, m in self.named_modules():
            for n, p in m.named_parameters(recurse=False):
                if p is not None and (m, n) in p_set:
                    if p in shared_param_memo:
                        mname, shared_m, shared_n = shared_param_memo[p]
                        shared_param_infos.append((module_name, mname, m, n, shared_m, shared_n))
                    else:
                        shared_param_memo[p] = (module_name, m, n)
                        param_infos.append((module_name, m, n))
                        params.append(p)
        del shared_param_memo

        assert len(set(p.dtype for p in params)) == 1, "expects all parameters to have same dtype"
        assert len(set(p.requires_grad for p in params)) == 1, "expects all parameters to have same requires_grad"
        assert len(params) == len(set(params)), "params list should not have dups"
        return params, param_infos, shared_param_infos

    @property
    def _param_infos(self) -> Iterator[Tuple[str, nn.Module, str]]:
        return chain(*[p._param_infos for p in self.flat_params])

    @property
    def _shared_param_infos(self) -> Iterator[Tuple[str, str, nn.Module, str, nn.Module, str]]:
        return chain(*[p._shared_param_infos for p in self.flat_params])

    def _flatten_params(self, flat_params: List[FlatParameter]) -> None:
        """ Flatten the managed parameters and replaced the original
            attributes with views to the flat params.
        """
        assert not self.is_flattened
        self.is_flattened = True

        # register the flatten ones and save it to self.
        assert len(self.flat_param_names) == len(flat_params), f"{len(self.flat_param_names)} vs. {len(flat_params)}"
        for n, flat_param in zip(self.flat_param_names, flat_params):
            self.register_parameter(n, flat_param)
        self.flat_params = flat_params

        # deregister the names as parameters
        for _, m, n in self._param_infos:
            delattr(m, n)
        for _, _, m, n, _, _ in self._shared_param_infos:
            delattr(m, n)

        # register the views as plain attributes
        self._unflatten_params_as_views()

    def _unflatten_params(self, external_data: Optional[List[Optional[Tensor]]] = None) -> None:
        """ Undo flattening and create separate parameters from the already flattened
            self.flat_param or a user supplied external data.
        """
        assert self.is_flattened or external_data is not None
        self.is_flattened = False

        ps = self.get_param_views(external_data)
        for (_, m, n), p in zip(self._param_infos, ps):
            if hasattr(m, n):
                delattr(m, n)
            m.register_parameter(n, nn.Parameter(p))
        for (_, _, m, n, shared_m, shared_n) in self._shared_param_infos:
            if hasattr(m, n):
                delattr(m, n)
            m.register_parameter(n, getattr(shared_m, shared_n))

        for n in self.flat_param_names:
            # This ensures the flat params are removed from the module.
            delattr(self, n)
        self.flat_params = []

    def _unflatten_params_as_views(self) -> None:
        """ Unlike ``_unflatten_params``, this function unflatten into views and keep
            self.flat_param unchanged.
        """
        assert self.is_flattened
        ps = self.get_param_views()
        for (_, m, n), p in zip(self._param_infos, ps):
            setattr(m, n, p)  # This will set as plain attr
        for (_, _, m, n, shared_m, shared_n) in self._shared_param_infos:
            setattr(m, n, getattr(shared_m, shared_n))

    @contextmanager
    def unflatten_params(self, flat_params: Optional[List[Tensor]] = None) -> Generator:
        """
        Unflatten params. If the current instance is already unflattened, then
        it will remain unflattened after the context manager exits.
        Args:
            flat_params (List[Tensor], Optional):
                flat params to use for unflattening.
                If provided, the current instance must be in a flattened state
                at the start of the context manager. The provided Tensor must be
                appropriately sized and will only be used within the context
                manager. After the context manager exits, we will revert to
                using ``self.flat_params``
                Default: None.
        """
        assert (
            flat_params is None or self.is_flattened
        ), "Unflattening with external flat_param requires current instance to be flattened"

        orig_flattened = self.is_flattened
        if orig_flattened:
            orig_flat_params = self.flat_params
            self._unflatten_params(cast(Optional[List[Optional[Tensor]]], flat_params))

        # Put yield in a try...finally in case the caller catches the exception and handles
        # it. In that case, we need to properly handle the undoing of state here.
        try:
            yield
        finally:
            if orig_flattened:
                self._flatten_params(orig_flat_params)

    def __getattr__(self, name: str) -> Any:
        """Forward missing attributes to wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.module, name)  # fallback to wrapped module

    def __getitem__(self, key: int) -> Any:
        """Forward indexing calls in case the module is a nn.Sequential."""
        return self.module.__getitem__(key)

    @typing.overload  # type: ignore[override]
    def state_dict(
        self, destination: Mapping[str, Tensor], prefix: str = ..., keep_vars: bool = ...
    ) -> Mapping[str, Tensor]:
        ...

    @typing.overload
    def state_dict(self, prefix: str = ..., keep_vars: bool = ...) -> "OrderedDict[str, Tensor]":
        ...

    # Since we have overloads above, we can use Any here.
    def state_dict(self, *args: Any, **kwargs: Any) -> Any:
        """Return the wrapped module's state_dict."""
        if self.is_flattened and self._auto_unflatten_state_dict:
            # Returns the original version.
            with self.unflatten_params():
                return super().state_dict(*args, **kwargs)
        else:
            # Returns flattened version.
            return super().state_dict(*args, **kwargs)

    def flat_state_dict(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Return the flattened state_dict."""
        assert self.is_flattened
        with self._no_auto_unflatten_state_dict():
            return self.state_dict(*args, **kwargs)

    @contextmanager
    def _no_auto_unflatten_state_dict(self) -> Generator:
        backup = self._auto_unflatten_state_dict
        self._auto_unflatten_state_dict = False
        # Put yield in a try...finally in case the caller catches the exception and handles
        # it. In that case, we need to properly handle the undoing of state.
        try:
            yield
        finally:
            self._auto_unflatten_state_dict = backup

    def load_state_dict(
        self, state_dict: Union[Dict[str, Tensor], "OrderedDict[str, Tensor]"], strict: bool = True
    ) -> NamedTuple:
        """
        Load a state dict. If necessary, ``unflatten_params`` will be called to
        match the input state_dict.
        """
        # unflatten the module automatically if the state_dict is non-flat
        if self.is_flattened and "flat_param_0" not in state_dict:
            # This object is flatten but state_dict is not. So we unflatten and load.
            with self.unflatten_params():
                return super().load_state_dict(state_dict, strict)  # type: ignore[arg-type]
        else:
            # Otherwise, load it as is but make older state dict compatible.
            if "flat_param" in state_dict:
                state_dict["flat_param_0"] = state_dict["flat_param"]
                del state_dict["flat_param"]
            return super().load_state_dict(state_dict, strict)  # type: ignore[arg-type]

    def forward(self, *inputs: Any, **kwinputs: Any) -> Any:
        self._unflatten_params_as_views()
        return self.module(*inputs, **kwinputs)

    def get_param_views(self, external_data_list: Optional[List[Optional[Tensor]]] = None) -> Iterator[Tensor]:
        """ Used to get a generator over all views from a list of external data list. """
        params = self.flat_params
        if external_data_list is None:
            external_data_list = [None] * len(params)
        assert len(external_data_list) == len(
            params
        ), f"Incorrect external data list: {len(external_data_list)} vs. {len(params)}"

        gens = []
        for p, data in zip(params, external_data_list):
            gens.append(p.get_param_views(data))

        return chain(*gens)

    def metadata(self, flat_param_idx: int) -> Tuple[List[str], List[torch.Size], List[int]]:
        """Return metadata for a flat param given its index in the flat_params list."""
        return self.flat_params[flat_param_idx].metadata()


def _post_state_dict_hook(
    module: nn.Module, state_dict: "OrderedDict[str, Tensor]", prefix: str, *args: Any
) -> "OrderedDict[str, Tensor]":
    # Move everything from .fpw_module up one level.
    replace_by_prefix_(state_dict, prefix + "_fpw_module.", prefix)
    return state_dict


def _pre_load_state_dict_hook(
    state_dict: Union[Dict[str, Tensor], "OrderedDict[str, Tensor]"], prefix: str, *args: Any
) -> None:
    # Push everything down to ._fpw_module level.
    replace_by_prefix_(state_dict, prefix, prefix + "_fpw_module.")
    # The flat_param_* keys actually needs to move one level up.
    flat_param_key = prefix + "_fpw_module.flat_param"
    for k in list(state_dict.keys()):
        if k.startswith(flat_param_key):
            last_part = k.split(".")[-1]
            assert last_part.startswith("flat_param_"), last_part
            replace_by_prefix_(state_dict, k, prefix + last_part)
