# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) Tongzhou Wang
# Licensed under the MIT License.

import contextlib
from itertools import accumulate
from typing import (
    Any,
    Dict,
    Generator,
    Iterator,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
)

import torch
import torch.nn as nn
from torch import Tensor

from ._utils import _replace_by_prefix


ParamOffset = Tuple[int, int]
SharedParamInfo = Tuple[str, str, nn.Module, str, nn.Module, str]
FLAT_PARAM = "flat_param"
FPW_MODULE = "_fpw_module"


def _post_state_dict_hook(
    module: nn.Module, state_dict: Dict[str, Any], prefix: str, *args: Any
) -> Dict[str, Any]:
    """
    _post_state_dict_hook() is called after the state_dict() is executed
    and before returning the state_dict to the users.
    This API post-processes the keys of the state_dict to remove the
    FlattenParamsWrapper internal prefix.
    """
    # Move everything from FPW_MODULE up one level.
    _replace_by_prefix(state_dict, prefix + f"{FPW_MODULE}.", prefix)
    return state_dict


def _pre_load_state_dict_hook(
    state_dict: Dict[str, Any],
    prefix: str,
    *args: Any,
) -> None:
    """
    _pre_load_state_dict_hook() is called before the _load_from_state_dict() is
    executed. This API pre-processes the keys of the state_dict to add the
    FlattenParamsWrapper internal prefix.
    """
    # Push everything down to FPW_MODULE level.
    _replace_by_prefix(state_dict, prefix, prefix + f"{FPW_MODULE}.")
    # The flat_param_* keys actually needs to move one level up.
    flat_param_key = prefix + f"{FPW_MODULE}.{FLAT_PARAM}"
    for k in list(state_dict.keys()):
        if k.startswith(flat_param_key):
            last_part = k.split(".")[-1]
            assert last_part.startswith(
                FLAT_PARAM
            ), f"Expected key to contain flat_param, but key name is {k}"
            _replace_by_prefix(state_dict, k, prefix + last_part)


class ParamInfo(NamedTuple):
    module_name: str
    module: nn.Module
    param_name: str


class ShardMetadata(NamedTuple):
    param_names: List[str]
    param_shapes: List[torch.Size]
    param_numels: List[int]
    param_offsets: List[ParamOffset]


class FlatParameter(nn.Parameter):
    """
    A parameter that is initialized from a list of parameters. All the
    parameters will be flattened and concatened to form the flat parameter.

    Args:
        params (Sequence[nn.Parameter])
            The parameters to be flattend and concatened.
        requires_grad (bool):
            Set to True if gradients need to be computed for this parameter,
            False otherwise.
    """

    def __new__(
        cls, params: Sequence[nn.Parameter], requires_grad: bool = True
    ) -> "FlatParameter":
        """Make an object using the parent's __new__ function."""

        # A empty or non-list input doesn't make sense.
        if not isinstance(params, (list, tuple)) or len(params) == 0:
            raise ValueError("An non-empty list or tuple argument is needed")

        # Normally, all items are Parameters. But during pickling, we will have a single
        # Tensor as the input and later in __init__, the correct _param_numels and _param_shapes
        # are set.
        if not all(isinstance(p, (nn.Parameter, Tensor)) for p in params):
            incorrect_parameters = [
                p for p in params if not isinstance(p, (nn.Parameter, Tensor))
            ]
            raise ValueError(
                f"List items need to be Parameter types {incorrect_parameters}"
            )

        # Flattening involves (1) making a tensor flat (i.e. single dimensional) and
        # (2) making a module hierarchy flat (using a single tensor to replace a tree of
        # tensors). Therefore, adding back nesting and hierarchy is counter-productive.
        # If nesting is encountered in the future, the reasonable thing to do is likely
        # for the top level FlatParameter to absorb the nested one and keep the result flat,
        # free from hierarchy.
        if any(isinstance(p, FlatParameter) for p in params):
            raise ValueError("Nesting FlatParameter is not supported")

        data = torch.cat(
            [
                p.detach().reshape(-1) if isinstance(p, nn.Parameter) else p.reshape(-1)
                for p in params
            ],
            0,
        )

        return super(FlatParameter, cls).__new__(
            cls, data, requires_grad=requires_grad
        )  # type: ignore[call-arg]

    def __init__(self, params: Sequence[nn.Parameter], requires_grad: bool = True):
        self._is_sharded = False
        self._param_numels = [p.numel() for p in params]
        # The total element numbers. This is equal to the summation of the
        # ``numel()`` of all the parameters.
        self.full_numel = sum(self._param_numels)
        assert self.numel() <= self.full_numel, (
            "Parameter numbers mismatched. "
            f"The number of elements in FlatParameter: {self.numel()} vs. "
            f"the number of elements in original parameters: {self.full_numel}."
        )
        # The shapes of each individual parameter.
        self._param_shapes = [p.size() for p in params]
        cumulative_sum = list(accumulate(self._param_numels))
        begin = [0] + cumulative_sum[:-1]
        end = [e - 1 for e in cumulative_sum]

        self._param_infos: List[ParamInfo] = []
        self._shared_param_infos: List[SharedParamInfo] = []

        # The element offsets (begin/end pair) in the flat parameter of each
        # individual parameter.
        self._param_offsets = list(zip(begin, end))
        # The indices (begin/end pair) of the parameters that are included in
        # this FlatParameter. The default value is all the parameters because
        # no sharding happen yet.
        self._param_indice_in_shard = (0, len(self._param_infos) - 1)
        # The offsets in each parameter that is included in the FlatParameter.
        self._sharded_param_offsets: List[ParamOffset] = [
            (0, numel) for numel in self._param_numels
        ]
        # The number of padding elements.
        self.num_padded = 0

    def shard_by_offsets(self, start: int, end: int, num_padded: int) -> None:
        assert self._is_sharded
        if start < 0 or end < 0 or end < start:
            raise ValueError(
                f"Shard the flatten parameter with an invalid offset pair {(start, end)}."
            )
        _shard_size = end - start + 1
        self.num_padded = num_padded
        if self.num_padded > _shard_size:
            raise ValueError("The number of padding is larger than the shard size.")
        self._sharded_param_offsets.clear()

        ranges = []
        for idx, offset in enumerate(self._param_offsets):
            if start > offset[1] or end < offset[0]:
                continue
            if start <= offset[0]:
                sharded_param_start = 0
                sharded_param_end = min(offset[1], end) - offset[0]
            else:
                sharded_param_start = start - offset[0]
                sharded_param_end = min(offset[1], end) - offset[0]
            ranges.append(idx)
            self._sharded_param_offsets.append((sharded_param_start, sharded_param_end))
        if ranges:
            self._param_indice_in_shard = (ranges[0], ranges[-1])

    def _offset_to_slice(self) -> slice:
        if self._param_indice_in_shard[0] > self._param_indice_in_shard[1]:
            return slice(0, 0)
        return slice(self._param_indice_in_shard[0], self._param_indice_in_shard[1] + 1)

    def get_param_views(
        self, external_data: Optional[Tensor] = None
    ) -> Iterator[Tensor]:
        """Return a generator of views that map to the original parameters."""
        # Note, self.data could be sharded, so its numel is <= to the sum.
        assert (
            self.data.numel() <= self.full_numel
        ), f"Incorrect internal state {self.data.numel()} vs. {self.full_numel}"
        data = external_data if external_data is not None else self
        if data.numel() != self.full_numel:
            raise ValueError(
                f"Incorrect numel of supplied data: got {data.numel()} but expected {self.full_numel}"
            )
        return (
            t.view(s)
            for (t, s) in zip(data.split(self._param_numels), self._param_shapes)
        )

    @property
    def _num_unflattened_params(self) -> int:
        """Returns the number of unflattened parameters that comprise this
        flattened parameter."""
        assert hasattr(self, "_param_infos"), \
            "`_param_infos` has not been set, meaning this `FlatParameter` " \
            "has not been initialized yet"
        num_unflat_params = len(self._param_infos)
        assert num_unflat_params > 0, "`FlatParameter` corresponding to 0 " \
            "unflattened parameters"
        return num_unflat_params

    @property
    def param_info(self) -> List[ParamInfo]:
        return self._param_infos

    @property
    def _param_names(self):
        return [".".join([m, n]) if m else n for (m, _, n) in self._param_infos]

    def metadata(self) -> Tuple[List[str], List[torch.Size], List[int]]:
        """Return tuple of (names, shapes, numels) metadata for this flat parameter."""
        return self._param_names, self._param_shapes, self._param_numels

    def shard_metadata(
        self,
    ) -> ShardMetadata:
        """
        Return tuple of (names, shapes, numels) metadata for the sharded parameter
        metadata of this flat parameter.
        """
        return ShardMetadata(
            self._param_names[self._offset_to_slice()],
            self._param_shapes[self._offset_to_slice()],
            self._param_numels[self._offset_to_slice()],
            self._sharded_param_offsets[:],
        )


class FlattenParamsWrapper(nn.Module):
    """
    A wrapper for transparently flattening a Module's parameters.
    The original implementation [1] reparameterizes a PyTorch module
    that is called ReparamModule. The ReparamModule has only a flattened
    parameter representing all parameters of the wrapped module.
    Compared to the original implementation [1], this version:
    - removes tracing
    - supports shared parameters
    - is renamed to FlattenParamsWrapper
    [1] https://github.com/SsnL/PyTorch-Reparam-Module
    Args:
        module (nn.Module):
            The module to wrap.
        param_list (List[nn.Parameter]):
            Only flatten parameters appearing in the given list.
            Note, if only a single param is in the list, it still gets
            flattened and the original param is removed and replaced
            with the flatten one.
    """

    def __init__(self, module: nn.Module, param_list: List[nn.Parameter]):
        super().__init__()
        self._fpw_module = module
        # People may test whether this module contains parameters by using
        # `getattr(module, "flat_param") is None`. This is not always accurate
        # as the above condition is also true if this module is unflattened.
        # `no_params` explicitly shows this module has no parameters and
        # is always correct regardless flattened or unflattened.
        self.no_params = True
        self.flat_param = None

        # Register hook to be called after state_dict() to remove the
        # "_fpw_module." prefix and before load_state_dict() to add it back.
        # The hooks must be registered even if the target param_list is empty as
        # all submodules in FlattenParamsWrapper should be pre/post processed by
        # the hooks.
        self._register_state_dict_hook(_post_state_dict_hook)
        self._register_load_state_dict_pre_hook(_pre_load_state_dict_hook)

        if len(param_list) == 0:
            return

        # A list of parameters to be flatten
        unique_param_list = set(param_list)
        self.no_params = False

        # convert from list of Parameters to set of (Module, parameter_name) tuples, which
        # will survive in case the Parameter instances are reset.
        # it includes (m, n) that points to the same parameter.
        self.param_set = set()
        for m in self.modules():
            for n, p in m.named_parameters(recurse=False):
                if p in unique_param_list:
                    self.param_set.add((m, n))

        params, param_infos, shared_param_infos = self._init_flatten_params()
        self.flat_param = FlatParameter(params, params[0].requires_grad)
        self.flat_param._param_infos = param_infos
        self.flat_param._shared_param_infos = shared_param_infos

        # This attribute is used to remember the flat_param inside the unflatten_params()
        # context. With this attribute, FSDP can access the flat parameter metadata
        # even if flat_param is temporarily deleted.
        # ``orig_flat_param` is a list to avoid being tracked by ``state_dict()``.
        self.orig_flat_param: List[Optional[FlatParameter]] = [None]
        self._flatten_params()

        # Sanity check for the string constants.
        assert getattr(self, FPW_MODULE) is self._fpw_module
        assert getattr(self, FLAT_PARAM) is self.flat_param

    @property
    def module(self) -> Any:
        """Support _fsdp_wrapped_module.module in case we are immitating DDP, which has .module
        property to the underlying module.
        """
        return self._fpw_module

    def _init_flatten_params(
        self,
    ) -> Tuple[List[nn.Parameter], List[ParamInfo], List[SharedParamInfo]]:
        """Build metadata for need-to-be-flatten parameters and returns a list
        contains the need-to-be-flatten parameters.
        This also fills param_infos and shared_param_infos.
        """
        param_infos: List[ParamInfo] = []
        shared_param_infos = []
        shared_param_memo: Dict[nn.Parameter, Tuple[str, nn.Module, str]] = {}
        params = []
        for module_name, m in self.named_modules():
            for n, p in m.named_parameters(recurse=False):
                if p is not None and (m, n) in self.param_set:
                    if p in shared_param_memo:
                        mname, shared_m, shared_n = shared_param_memo[p]
                        shared_param_infos.append(
                            (module_name, mname, m, n, shared_m, shared_n)
                        )
                    else:
                        shared_param_memo[p] = (module_name, m, n)
                        param_infos.append(ParamInfo(module_name, m, n))
                        params.append(p)
        del shared_param_memo

        assert (
            len(set(p.dtype for p in params)) == 1
        ), "expects all parameters to have same dtype"
        assert (
            len(set(p.requires_grad for p in params)) == 1
        ), "expects all parameters to have same requires_grad"
        assert len(params) == len(set(params)), "params list should not have dups"

        return params, param_infos, shared_param_infos

    def _flatten_params(self, external_data: Optional[FlatParameter] = None) -> None:
        """Flatten the managed parameters and replaced the original
        attributes with views to the flat param. If `external_data`
        is passed, it will be used as the flat_param.
        """
        # register the flatten one
        assert (
            getattr(self, "flat_param", None) is not None or external_data is not None
        ), "Can not flatten params when both flat_param and external_data are None."
        if external_data is not None:
            self.flat_param = external_data
        self.register_parameter("flat_param", self.flat_param)

        assert self.flat_param is not None  # avoid mypy complain.
        # deregister the names as parameters
        for _, m, n in self.flat_param._param_infos:
            delattr(m, n)
        for _, _, m, n, _, _ in self.flat_param._shared_param_infos:
            delattr(m, n)

        # register the views as plain attributes
        self._unflatten_params_as_views()

    def _unflatten_params_as_views(self) -> None:
        """Unlike ``_unflatten_params``, this function unflatten into views and keep
        self.flat_param unchanged.
        """
        assert (
            self.flat_param is not None
        ), "Can not unflatten params as views when flat_param is None."
        ps = self._get_param_views()
        for (_, m, n), p in zip(self.flat_param._param_infos, ps):
            setattr(m, n, p)  # This will set as plain attr

        for (_, _, m, n, shared_m, shared_n) in self.flat_param._shared_param_infos:
            setattr(m, n, getattr(shared_m, shared_n))

    def _unflatten_params(self) -> None:
        """Undo flattening and create separate parameters from the already flattened
        self.flat_param.
        """
        assert (
            self.flat_param is not None
        ), "Can not unflatten params when flat_param is None."
        ps = self._get_param_views()
        for (_, m, n), p in zip(self.flat_param._param_infos, ps):
            if hasattr(m, n):
                delattr(m, n)
            m.register_parameter(n, nn.Parameter(p))
        for (_, _, m, n, shared_m, shared_n) in self.flat_param._shared_param_infos:
            if hasattr(m, n):
                delattr(m, n)
            m.register_parameter(n, getattr(shared_m, shared_n))

        del self.flat_param

    @contextlib.contextmanager
    def unflatten_params(self) -> Generator:
        """
        Unflatten params. If the current instance is already unflattened, then
        it will remain unflattened after the context manager exits.
        """
        if getattr(self, "flat_param", None) is None:
            yield
        else:
            self.orig_flat_param[0] = self.flat_param
            self._unflatten_params()
            # Put yield in a try...finally in case the caller catches the exception and handles
            # it. In that case, we need to properly handle the undoing of state here.
            try:
                yield
            finally:
                self._flatten_params(self.orig_flat_param[0])
                self.orig_flat_param[0] = None

    def _get_param_views(
        self, external_data: Optional[Tensor] = None
    ) -> Iterator[Tensor]:
        """Return a generator of views that map to the original parameters."""
        assert self.flat_param is not None
        return self.flat_param.get_param_views(external_data)

    def __getattr__(self, name: str) -> Any:
        """Forward missing attributes to wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.module, name)  # fallback to wrapped module

    def __getitem__(self, key: int) -> Any:
        """Forward indexing calls in case the module is a nn.Sequential."""
        return self.module.__getitem__(key)

    def _unflatten_params_if_needed(self) -> None:
        if self.flat_param is not None:
            self._unflatten_params_as_views()

    def forward(self, *inputs: Any, **kwinputs: Any) -> Any:
        self._unflatten_params_if_needed()
        return self.module(*inputs, **kwinputs)
