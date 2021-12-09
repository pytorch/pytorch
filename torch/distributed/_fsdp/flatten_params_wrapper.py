# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) Tongzhou Wang
# Licensed under the MIT License.

from typing import (
    Any,
    Dict,
    Iterator,
    List,
    Optional,
    Tuple,
)

import torch
import torch.nn as nn
from torch import Tensor


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
        self.flat_param = None

        if len(param_list) == 0:
            return

        # A list of parameters to be flatten
        unique_param_list = set(param_list)

        # convert from list of Parameters to set of (Module, parameter_name) tuples, which
        # will survive in case the Parameter instances are reset.
        # it includes (m, n) that points to the same parameter.
        self.param_set = set()
        for m in self.modules():
            for n, p in m.named_parameters(recurse=False):
                if p in unique_param_list:
                    self.param_set.add((m, n))

        params = self._init_flatten_params()
        self.flat_param = nn.Parameter(
            torch.cat(
                [
                    p.detach().reshape(-1)
                    if isinstance(p, nn.Parameter)
                    else p.reshape(-1)
                    for p in params
                ],
                0,
            ),
            requires_grad=params[0].requires_grad,
        )
        self._flatten_params()

    @property
    def module(self) -> Any:
        """Support _fsdp_wrapped_module.module in case we are immitating DDP, which has .module
        property to the underlying module.
        """
        return self._fpw_module

    def _init_flatten_params(self) -> List[nn.Parameter]:
        """Build metadata for need-to-be-flatten parameters and returns a list
        contains the need-to-be-flatten parameters.
        This also fills param_infos and shared_param_infos.
        """
        self._param_infos = []
        shared_param_memo: Dict[nn.Parameter, Tuple[str, nn.Module, str]] = {}
        self._shared_param_infos = []
        params = []
        for module_name, m in self.named_modules():
            for n, p in m.named_parameters(recurse=False):
                if p is not None and (m, n) in self.param_set:
                    if p in shared_param_memo:
                        mname, shared_m, shared_n = shared_param_memo[p]
                        self._shared_param_infos.append(
                            (module_name, mname, m, n, shared_m, shared_n)
                        )
                    else:
                        shared_param_memo[p] = (module_name, m, n)
                        self._param_infos.append((module_name, m, n))
                        params.append(p)
        del shared_param_memo

        assert (
            len(set(p.dtype for p in params)) == 1
        ), "expects all parameters to have same dtype"
        assert (
            len(set(p.requires_grad for p in params)) == 1
        ), "expects all parameters to have same requires_grad"
        assert len(params) == len(set(params)), "params list should not have dups"

        self._param_numels = [p.numel() for p in params]
        self._param_shapes = [p.size() for p in params]
        self._param_names = [
            ".".join([m, n]) if m else n for (m, _, n) in self._param_infos
        ]

        return params

    def _flatten_params(self) -> None:
        """Flatten the managed parameters and replaced the original
        attributes with views to the flat param.
        """
        # register the flatten one
        assert self.flat_param is not None, (
            "Can not flatten params when flat_param is None"
        )
        self.register_parameter("flat_param", self.flat_param)

        # deregister the names as parameters
        for _, m, n in self._param_infos:
            delattr(m, n)
        for _, _, m, n, _, _ in self._shared_param_infos:
            delattr(m, n)

        # register the views as plain attributes
        self._unflatten_params_as_views()

    def _unflatten_params_as_views(self) -> None:
        """Unlike ``_unflatten_params``, this function unflatten into views and keep
        self.flat_param unchanged.
        """
        assert self.flat_param is not None, (
            "Can not unflatten params as views when flat_param is None."
        )
        ps = self._get_param_views()
        for (_, m, n), p in zip(self._param_infos, ps):
            setattr(m, n, p)  # This will set as plain attr

        for (_, _, m, n, shared_m, shared_n) in self._shared_param_infos:
            setattr(m, n, getattr(shared_m, shared_n))

    def _unflatten_params(self) -> None:
        """Undo flattening and create separate parameters from the already flattened
        self.flat_param.
        """
        assert self.flat_param is not None, (
            "Can not unflatten params when flat_param is None."
        )
        ps = self._get_param_views()
        for (_, m, n), p in zip(self._param_infos, ps):
            if hasattr(m, n):
                delattr(m, n)
            m.register_parameter(n, nn.Parameter(p))
        for (_, _, m, n, shared_m, shared_n) in self._shared_param_infos:
            if hasattr(m, n):
                delattr(m, n)
            m.register_parameter(n, getattr(shared_m, shared_n))

        del self.flat_param

    def _get_param_views(
        self, external_data: Optional[Tensor] = None
    ) -> Iterator[Tensor]:
        """Return a generator of views that map to the original parameters."""
        data = external_data if external_data is not None else self.flat_param
        # Data should not be sharded when getting param views
        if data.numel() != sum(self._param_numels):  # type: ignore[union-attr]
            raise ValueError(
                f"Incorrect numel of supplied data: \
                got {data.numel()} but expected {sum(self._param_numels)}"  # type: ignore[union-attr]
            )
        return (
            t.view(s)
            for (t, s) in zip(data.split(self._param_numels), self._param_shapes)  # type: ignore[union-attr]
        )

    def metadata(self) -> Tuple[List[str], List[torch.Size], List[int]]:
        """Return tuple of (names, shapes, numels) metadata for self.flat_param."""
        return self._param_names, self._param_shapes, self._param_numels

    def __getattr__(self, name: str) -> Any:
        """Forward missing attributes to wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.module, name)  # fallback to wrapped module

    def __getitem__(self, key: int) -> Any:
        """Forward indexing calls in case the module is a nn.Sequential."""
        return self.module.__getitem__(key)

    def forward(self, *inputs: Any, **kwinputs: Any) -> Any:
        if self.flat_param is not None:
            self._unflatten_params_as_views()
        return self.module(*inputs, **kwinputs)
