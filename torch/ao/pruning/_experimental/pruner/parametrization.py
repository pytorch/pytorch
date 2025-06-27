from __future__ import annotations

from typing import Any, overload, TypeVar


_T_destination = TypeVar("_T_destination", bound=dict[str, Any])

import torch
from torch import nn
from torch.nn.utils.parametrize import is_parametrized


def module_contains_param(module: nn.Module, parametrization: type[nn.Module]) -> bool:
    if is_parametrized(module):
        # see if any of the module tensors have a parametriztion attached that matches the one passed in
        return any(
            any(isinstance(param, parametrization) for param in param_list)
            for key, param_list in getattr(module, "parametrizations", {}).items()
        )
    return False


# Structured Pruning Parameterizations
class FakeStructuredSparsity(nn.Module):
    r"""
    Parametrization for Structured Pruning. Like FakeSparsity, this should be attached to
    the  'weight' or any other parameter that requires a mask.

    Instead of an element-wise bool mask, this parameterization uses a row-wise bool mask.
    """

    def __init__(self, mask: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert isinstance(self.mask, torch.Tensor)
        assert self.mask.shape[0] == x.shape[0]
        shape = [1] * len(x.shape)
        shape[0] = -1
        return self.mask.reshape(shape) * x

    @overload
    def state_dict(
        self, *, destination: _T_destination, prefix: str = "", keep_vars: bool = False
    ) -> _T_destination: ...

    @overload
    def state_dict(
        self, *, prefix: str = "", keep_vars: bool = False
    ) -> dict[str, Any]: ...

    def state_dict(
        self,
        *,
        destination: _T_destination | None = None,
        prefix: str = "",
        keep_vars: bool = False,
    ) -> _T_destination | dict[str, Any]:
        # avoid double saving masks
        return destination if destination is not None else {}


class BiasHook:
    def __init__(
        self, parametrization: FakeStructuredSparsity, prune_bias: bool
    ) -> None:
        self.param = parametrization
        self.prune_bias = prune_bias

    def __call__(
        self, module: nn.Module, input: torch.Tensor, output: torch.Tensor
    ) -> torch.Tensor:
        if hasattr(module, "_bias") and module._bias is not None:
            bias_param = module._bias
            if hasattr(bias_param, "data"):
                bias = bias_param.data
            else:
                assert isinstance(bias_param, torch.Tensor)
                bias = bias_param
            if self.prune_bias and hasattr(self.param, "mask"):
                mask = self.param.mask
                if isinstance(mask, torch.Tensor):
                    assert isinstance(bias, torch.Tensor)
                    bias[~mask] = 0

            # reshape bias to broadcast over output dimensions
            idx = [1] * len(output.shape)
            idx[1] = -1
            assert isinstance(bias, torch.Tensor)
            reshaped_bias = bias.reshape(idx)

            output += reshaped_bias
        return output
