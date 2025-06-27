from typing import Any, overload, TypeVar


T_destination = TypeVar("T_destination", bound=dict[str, Any])

import torch
from torch import nn
from torch.nn.utils.parametrize import is_parametrized


def module_contains_param(module: nn.Module, parametrization: type[nn.Module]) -> bool:
    if is_parametrized(module):
        # see if any of the module tensors have a parametriztion attached that matches the one passed in
        parametrizations = getattr(module, "parametrizations", {})
        return any(
            any(isinstance(param, parametrization) for param in param_list)
            for key, param_list in parametrizations.items()
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
        self, *, destination: T_destination, prefix: str = "", keep_vars: bool = False
    ) -> T_destination: ...

    @overload
    def state_dict(
        self, *, prefix: str = "", keep_vars: bool = False
    ) -> dict[str, Any]: ...

    def state_dict(
        self,
        *,
        destination: T_destination | None = None,
        prefix: str = "",
        keep_vars: bool = False,
    ) -> T_destination | dict[str, Any]:
        # avoid double saving masks
        if destination is not None:
            return destination
        return {}


class BiasHook:
    def __init__(
        self, parametrization: FakeStructuredSparsity, prune_bias: bool
    ) -> None:
        self.param = parametrization
        self.prune_bias = prune_bias

    def __call__(
        self, module: nn.Module, input: torch.Tensor, output: torch.Tensor
    ) -> torch.Tensor:
        if getattr(module, "_bias", None) is not None:
            bias_tensor = module._bias
            assert isinstance(bias_tensor, torch.Tensor)
            bias = bias_tensor.data
            if self.prune_bias:
                mask = self.param.mask
                assert isinstance(mask, torch.Tensor)
                bias = bias.clone()  # Ensure we can modify bias
                bias[~mask] = 0

            # reshape bias to broadcast over output dimensions
            idx = [1] * len(output.shape)
            idx[1] = -1
            bias = bias.reshape(idx)

            output = output + bias  # Avoid in-place operation
        return output
