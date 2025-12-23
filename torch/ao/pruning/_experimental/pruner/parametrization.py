# mypy: allow-untyped-defs
import torch
from torch import nn
from torch.nn.utils.parametrize import is_parametrized


def module_contains_param(module, parametrization):
    if is_parametrized(module):
        # see if any of the module tensors have a parametriztion attached that matches the one passed in
        return any(
            any(isinstance(param, parametrization) for param in param_list)
            for key, param_list in module.parametrizations.items()
        )
    return False


# Structured Pruning Parameterizations
class FakeStructuredSparsity(nn.Module):
    r"""
    Parametrization for Structured Pruning. Like FakeSparsity, this should be attached to
    the  'weight' or any other parameter that requires a mask.

    Instead of an element-wise bool mask, this parameterization uses a row-wise bool mask.
    """

    def __init__(self, mask):
        super().__init__()
        self.register_buffer("mask", mask)

    def forward(self, x):
        if not isinstance(self.mask, torch.Tensor):
            raise AssertionError("mask must be a torch.Tensor")
        if self.mask.shape[0] != x.shape[0]:
            raise AssertionError(
                f"mask shape[0] ({self.mask.shape[0]}) must match x shape[0] ({x.shape[0]})"
            )
        shape = [1] * len(x.shape)
        shape[0] = -1
        return self.mask.reshape(shape) * x

    def state_dict(self, *args, **kwargs):
        # avoid double saving masks
        return {}


class BiasHook:
    def __init__(self, parametrization, prune_bias):
        self.param = parametrization
        self.prune_bias = prune_bias

    def __call__(self, module, input, output):
        if getattr(module, "_bias", None) is not None:
            bias = module._bias.data
            if self.prune_bias:
                bias[~self.param.mask] = 0

            # reshape bias to broadcast over output dimensions
            idx = [1] * len(output.shape)
            idx[1] = -1
            bias = bias.reshape(idx)

            output += bias
        return output
