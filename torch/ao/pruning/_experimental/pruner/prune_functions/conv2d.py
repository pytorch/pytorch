"""
Collection of pruning functions for patterns that start with Conv2d modules .
"""
from typing import cast, Optional, Callable, Tuple

import torch
from torch import nn, Tensor
from torch.nn.utils import parametrize
from torch.nn.utils.parametrize import ParametrizationList
from .parametrization import FakeStructuredSparsity, BiasHook

# CONV2D
def _prune_conv2d_helper(conv2d: nn.Conv2d) -> Tensor:
    parametrization_dict = cast(nn.ModuleDict, conv2d.parametrizations)
    weight_parameterizations = cast(ParametrizationList, parametrization_dict.weight)
    for p in weight_parameterizations:
        if isinstance(p, FakeStructuredSparsity):
            mask = cast(Tensor, p.mask)

    with torch.no_grad():
        parametrize.remove_parametrizations(conv2d, "weight", leave_parametrized=True)
        conv2d.weight = nn.Parameter(conv2d.weight[mask])
    conv2d.out_channels = conv2d.weight.shape[0]

    _remove_bias_handles(conv2d)
    return mask


def prune_conv2d_padded(conv2d_1: nn.Conv2d) -> None:
    parametrization_dict = cast(nn.ModuleDict, conv2d_1.parametrizations)
    weight_parameterizations = cast(ParametrizationList, parametrization_dict.weight)
    for p in weight_parameterizations:
        if isinstance(p, FakeStructuredSparsity):
            mask = cast(Tensor, p.mask)

    with torch.no_grad():
        parametrize.remove_parametrizations(conv2d_1, "weight", leave_parametrized=True)

    if getattr(conv2d_1, "_bias", None) is not None:
        if (
            conv2d_1.bias is not None
        ):  # conv2d_1 has original bias and bias propagated from previous layer
            new_bias = torch.zeros(conv2d_1.bias.shape)
            new_bias[mask] = conv2d_1.bias[mask]
            # adjusted bias that to keep in conv2d_1
            new_bias[~mask] = cast(Tensor, conv2d_1._bias)[~mask]
            # pruned biases that are kept instead of propagated
            conv2d_1.bias = nn.Parameter(new_bias)
        else:  # conv2d_1 has only original bias
            conv2d_1.bias = nn.Parameter(cast(Tensor, conv2d_1._bias))
    else:
        # no original bias, only propagated bias
        if (
            conv2d_1.bias is not None
        ):  # conv2d_1 has bias propagated from previous layer
            conv2d_1.bias.data[~mask] = 0

    if hasattr(conv2d_1, "_bias"):
        delattr(conv2d_1, "_bias")


def prune_conv2d(conv2d: nn.Conv2d) -> None:
    mask = _prune_conv2d_helper(conv2d)
    if getattr(conv2d, "prune_bias", False):
        _prune_module_bias(conv2d, mask)


def prune_conv2d_conv2d(conv2d_1: nn.Conv2d, conv2d_2: nn.Conv2d) -> None:
    prune_conv2d_activation_conv2d(conv2d_1, None, conv2d_2)


def prune_conv2d_activation_conv2d(
    conv2d_1: nn.Conv2d,
    activation: Optional[Callable[[Tensor], Tensor]],
    conv2d_2: nn.Conv2d,
):
    r"""
    Fusion Pattern for conv2d -> some activation module / function -> conv2d layers
    """
    parametrization_dict = cast(nn.ModuleDict, conv2d_1.parametrizations)
    weight_parameterizations = cast(ParametrizationList, parametrization_dict.weight)
    for p in weight_parameterizations:
        if isinstance(p, FakeStructuredSparsity):
            mask = cast(Tensor, p.mask)

    prune_bias = getattr(conv2d_1, "prune_bias", False)
    if (
        hasattr(conv2d_2, "padding")
        and cast(Tuple[int], conv2d_2.padding) > (0, 0)
        and (conv2d_1.bias is not None or getattr(conv2d_1, "_bias", None) is not None)
    ):
        prune_conv2d_padded(conv2d_1)
    else:
        mask = _prune_conv2d_helper(conv2d_1)
        if prune_bias:
            _prune_module_bias(conv2d_1, mask)
        else:
            pruned_biases = _propogate_module_bias(conv2d_1, mask)
            if pruned_biases is not None:
                if activation:
                    pruned_biases = activation(pruned_biases)
                conv2d_2.bias = _get_adjusted_next_layer_bias(
                    conv2d_2, pruned_biases, mask
                )

        if (
            not (
                hasattr(conv2d_2, "padding")
                and cast(Tuple[int], conv2d_2.padding) > (0, 0)
            )
            or conv2d_1.bias is None
        ):
            with torch.no_grad():
                if parametrize.is_parametrized(conv2d_2):
                    parametrization_dict = cast(
                        nn.ModuleDict, conv2d_2.parametrizations
                    )
                    weight_parameterizations = cast(
                        ParametrizationList, parametrization_dict.weight
                    )
                    weight_parameterizations.original = nn.Parameter(
                        weight_parameterizations.original[:, mask]
                    )
                    conv2d_2.in_channels = weight_parameterizations.original.shape[1]
                else:
                    conv2d_2.weight = nn.Parameter(conv2d_2.weight[:, mask])
                    conv2d_2.in_channels = conv2d_2.weight.shape[1]


def prune_conv2d_pool_activation_conv2d(
    c1: nn.Conv2d,
    pool: nn.Module,
    activation: Optional[Callable[[Tensor], Tensor]],
    c2: nn.Conv2d,
) -> None:
    prune_conv2d_activation_conv2d(c1, activation, c2)


def prune_conv2d_activation_pool_conv2d(
    c1: nn.Conv2d,
    activation: Optional[Callable[[Tensor], Tensor]],
    pool: nn.Module,
    c2: nn.Conv2d,
) -> None:
    prune_conv2d_activation_conv2d(c1, activation, c2)


def prune_conv2d_pool_flatten_linear(
    conv2d: nn.Conv2d,
    pool: nn.Module,
    flatten: Optional[Callable[[Tensor], Tensor]],
    linear: nn.Linear,
) -> None:
    mask = _prune_conv2d_helper(conv2d)

    # We map the pruned indices of the Conv2d output to the flattened indices of the Linear following the Flatten layer.
    # we determine the flattening scale (h * w), and readjust `first_pruned_indices`
    # (each idx maps to range idx * h * w to (idx+1) * h * w), `first_valid_indices`,
    # and `pruned_biases` (repeat each bias by h * w).
    if parametrize.is_parametrized(linear):
        parametrization_dict = cast(nn.ModuleDict, linear.parametrizations)
        weight_parameterizations = cast(
            ParametrizationList, parametrization_dict.weight
        )
        linear_ic = weight_parameterizations.original.shape[1]
    else:
        linear_ic = linear.weight.shape[1]

    conv2d_oc = len(mask)
    assert (
        linear_ic % conv2d_oc == 0
    ), f"Flattening from dimensions {conv2d_oc} to {linear_ic} not supported"

    flatten_scale = linear_ic // conv2d_oc
    flattened_mask = torch.tensor(
        [[val] * flatten_scale for val in mask], dtype=torch.bool, device=mask.device
    ).flatten()

    if getattr(conv2d, "prune_bias", False):
        _prune_module_bias(conv2d, mask)
    else:
        pruned_biases = cast(Tensor, _propogate_module_bias(conv2d, mask))
        flattened_pruned_biases = torch.tensor(
            [[bias] * flatten_scale for bias in pruned_biases], device=mask.device
        ).flatten()
        linear.bias = _get_adjusted_next_layer_bias(
            linear, flattened_pruned_biases, flattened_mask
        )

    with torch.no_grad():
        if parametrize.is_parametrized(linear):
            parametrization_dict = cast(nn.ModuleDict, linear.parametrizations)
            weight_parameterizations = cast(
                ParametrizationList, parametrization_dict.weight
            )
            weight_parameterizations.original = nn.Parameter(
                weight_parameterizations.original[:, flattened_mask]
            )
            linear.in_features = weight_parameterizations.original.shape[1]
        else:
            linear.weight = nn.Parameter(linear.weight[:, flattened_mask])
            linear.in_features = linear.weight.shape[1]
