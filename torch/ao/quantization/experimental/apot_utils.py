# mypy: allow-untyped-defs
r"""
This file contains utility functions to convert values
using APoT nonuniform quantization methods.
"""

import math

import torch


r"""Converts floating point input into APoT number
    based on quantization levels
"""


def float_to_apot(x, levels, indices, alpha):
    # clip values based on alpha
    if x < -alpha:
        return -alpha
    elif x > alpha:
        return alpha

    levels_lst = list(levels)
    indices_lst = list(indices)

    min_delta = math.inf
    best_idx = 0

    for level, idx in zip(levels_lst, indices_lst):
        cur_delta = abs(level - x)
        if cur_delta < min_delta:
            min_delta = cur_delta
            best_idx = idx

    return best_idx


def float_to_apot_device(
    tensor: torch.Tensor,
    levels: torch.Tensor,
    indices: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    r"""Converts a floating point tensor into APoT numbers on its device."""
    alpha = alpha.reshape(())
    level_deltas = torch.abs(tensor.unsqueeze(-1) - levels)
    nearest_level_indices = indices[level_deltas.argmin(dim=-1)]
    nearest_level_indices = torch.where(
        torch.isnan(tensor),
        torch.zeros_like(nearest_level_indices),
        nearest_level_indices,
    )
    quantized = torch.where(tensor < -alpha, -alpha, nearest_level_indices)
    return torch.where(tensor > alpha, alpha, quantized)


r"""Converts floating point input into
    reduced precision floating point value
    based on quantization levels
"""


def quant_dequant_util(x, levels, indices):
    min_delta = math.inf
    best_fp = 0.0

    for level in levels:
        cur_delta = abs(level - x)
        if cur_delta < min_delta:
            min_delta = cur_delta
            best_fp = level

    return best_fp


r"""Converts APoT input into floating point number
based on quantization levels
"""


def apot_to_float(x_apot, levels, indices):
    idx = list(indices).index(x_apot)
    return levels[idx]
