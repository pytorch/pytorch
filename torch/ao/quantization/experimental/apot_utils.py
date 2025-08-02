# mypy: allow-untyped-defs
r"""
This file contains utility functions to convert values
using APoT nonuniform quantization methods.
"""

import torch


r"""Converts floating point input into APoT number
    based on quantization levels
"""

def float_to_apot_tensor(x, levels, indices, quantization_partitions, alpha):
    x.clamp_(min=-alpha, max=alpha)

    grid = levels.to(x.device)
    indices = indices.to(x.device)
    part = quantization_partitions.to(x.device)

    xhard = torch.searchsorted(part, x, right=False)
    xhard = torch.clamp(xhard, min=0, max=grid.shape[0]-1)
    xhard[xhard >= grid.size(0)] = 0
    xhard = indices[xhard]

    return xhard


r"""Converts floating point input into
    reduced precision floating point value
    based on quantization levels
"""

def quant_dequant_tensor(x, levels, indices, quantization_partitions):
    grid = levels.to(x.device)
    indices = indices.to(x.device)
    part = quantization_partitions.to(x.device)

    xhard = torch.searchsorted(part, x, right=False)
    xhard = torch.clamp(xhard, min=0, max=grid.shape[0]-1)
    xhard[xhard >= grid.size(0)] = 0
    xhard = grid[xhard]

    return xhard


r"""Converts APoT input into floating point number
based on quantization levels
"""

def apot_to_float_tensor(x_apot, levels, indices):
    levels = levels.to(x_apot.device)
    indices = indices.to(x_apot.device)

    reverse_lookup_map = torch.empty(indices.max().item() + 1, dtype=indices.dtype).to(x_apot.device)
    reverse_lookup_map[indices] = torch.arange(len(indices), dtype=indices.dtype).to(x_apot.device)
    idx = reverse_lookup_map[x_apot]

    mapped_values = levels[idx]

    return mapped_values
