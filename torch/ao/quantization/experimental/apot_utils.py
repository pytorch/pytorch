r"""
This file contains utility functions to convert values
using APoT nonuniform quantization methods.
"""

import math

from typing import Dict
float_to_apot_cache: Dict[float, int] = dict()
apot_to_float_cache: Dict[int, float] = dict()
quant_dequant_cache: Dict[float, float] = dict()

r"""Converts floating point input into APoT number
    based on quantization levels
"""
def float_to_apot(x, levels, indices, alpha):
    if round(x, 4) in float_to_apot_cache:
        return float_to_apot_cache[round(x, 4)]
    else:
        result = 0
        # clip values based on alpha
        if x < -alpha or x > alpha:
            if x < -alpha:
                result = -alpha
            elif x > alpha:
                result = alpha
        else:
            min_delta = math.inf

            for level, idx in zip(levels, indices):
                cur_delta = abs(level - x)
                if cur_delta < min_delta:
                    min_delta = cur_delta
                    result = idx

        float_to_apot_cache[round(x, 4)] = result
        return result

r"""Converts floating point input into
    reduced precision floating point value
    based on quantization levels
"""
def float_to_reduced_precision(x, levels):
    if (round(x, 4) in quant_dequant_cache):
        return quant_dequant_cache[round(x, 4)]
    else:
        min_delta = math.inf
        best_fp = 0.0

        for level in levels:
            cur_delta = abs(level - x)
            if cur_delta < min_delta:
                min_delta = cur_delta
                best_fp = level

        quant_dequant_cache[round(x, 4)] = best_fp
        return best_fp

r"""Converts APoT input into floating point number
based on quantization levels
"""
def apot_to_float(x_apot, levels, indices):
    if x_apot in apot_to_float_cache:
        return apot_to_float_cache[x_apot]
    else:
        idx = indices.index(x_apot)
        apot_to_float_cache[x_apot] = levels[idx]
        return levels[idx]
