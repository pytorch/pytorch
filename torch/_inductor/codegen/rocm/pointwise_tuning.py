# mypy: allow-untyped-defs
"""Empirically tuned Triton configs for ROCm pointwise kernels."""

from __future__ import annotations

from typing import List, NamedTuple, Tuple


class RocmPointwiseParams1d(NamedTuple):
    dominant: Tuple[int, int, int]          # (XBLOCK, num_warps, num_stages)
    candidates: List[Tuple[int, int, int]]  # dominant first; full list for autotuning


class RocmPointwiseParams2d(NamedTuple):
    dominant: Tuple[int, int, int, int]          # (XBLOCK, YBLOCK, num_warps, num_stages)
    candidates: List[Tuple[int, int, int, int]]  # dominant first; full list for autotuning


# 1-D configs: (XBLOCK, num_warps, num_stages)
_P1: dict = {
     1: (  256, 1, 1),
     2: (    1, 2, 1),
     3: (  512, 1, 1),
     4: (  128, 2, 1),
     5: ( 1024, 2, 1),
     6: ( 1024, 4, 1),
     7: (   16, 2, 1),
     8: ( 4096, 4, 1),
     9: ( 1024, 8, 1),
    10: (    2, 4, 1),
    11: ( 1024, 1, 1),
    12: (  128, 1, 1),
    13: ( 2048, 4, 1),
    14: (    8, 8, 1),
    15: ( 2048, 2, 1),
    16: (  512, 2, 1),
    17: (    1, 8, 1),
    18: (   16, 8, 1),
    19: (  512, 4, 1),
    20: (    1, 1, 1),
    21: ( 2048, 1, 1),
    22: ( 2048, 8, 1),
    23: ( 4096, 8, 1),
    24: (    2, 1, 1),
    25: (   64, 1, 1),
    26: (  256, 2, 1),
    28: (   32, 4, 1),
    29: (  512, 8, 1),
    32: (    4, 1, 1),
    33: (    4, 2, 1),
    34: (   64, 4, 1),
    37: (    1, 4, 1),
}

# (upper_bound, [config_ids])  — first id is the no-autotune default
_DISPATCH_1D: List[Tuple] = [
    (         128, [ 2,  4,  7, 10, 12, 14, 17, 18, 20, 24, 25, 28, 32, 33]),             # xnumel ≤ 128       (219/241 = 90.9%)
    (         512, [ 1,  2,  3,  4,  7, 10, 12, 14, 16, 17, 18, 19, 20, 24]),  # 128 < x ≤ 512 (172/190 = 90.5%)
    (        1024, [ 1,  2,  3,  4,  5,  6,  7,  9, 10, 11, 12, 14, 16, 18]),  # 512 < x ≤ 1 K (205/225 = 91.1%)
    (        2048, [ 1,  2,  3,  4,  5,  6,  7,  9, 10, 11, 12, 13, 14, 16]),  # 1 K < x ≤ 2 K (133/145 = 91.7%)
    (        4096, [ 1,  2,  3,  4,  5,  6,  7,  8,  9, 11, 12, 13, 14, 15]),                 # 2 K < x ≤ 4 K       (178/191 = 93.2%)
    (        8192, [ 1,  2,  3,  4,  5,  6,  7,  8,  9, 11, 13, 15]),                              # 4 K < x ≤ 8 K       (124/135 = 91.9%)
    (       16384, [ 1,  3,  4,  5,  6,  7,  8,  9, 10, 11, 13, 15]),                              # 8 K < x ≤ 16 K      (175/193 = 90.7%)
    (       32768, [ 1,  3,  4,  5,  6,  7,  8,  9, 11, 12, 13, 15, 16, 18]),                  # 16 K < x ≤ 32 K     (138/149 = 92.6%)
    (       65536, [ 1,  3,  4,  5,  6,  7,  8,  9, 11, 12, 13, 15, 16]),                          # 32 K < x ≤ 64 K     (212/232 = 91.4%)
    (      131072, [ 1,  3,  4,  5,  6,  7,  8,  9, 11, 12, 13, 15, 16]),                          # 64 K < x ≤ 128 K    (248/275 = 90.2%)
    (      262144, [ 1,  3,  4,  5,  6,  8,  9, 11, 12, 13, 15, 16, 19, 21]),                      # 128 K < x ≤ 256 K   (221/242 = 91.3%)
    (      524288, [ 1,  3,  4,  5,  6,  7,  8,  9, 11, 12, 13, 15, 16, 19]),                  # 256 K < x ≤ 512 K   (242/266 = 91.0%)
    (     1048576, [ 1,  3,  4,  5,  6,  7,  8,  9, 11, 12, 13, 15, 16]),                          # 512 K < x ≤ 1 M     (408/448 = 91.1%)
    (     2097152, [ 1,  3,  4,  5,  6,  8,  9, 11, 12, 13, 15, 16, 19, 20]),              # 1 M < x ≤ 2 M       (361/379 = 95.3%)
    (     4194304, [ 1,  3,  4,  5,  6,  8,  9, 11, 12, 13]),                                      # 2 M < x ≤ 4 M       (454/502 = 90.4%)
    (     8388608, [ 1,  3,  4,  5,  6,  7,  8,  9, 11, 12, 13, 15, 16, 19]),                      # 4 M < x ≤ 8 M       (380/420 = 90.5%)
    (    16777216, [ 1,  2,  3,  4,  5,  6,  8,  9, 11, 12, 13, 15, 16, 19]),          # 8 M < x ≤ 16 M      (331/360 = 91.9%)
    (    33554432, [ 1,  3,  4,  5,  6,  8,  9, 11, 12, 13, 15, 16, 17]),                          # 16 M < x ≤ 32 M     (331/362 = 91.4%)
    (    67108864, [ 1,  3,  4,  5,  6,  8,  9, 10, 11, 12, 13, 16]),                              # 32 M < x ≤ 64 M     (300/333 = 90.1%)
    (  1073741824, [ 1,  3,  5,  6,  9, 11, 13, 14, 16, 18, 19]),                                  # 64 M < x ≤ 1 G      (157/168 = 93.5%)
]


def rocm_pointwise_params_1d(xnumel: int) -> RocmPointwiseParams1d:
    """Config params for a 1-D ROCm pointwise kernel, dispatched by xnumel."""
    for upper, indices in _DISPATCH_1D:
        if xnumel <= upper:
            params = [_P1[i] for i in indices]
            return RocmPointwiseParams1d(dominant=params[0], candidates=params)
    # xnumel > 1 G: fall back to the largest-range set
    params = [_P1[i] for i in _DISPATCH_1D[-1][1]]
    return RocmPointwiseParams1d(dominant=params[0], candidates=params)


# 2-D configs: (XBLOCK, YBLOCK, num_warps, num_stages)
_P2: dict = {
     1: (   64,   32, 4, 1),
     2: (    8,   64, 2, 1),
     3: (   64,    2, 2, 1),
     6: (  256,   32, 8, 1),
     7: (   32,   64, 4, 1),
     8: (   16,   32, 4, 1),
     9: (   64,   64, 8, 1),
    10: (    4,  128, 4, 1),
    11: (  128,   16, 4, 1),
    12: ( 1024,    2, 4, 1),
    13: (   64,   16, 4, 1),
    14: (   16,   64, 1, 1),
    15: (   64,  128, 2, 1),
    16: (  256,    8, 2, 1),
    17: (  128,   32, 8, 1),
    18: (   64,   16, 2, 1),
    19: (   32,  256, 4, 1),
    20: (    4,  128, 8, 1),
    21: (   16,   16, 2, 1),
    22: (  256,   32, 4, 1),
    23: (    4,  256, 4, 1),
    24: (   64,   32, 2, 1),
    25: (    8,   32, 1, 1),
    26: (    8, 1024, 2, 1),
    27: (   64,   32, 8, 1),
    28: (  256,    4, 1, 1),
    29: (   64,  256, 1, 1),
    30: ( 1024,   16, 8, 1),
    31: (    1, 2048, 4, 1),
    32: (  256,    4, 2, 1),
    33: (   32,   16, 1, 1),
    34: (   32,   32, 1, 1),
    35: (   64,  256, 8, 1),
    36: (  256,   32, 2, 1),
    37: (    4,  128, 1, 1),
    38: (    8,   16, 1, 1),
    39: (   32,  512, 1, 1),
    40: (   32,   32, 2, 1),
    41: (   32,  128, 8, 1),
    45: (   64,    2, 1, 1),
    46: (   32,   64, 2, 1),
    47: (  128,    8, 1, 1),
    48: (   64,   32, 1, 1),
    49: (    2,    2, 1, 1),
    50: (    8,  128, 1, 1),
    51: (    1,  256, 2, 1),
    52: (   64,   64, 2, 1),
    53: (  256,   64, 8, 1),
    54: (  128,    1, 2, 1),
    55: (    4,   64, 1, 1),
    56: (    2,    2, 4, 1),
    58: (   64,   64, 4, 1),
    73: (    1,    1, 8, 1),
    81: (   16,   32, 1, 1),
}

# (upper_bound, [config_ids])  — first id is the no-autotune default
# Pruned: ranges with >10 configs trimmed to minimum prefix covering ≥90% of ceiling
_DISPATCH_2D: List[Tuple] = [
    (      131072, [ 1,  2,  3,  7,  8,  9, 11, 13, 18, 19, 21, 24, 47, 48, 49, 54]),      # ≤ 128 K    (25/27  = 92.6%)
    (      524288, [ 1,  7,  8,  9, 10, 14, 17, 18, 20, 40]),                                        # 128 K–512 K (22/24 = 91.7%)
    (     1048576, [ 1,  3,  6,  7,  9, 11, 15, 18, 20, 21, 33, 41, 45]),                            # 512 K–1 M  (24/26  = 92.3%)
    (     2097152, [ 1,  3,  6,  7,  8,  9, 11, 13, 15, 16, 17, 18, 19, 21, 23, 27]),   # 1 M–2 M    (45/50  = 90.0%)
    (     4194304, [ 1,  3,  7,  8, 10, 11, 12, 13, 14, 17, 21, 24, 33, 37, 38, 40]),                # 2 M–4 M    (26/28  = 92.9%)
    (     8388608, [ 1,  3,  6,  7,  9, 11, 15, 17, 18, 24, 25, 27, 37, 41, 46, 48]),   # 4 M–8 M    (34/37  = 91.9%)
    (    16777216, [ 1,  6,  7,  8, 13, 20, 24, 25, 26, 27, 28, 29, 34, 36, 39]),                    # 8 M–16 M   (31/33  = 93.9%)
    (    67108864, [ 1,  2,  9, 16, 17, 19, 22, 23, 25, 30, 31, 32, 35, 38, 47, 50]),        # 16 M–64 M  (26/28  = 92.9%)
    (   268435456, [ 1, 25, 81,  2,  9, 16, 17, 19, 22, 23, 30, 31, 32, 35, 38]),                     # 64 M–256 M (padded to 15 w/ best from prev range)
]


def rocm_pointwise_params_2d(xnumel: int, ynumel: int) -> RocmPointwiseParams2d:
    """Config params for a 2-D ROCm pointwise kernel, dispatched by xnumel * ynumel."""
    total = xnumel * ynumel
    for upper, indices in _DISPATCH_2D:
        if total <= upper:
            params = [_P2[i] for i in indices]
            return RocmPointwiseParams2d(dominant=params[0], candidates=params)
    # For total > 256 M: fall back to the largest-range set
    params = [_P2[i] for i in _DISPATCH_2D[-1][1]]
    return RocmPointwiseParams2d(dominant=params[0], candidates=params)
