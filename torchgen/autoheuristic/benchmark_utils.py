import random

from typing import Any, Tuple

import torch


def transpose_tensors(p_transpose_both: float = 0.05) -> Tuple[bool, bool]:
    transpose_both = random.choices(
        [True, False], [p_transpose_both, 1 - p_transpose_both]
    )[0]
    if transpose_both:
        return (True, True)
    transpose_left = (True, False)
    transpose_right = (False, True)
    no_transpose = (False, False)
    return random.choices([transpose_left, transpose_right, no_transpose])[0]


def fits_in_memory(
    dtype: Any, m: int, k: int, n: int, threshold_memory: int = 21474836480
) -> Any:
    # A100: 81920MiB
    # default: threshold_memory = 81920MiB / 4 = 21474836480
    # dividing by 4 beause we otherwise sometimes run out of memory, I assume because
    # inductor creates copies of tensors for benchmarking?
    return dtype.itemsize * (m * k + k * n + m * n) < threshold_memory


def get_mm_tensors(
    m: int,
    k: int,
    n: int,
    transpose_left: bool,
    transpose_right: bool,
    dtype_left: Any,
    dtype_right: Any,
) -> Tuple[Any, Any]:
    if transpose_left:
        a = torch.randn(k, m, dtype=dtype_left).t()
    else:
        a = torch.randn(m, k, dtype=dtype_left)

    if transpose_right:
        b = torch.randn(n, k, dtype=dtype_right).t()
    else:
        b = torch.randn(k, n, dtype=dtype_right)
    return (a, b)
