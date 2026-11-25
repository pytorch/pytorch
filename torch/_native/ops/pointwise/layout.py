from __future__ import annotations

from ...cutedsl import config


r"""Host layout classification for pointwise kernels."""


from functools import lru_cache
from typing import NamedTuple, TYPE_CHECKING

import torch
from torch._tensor_iterator import TensorIterator


if TYPE_CHECKING:
    from ._types import RowGap, Shape


@lru_cache(maxsize=config.POINTWISE_HOST_CACHE_LIMIT)
def output_strides(
    operands: tuple[tuple[torch.dtype, Shape, Shape], ...], dtype: torch.dtype
) -> Shape:
    inputs = [
        torch.empty_strided(shape, strides, dtype=d, device="meta")
        for d, shape, strides in operands
    ]
    iterator = TensorIterator(
        outputs=[None],
        const_inputs=inputs,
        check_all_same_dtype=False,
        static_dtype=dtype,
        static_device=torch.device("meta"),
    )
    return iterator.output().stride()


class IterationLayout(NamedTuple):
    shape: Shape
    strides: tuple[Shape, ...]


def iteration_layout(
    inputs: list[torch.Tensor], outputs: list[torch.Tensor], *, with_index: bool = False
) -> IterationLayout:
    iterator = TensorIterator(
        outputs=[*outputs],
        const_inputs=inputs,
        resize_outputs=False,
        check_all_same_dtype=False,
        check_all_same_device=False,
        check_mem_overlap=False,
        allow_cpu_scalars=True,
        enforce_linear_iteration=with_index,
    )
    return IterationLayout(
        tuple(iterator.shape),
        tuple(iterator.element_strides(i) for i in range(iterator.ntensors)),
    )


def row_gap_view(t: torch.Tensor) -> RowGap | None:
    shape, stride = tuple(t.shape), t.stride()
    if not shape or stride[-1] != 1:
        return None
    run, expect, k = 1, 1, 0
    for i in range(len(shape) - 1, -1, -1):
        if stride[i] == expect:
            run *= shape[i]
            expect *= shape[i]
            k += 1
        else:
            break
    if k == len(shape):
        return None  # Dense inputs use the flat vector path.
    outer_shape, outer_stride = shape[: len(shape) - k], stride[: len(shape) - k]
    for i in range(len(outer_shape) - 1):
        if outer_shape[i + 1] * outer_stride[i + 1] != outer_stride[i]:
            return None
    return t.numel() // run, run, outer_stride[-1]
