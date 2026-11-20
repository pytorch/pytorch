from __future__ import annotations

from typing import NamedTuple, TYPE_CHECKING

from torch._tensor_iterator import TensorIterator


if TYPE_CHECKING:
    import torch

    from ._types import Shape


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
