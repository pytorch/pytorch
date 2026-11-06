from __future__ import annotations

import gc
from typing import Protocol, TYPE_CHECKING

import torch
from torch.utils._ordered_set import OrderedSet


if TYPE_CHECKING:
    from collections.abc import Hashable, Sequence


class HelionKernelProtocol(Protocol):
    def __call__(self, *args: torch.Tensor) -> None: ...

    def specialization_key(
        self, args: Sequence[torch.Tensor]
    ) -> tuple[Hashable, ...]: ...


class HelionKernelWrapper:
    def __init__(self, kernel: HelionKernelProtocol) -> None:
        self.kernel = kernel
        self.prepared: OrderedSet[tuple[Hashable, ...]] = OrderedSet()

    def __call__(self, *args: torch.Tensor) -> None:
        # Triton's launch specialization also depends on pointer alignment.
        alignment = tuple((arg.device, arg.data_ptr() % 16) for arg in args)
        if torch.cuda.is_current_stream_capturing():
            key = (*self.kernel.specialization_key(args), alignment)
            if key not in self.prepared:
                raise RuntimeError(
                    "Helion matmul must be warmed up outside CUDA graph capture"
                )
            self.kernel(*args)
        else:
            self.kernel(*args)
            # Compilation can discover new specialization facts. Record the
            # final key only after real execution has initialized the launcher.
            key = (*self.kernel.specialization_key(args), alignment)
            if key not in self.prepared:
                self.prepared.add(key)
                # CuTeDSL's @cute.kernel keeps inspect.currentframe() in a
                # reference cycle, which pins every caller frame of the first
                # compile, including tensors of the running graph. Collect it
                # now: inside a CUDA graph tree warmup those tensors would
                # otherwise outlive their pool accounting.
                gc.collect()
