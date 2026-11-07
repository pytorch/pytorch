from __future__ import annotations

import gc
from typing import Any, Protocol, TYPE_CHECKING

import torch
from torch.utils._ordered_set import OrderedSet


if TYPE_CHECKING:
    from collections.abc import Hashable, Sequence


class HelionKernelProtocol(Protocol):
    settings: Any

    def __call__(self, *args: torch.Tensor) -> None: ...

    def specialization_key(
        self, args: Sequence[torch.Tensor]
    ) -> tuple[Hashable, ...]: ...


class HelionKernelWrapper:
    """Launch an Inductor-generated Helion kernel, also under CUDA graph capture.

    Helion compiles a kernel on the first call with a new specialization, which
    cannot happen while a CUDA graph is being captured. The wrapper records the
    specializations it has run and refuses to capture an unseen one.
    """

    def __init__(self, kernel: HelionKernelProtocol) -> None:
        self.kernel = kernel
        # The kernel has static shapes, so Helion's specialization key is the
        # same for every call; only pointer alignment can change.
        self.shape_key: tuple[Hashable, ...] | None = None
        self.prepared: OrderedSet[tuple[Hashable, ...]] = OrderedSet()

    @staticmethod
    def alignment(args: Sequence[torch.Tensor]) -> tuple[Hashable, ...]:
        # Triton also specializes pointer arguments on 16-byte alignment, which
        # Helion's specialization_key does not include.
        return tuple((arg.device, arg.data_ptr() % 16 == 0) for arg in args)

    def __call__(self, *args: torch.Tensor) -> None:
        if torch.cuda.is_current_stream_capturing():
            if (self.shape_key, self.alignment(args)) not in self.prepared:
                raise RuntimeError(
                    "Helion kernels must be warmed up outside CUDA graph capture"
                )
            self.kernel(*args)
            return
        self.kernel(*args)
        if self.shape_key is None:
            # The first run can add specialization facts, so take the key after it.
            self.shape_key = self.kernel.specialization_key(args)
        key = (self.shape_key, self.alignment(args))
        if key not in self.prepared:
            self.prepared.add(key)
            if self.kernel.settings.backend == "cute":
                # CuTeDSL's @cute.kernel keeps inspect.currentframe() in a
                # reference cycle, which pins every caller frame of the compile,
                # including the tensors of a running CUDA graph tree warmup;
                # those would outlive their pool accounting.
                gc.collect()
