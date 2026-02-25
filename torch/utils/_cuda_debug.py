from __future__ import annotations

import torch
from torch import Tensor
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_iter


class _TensorTrackingMode(TorchDispatchMode):
    def __init__(self, cuda_graph: torch.cuda.CUDAGraph) -> None:
        self._graph = cuda_graph

    # Mirrors the CUDA dispatch key selection rule: if at least one tensor
    # input is a CUDA tensor, the CUDA dispatch key is selected and the op
    # runs (and is captured) on GPU.
    # See aten/src/ATen/core/dispatch/DispatchKeyExtractor.h for details.
    def _selects_cuda_dispatch_key(self, values: object) -> bool:
        for v in tree_iter(values):
            if isinstance(v, Tensor) and v.is_cuda:
                return True
        return False

    def __torch_dispatch__(
        self,
        func: object,
        types: object,
        args: tuple[object, ...] = (),
        kwargs: dict[str, object] | None = None,
    ) -> object:
        kwargs = kwargs or {}
        inputs = [args, kwargs]
        if torch.cuda.is_current_stream_capturing() and self._selects_cuda_dispatch_key(
            inputs
        ):
            out = func(*args, **kwargs)  # type: ignore[operator]
            self._track_inputs(inputs)
            self._mark_outputs(out)
        else:
            out = func(*args, **kwargs)  # type: ignore[operator]
        return out

    def _track_inputs(self, values: object) -> None:
        for v in tree_iter(values):
            if not isinstance(v, Tensor) or v.data_ptr() == 0:
                continue
            if v.is_cuda or v.is_pinned():
                self._graph._track_external_input(v)

    def _mark_outputs(self, values: object) -> None:
        for v in tree_iter(values):
            if isinstance(v, Tensor) and v.is_cuda:
                self._graph._mark_internal_output(v)
