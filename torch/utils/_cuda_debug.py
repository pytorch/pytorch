from __future__ import annotations

import torch
from torch import Tensor
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_iter


class _TensorTrackingMode(TorchDispatchMode):
    def __init__(self, cuda_graph: torch.cuda.CUDAGraph) -> None:
        self._graph = cuda_graph

    # TODO: Do we want to handle pinned host tensors for any other ops?
    _COPY_OPS = frozenset({"aten::copy_", "aten::_to_copy"})

    # This is a proxy for telling if an op launched a CUDA kernel.
    # If it only ever returns host tensors, does it ever launch work
    # on GPU while also being capturable in a CUDA graph (i.e. not
    # causing a stream sync)
    def _has_cuda_tensor_output(self, values: object) -> bool:
        for v in tree_iter(values):
            if isinstance(v, Tensor) and v.is_cuda:
                return True
        return False

    def _is_copy_op(self, func: object) -> bool:
        if hasattr(func, "_schema"):
            return func._schema.name in self._COPY_OPS
        return False

    def __torch_dispatch__(
        self,
        func: object,
        types: object,
        args: tuple[object, ...] = (),
        kwargs: dict[str, object] | None = None,
    ) -> object:
        kwargs = kwargs or {}
        out = func(*args, **kwargs)  # type: ignore[operator]
        if torch.cuda.is_current_stream_capturing() and self._has_cuda_tensor_output(
            out
        ):
            self._track_inputs([args, kwargs], track_pinned=self._is_copy_op(func))
            self._mark_outputs(out)
        return out

    def _track_inputs(self, values: object, *, track_pinned: bool = False) -> None:
        for v in tree_iter(values):
            if not isinstance(v, Tensor) or v.data_ptr() == 0:
                continue
            if v.is_cuda:
                self._graph._track_external_input(v)
            elif track_pinned and v.is_pinned():
                self._graph._track_external_input(v)

    def _mark_outputs(self, values: object) -> None:
        for v in tree_iter(values):
            if isinstance(v, Tensor) and v.is_cuda:
                self._graph._mark_internal_output(v)
