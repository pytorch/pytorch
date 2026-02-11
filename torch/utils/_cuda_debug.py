from __future__ import annotations

import torch
from torch import Tensor
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_iter


class _TensorTrackingMode(TorchDispatchMode):
    def __init__(self, cuda_graph: torch.cuda.CUDAGraph) -> None:
        self._graph = cuda_graph
        self._capture_mempool_id: tuple[int, int] | None = None

    def _get_capture_mempool_id(self) -> tuple[int, int]:
        if self._capture_mempool_id is None:
            device = torch.cuda.current_device()
            stream = torch.cuda.current_stream(device)
            self._capture_mempool_id = torch._C._cuda_getCaptureMempoolId(
                device, stream.cuda_stream
            )
        return self._capture_mempool_id

    def _is_tensor_from_capture_pool(self, tensor: Tensor) -> bool:
        capture_pool = self._get_capture_mempool_id()
        if capture_pool == (0, 0):
            return False
        device = tensor.device.index
        if device is None:
            return False
        tensor_pool = torch._C._cuda_getBlockMempoolId(device, tensor.data_ptr())
        return tensor_pool == capture_pool

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
                if not self._is_tensor_from_capture_pool(v):
                    self._graph._track_external_input(v)
            elif track_pinned and v.is_pinned():
                self._graph._track_external_input(v)

    def _mark_outputs(self, values: object) -> None:
        for v in tree_iter(values):
            if isinstance(v, Tensor) and v.is_cuda:
                self._graph._mark_internal_output(v)
