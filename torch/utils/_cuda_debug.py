from __future__ import annotations

import textwrap
import traceback
import weakref
from typing import Any, TYPE_CHECKING

import torch
from torch import Tensor
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_iter


if TYPE_CHECKING:
    from torch.cuda import _POOL_HANDLE


class _TrackedTensorInfo:
    __slots__ = (
        "weakref",
        "creation_traceback_py",
        "creation_traceback_cpp",
        "deletion_traceback_py",
        "data_ptr",
        "device_index",
        "_storage_data_ptr",
    )

    def __init__(self, tensor: Tensor) -> None:
        from torch.utils import get_cpp_backtrace

        self.creation_traceback_py = "".join(traceback.format_stack()[:-3])
        self.creation_traceback_cpp = get_cpp_backtrace()
        self.deletion_traceback_py: str | None = None
        self.data_ptr = tensor.data_ptr()
        self.device_index = tensor.device.index

        def on_release(_: weakref.ref[torch.UntypedStorage]) -> None:
            try:
                self.deletion_traceback_py = "".join(traceback.format_stack()[:-1])
            except Exception:
                pass  # don't raise under GC

        storage = tensor.untyped_storage()
        self._storage_data_ptr = storage.data_ptr()
        self.weakref: weakref.ref[torch.UntypedStorage] = weakref.ref(
            storage, on_release
        )

    def is_alive(self) -> bool:
        storage = self.weakref()
        if storage is None:
            return False
        return storage.data_ptr() == self._storage_data_ptr


class _CUDAGraphInputLivenessTracker:
    def __init__(self) -> None:
        self._external_inputs: dict[int, _TrackedTensorInfo] = {}
        self._internal_outputs: set[int] = set()
        self._memory_snapshots: dict[_POOL_HANDLE, list[dict[str, Any]]] = {}
        self._dispatch_mode: _TensorTrackingMode | None = None

    def start(self) -> None:
        self._dispatch_mode = _TensorTrackingMode(self)
        self._dispatch_mode.__enter__()

    def stop(self) -> None:
        if self._dispatch_mode is not None:
            mode, self._dispatch_mode = self._dispatch_mode, None
            mode.__exit__(None, None, None)

    def track_external_input(self, tensor: Tensor) -> None:
        data_ptr = tensor.data_ptr()
        if (
            data_ptr not in self._internal_outputs
            and data_ptr not in self._external_inputs
        ):
            self._external_inputs[data_ptr] = _TrackedTensorInfo(tensor)

    def mark_internal_output(self, tensor: Tensor) -> None:
        data_ptr = tensor.data_ptr()
        if data_ptr not in self._external_inputs:
            self._internal_outputs.add(data_ptr)

    def check_alive(self, capture_pools: list[_POOL_HANDLE]) -> None:
        dead = [
            i
            for i in self._external_inputs.values()
            if not i.is_alive()
            and not any(
                self._is_tensor_from_capture_pool(i, pool) for pool in capture_pools
            )
        ]
        if not dead:
            return

        def fmt(label: str, tb: str | None) -> str:
            return f"  {label}:\n{textwrap.indent(tb.strip(), '    ')}\n" if tb else ""

        parts = [f"CUDA graph replay detected {len(dead)} dead tensor(s).\n"]
        for i, info in enumerate(dead[:5], 1):
            parts.append(f"Dead tensor #{i} (data_ptr={info.data_ptr:#x}):\n")
            parts.append(fmt("Creation Traceback (Python)", info.creation_traceback_py))
            parts.append(fmt("Creation Traceback (C++)", info.creation_traceback_cpp))
            parts.append(fmt("Deletion Traceback (Python)", info.deletion_traceback_py))
        if len(dead) > 5:
            parts.append(f"  ... and {len(dead) - 5} more\n")
        parts.append(
            fmt("Replay Traceback (Python)", "".join(traceback.format_stack()[:-2]))
        )
        raise RuntimeError("".join(parts))

    def _get_memory_snapshot(self, capture_pool: _POOL_HANDLE) -> list[dict[str, Any]]:
        if capture_pool not in self._memory_snapshots:
            self._memory_snapshots[capture_pool] = torch.cuda.memory.memory_snapshot(
                capture_pool, include_traces=False
            )
        return self._memory_snapshots[capture_pool]

    def _is_tensor_from_capture_pool(
        self, tensor: _TrackedTensorInfo, capture_pool: _POOL_HANDLE
    ) -> bool:
        device = tensor.device_index
        if device is None:
            return False
        tensor_ptr = tensor.data_ptr
        for meminfo in self._get_memory_snapshot(capture_pool):
            if meminfo["device"] != device:
                continue
            for block in meminfo["blocks"]:
                addr = block["address"]
                if (
                    addr <= tensor_ptr < addr + block["size"]
                    and "active" in block["state"]
                ):
                    return True
        return False


class _TensorTrackingMode(TorchDispatchMode):
    supports_higher_order_operators = True

    def __init__(self, tracker: _CUDAGraphInputLivenessTracker) -> None:
        self._tracker = tracker

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
                self._tracker.track_external_input(v)

    def _mark_outputs(self, values: object) -> None:
        for v in tree_iter(values):
            if isinstance(v, Tensor) and v.is_cuda:
                self._tracker.mark_internal_output(v)
