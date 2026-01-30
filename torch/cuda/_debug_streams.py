# mypy: allow-untyped-defs
r"""Utilities to warn about work launched on NULL CUDA streams."""

from __future__ import annotations

import ctypes
import threading
import warnings
from typing import Optional


# Lazy import cupti to avoid import-time warning
_cupti: Optional[object] = None

# Lazy init the mode class to avoid importing torch.utils at module import time.
_NULLStreamUseWarningMode = None


def _get_cupti():
    global _cupti
    if _cupti is not None:
        return _cupti
    try:
        from cupti import cupti  # type: ignore[import]

        _cupti = cupti
    except ImportError:
        warnings.warn(
            "cupti-python package not installed. warn_on_null_stream_use() will "
            "have no effect. Install with: pip install cupti-python",
            UserWarning,
        )
    return _cupti


_RUNTIME_STREAM_MAP: Optional[dict] = None
_DRIVER_STREAM_MAP: Optional[dict] = None


def _init_api_maps():
    global _RUNTIME_STREAM_MAP, _DRIVER_STREAM_MAP

    if _RUNTIME_STREAM_MAP is not None:
        return

    _RUNTIME_STREAM_MAP = {}
    _DRIVER_STREAM_MAP = {}

    cupti = _get_cupti()
    if cupti is None:
        return

    runtime_apis = [
        ("cudaLaunchKernel_v7000", 6),
        ("cudaLaunchKernel_ptsz_v7000", 6),
        ("cudaLaunchCooperativeKernel_v9000", 6),
        ("cudaLaunchCooperativeKernel_ptsz_v9000", 6),
        ("cudaLaunchHostFunc_v10000", 1),
        ("cudaLaunchHostFunc_ptsz_v10000", 1),
        ("cudaMemcpyAsync_v3020", 4),
        ("cudaMemcpyAsync_ptsz_v7000", 4),
        ("cudaMemcpy2DAsync_v3020", 5),
        ("cudaMemcpy2DAsync_ptsz_v7000", 5),
        ("cudaMemcpy3DAsync_v3020", 1),
        ("cudaMemcpy3DAsync_ptsz_v7000", 1),
        ("cudaMemcpyFromSymbolAsync_v3020", 5),
        ("cudaMemcpyFromSymbolAsync_ptsz_v7000", 5),
        ("cudaMemcpyToSymbolAsync_v3020", 5),
        ("cudaMemcpyToSymbolAsync_ptsz_v7000", 5),
        ("cudaMemcpyPeerAsync_v4000", 5),
        ("cudaMemcpy3DPeerAsync_v4000", 1),
        # Synchronous memcpy (legacy stream)
        ("cudaMemcpy_v3020", -1),
        ("cudaMemcpy2D_v3020", -1),
        ("cudaMemcpy3D_v3020", -1),
        ("cudaMemcpy2DFromArray_v3020", -1),
        ("cudaMemcpy2DToArray_v3020", -1),
        ("cudaMemcpyFromArray_v3020", -1),
        ("cudaMemcpyToArray_v3020", -1),
        ("cudaMemcpyFromSymbol_v3020", -1),
        ("cudaMemcpyToSymbol_v3020", -1),
        ("cudaMemcpyPeer_v4000", -1),
        ("cudaMemcpy3DPeer_v4000", -1),
        ("cudaMemcpyArrayToArray_v3020", -1),
        ("cudaMemcpy2DArrayToArray_v3020", -1),
        ("cudaMemsetAsync_v3020", 3),
        ("cudaMemsetAsync_ptsz_v7000", 3),
        ("cudaMemset2DAsync_v3020", 4),
        ("cudaMemset2DAsync_ptsz_v7000", 4),
        ("cudaMemset3DAsync_v3020", 1),
        ("cudaMemset3DAsync_ptsz_v7000", 1),
        # Synchronous memset (legacy stream)
        ("cudaMemset_v3020", -1),
        ("cudaMemset2D_v3020", -1),
        ("cudaMemset3D_v3020", -1),
        ("cudaMallocAsync_v11020", 2),
        ("cudaMallocAsync_ptsz_v11020", 2),
        ("cudaFreeAsync_v11020", 1),
        ("cudaFreeAsync_ptsz_v11020", 1),
        ("cudaMemPoolTrimTo_v11020", 1),
        ("cudaMemPoolSetAttribute_v11020", 2),
        ("cudaMemPoolGetAttribute_v11020", 2),
        ("cudaEventRecord_v3020", 1),
        ("cudaEventRecord_ptsz_v7000", 1),
        ("cudaEventRecordWithFlags_v11010", 1),
        ("cudaEventRecordWithFlags_ptsz_v11010", 1),
        ("cudaStreamWaitEvent_v3020", 0),
        ("cudaStreamWaitEvent_ptsz_v7000", 0),
        ("cudaStreamAddCallback_v5000", 0),
        ("cudaStreamAddCallback_ptsz_v7000", 0),
        ("cudaStreamAttachMemAsync_v6000", 0),
        ("cudaStreamAttachMemAsync_ptsz_v7000", 0),
        ("cudaStreamQuery_v3020", 0),
        ("cudaStreamQuery_ptsz_v7000", 0),
        ("cudaStreamSynchronize_v3020", 0),
        ("cudaStreamSynchronize_ptsz_v7000", 0),
        ("cudaStreamGetCaptureInfo_v10010", 0),
        ("cudaStreamGetCaptureInfo_ptsz_v10010", 0),
        ("cudaStreamGetCaptureInfo_v2_v11030", 0),
        ("cudaStreamGetCaptureInfo_v2_ptsz_v11030", 0),
        ("cudaStreamUpdateCaptureDependencies_v11030", 0),
        ("cudaStreamUpdateCaptureDependencies_ptsz_v11030", 0),
        ("cudaGraphicsMapResources_v3020", 2),
        ("cudaGraphicsUnmapResources_v3020", 2),
    ]

    driver_apis = [
        ("cuLaunchKernel", 5),
        ("cuLaunchKernel_ptsz", 5),
        ("cuLaunchCooperativeKernel", 5),
        ("cuLaunchCooperativeKernel_ptsz", 5),
        ("cuLaunchHostFunc", 1),
        ("cuLaunchHostFunc_ptsz", 1),
        ("cuMemcpyAsync", 3),
        ("cuMemcpyAsync_ptsz", 3),
        ("cuMemcpy2DAsync_v2", 1),
        ("cuMemcpy2DAsync_v2_ptsz", 1),
        ("cuMemcpy3DAsync_v2", 1),
        ("cuMemcpy3DAsync_v2_ptsz", 1),
        ("cuMemcpyDtoDAsync_v2", 3),
        ("cuMemcpyDtoDAsync_v2_ptsz", 3),
        ("cuMemcpyDtoHAsync_v2", 3),
        ("cuMemcpyDtoHAsync_v2_ptsz", 3),
        ("cuMemcpyHtoDAsync_v2", 3),
        ("cuMemcpyHtoDAsync_v2_ptsz", 3),
        # Synchronous memcpy (legacy stream)
        ("cuMemcpy", -1),
        ("cuMemcpy_v2", -1),
        ("cuMemcpy2D", -1),
        ("cuMemcpy2D_v2", -1),
        ("cuMemcpy2DUnaligned", -1),
        ("cuMemcpy2DUnaligned_v2", -1),
        ("cuMemcpy3D", -1),
        ("cuMemcpy3D_v2", -1),
        ("cuMemcpy3DPeer", -1),
        ("cuMemcpyPeer", -1),
        ("cuMemcpyAtoA", -1),
        ("cuMemcpyAtoA_v2", -1),
        ("cuMemcpyAtoD", -1),
        ("cuMemcpyAtoD_v2", -1),
        ("cuMemcpyAtoH", -1),
        ("cuMemcpyAtoH_v2", -1),
        ("cuMemcpyDtoA", -1),
        ("cuMemcpyDtoA_v2", -1),
        ("cuMemcpyDtoD", -1),
        ("cuMemcpyDtoD_v2", -1),
        ("cuMemcpyDtoH", -1),
        ("cuMemcpyDtoH_v2", -1),
        ("cuMemcpyHtoA", -1),
        ("cuMemcpyHtoA_v2", -1),
        ("cuMemcpyHtoD", -1),
        ("cuMemcpyHtoD_v2", -1),
        ("cu64Memcpy2D", -1),
        ("cu64Memcpy2DUnaligned", -1),
        ("cu64Memcpy3D", -1),
        ("cu64MemcpyAtoD", -1),
        ("cu64MemcpyDtoA", -1),
        ("cu64MemcpyDtoD", -1),
        ("cu64MemcpyDtoH", -1),
        ("cu64MemcpyHtoD", -1),
        ("cuMemsetD8Async", 3),
        ("cuMemsetD8Async_ptsz", 3),
        ("cuMemsetD16Async", 3),
        ("cuMemsetD16Async_ptsz", 3),
        ("cuMemsetD32Async", 3),
        ("cuMemsetD32Async_ptsz", 3),
        ("cuMemsetD2D8Async", 5),
        ("cuMemsetD2D8Async_ptsz", 5),
        ("cuMemsetD2D16Async", 5),
        ("cuMemsetD2D16Async_ptsz", 5),
        ("cuMemsetD2D32Async", 5),
        ("cuMemsetD2D32Async_ptsz", 5),
        # Synchronous memset (legacy stream)
        ("cuMemsetD8", -1),
        ("cuMemsetD8_v2", -1),
        ("cuMemsetD16", -1),
        ("cuMemsetD16_v2", -1),
        ("cuMemsetD32", -1),
        ("cuMemsetD32_v2", -1),
        ("cuMemsetD2D8", -1),
        ("cuMemsetD2D8_v2", -1),
        ("cuMemsetD2D16", -1),
        ("cuMemsetD2D16_v2", -1),
        ("cuMemsetD2D32", -1),
        ("cuMemsetD2D32_v2", -1),
        ("cu64MemsetD8", -1),
        ("cu64MemsetD16", -1),
        ("cu64MemsetD32", -1),
        ("cu64MemsetD2D8", -1),
        ("cu64MemsetD2D16", -1),
        ("cu64MemsetD2D32", -1),
        ("cuMemAllocAsync", 2),
        ("cuMemAllocAsync_ptsz", 2),
        ("cuMemFreeAsync", 1),
        ("cuMemFreeAsync_ptsz", 1),
        ("cuEventRecord", 1),
        ("cuEventRecord_ptsz", 1),
        ("cuEventRecordWithFlags", 1),
        ("cuEventRecordWithFlags_ptsz", 1),
        ("cuStreamWaitEvent", 0),
        ("cuStreamWaitEvent_ptsz", 0),
    ]

    for api_name, stream_idx in runtime_apis:
        cbid = getattr(cupti.runtime_api_trace_cbid, api_name)
        _RUNTIME_STREAM_MAP[cbid] = (stream_idx, api_name)

    for api_name, stream_idx in driver_apis:
        cbid = getattr(cupti.driver_api_trace_cbid, api_name)
        _DRIVER_STREAM_MAP[cbid] = (stream_idx, api_name)


def _is_null_stream_ptr(stream_ptr):
    # stream_ptr can be None since ctypes returns None for
    # NULL/nullptr values.
    return stream_ptr == 0 or stream_ptr is None


def _check_null_stream(cbdata, stream_idx, func_name, mode):
    if stream_idx < 0:
        warnings.warn(
            f"{mode}::{func_name} launched on NULL stream", UserWarning, stacklevel=4
        )
        return
    params_ptr = int(cbdata._data[0]["function_params"])
    params = ctypes.cast(params_ptr, ctypes.POINTER(ctypes.c_void_p))
    stream = params[stream_idx]
    if _is_null_stream_ptr(stream):
        warnings.warn(
            f"{mode}::{func_name} launched on NULL stream", UserWarning, stacklevel=4
        )


def _callback(_, domain, cbid, cbdata):
    cupti = _get_cupti()
    if cbdata.callback_site != cupti.ApiCallbackSite.API_ENTER:
        return
    if domain == cupti.CallbackDomain.RUNTIME_API:
        # pyrefly: ignore [not-iterable]
        if cbid in _RUNTIME_STREAM_MAP:
            # pyrefly: ignore [unsupported-operation]
            stream_idx, func_name = _RUNTIME_STREAM_MAP[cbid]
            _check_null_stream(cbdata, stream_idx, func_name, "Runtime")
    elif domain == cupti.CallbackDomain.DRIVER_API:
        # pyrefly: ignore [not-iterable]
        if cbid in _DRIVER_STREAM_MAP:
            # pyrefly: ignore [unsupported-operation]
            stream_idx, func_name = _DRIVER_STREAM_MAP[cbid]
            _check_null_stream(cbdata, stream_idx, func_name, "Driver")


def _get_mode_class():
    global _NULLStreamUseWarningMode
    if _NULLStreamUseWarningMode is not None:
        return _NULLStreamUseWarningMode

    from torch.utils._python_dispatch import TorchDispatchMode

    class _NULLStreamUseWarningMode(TorchDispatchMode):
        _ref_count: int = 0
        _lock = threading.RLock()
        _subscriber = None
        _orig_stream: Optional[object] = None
        _new_stream: Optional[object] = None

        @classmethod
        def _increment_ref_count(cls):
            with cls._lock:
                if cls._subscriber is None and cls._ref_count == 0:
                    from torch.cuda import current_stream, set_stream, Stream

                    _init_api_maps()
                    # pyrefly: ignore [bad-assignment, missing-attribute, bad-argument-type]
                    cls._orig_stream = current_stream()
                    if _is_null_stream_ptr(cls._orig_stream.cuda_stream):
                        cls._new_stream = Stream()
                        set_stream(cls._new_stream)
                    cls._subscribe_cupti()
                cls._ref_count += 1

        @classmethod
        def _decrement_ref_count(cls):
            with cls._lock:
                cls._ref_count -= 1
                if cls._ref_count <= 0:
                    from torch.cuda import current_stream, set_stream

                    cls._unsubscribe_cupti()
                    if cls._new_stream is not None and cls._orig_stream is not None:
                        if current_stream() == cls._new_stream:
                            # pyrefly: ignore [bad-argument-type]
                            set_stream(cls._orig_stream)

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            self._increment_ref_count()
            out = func(*args, **(kwargs or {}))
            self._decrement_ref_count()
            return out

        def __enter__(self):
            self._increment_ref_count()
            return super().__enter__()

        def __exit__(self, exc_type, exc_val, exc_tb):
            super().__exit__(exc_type, exc_val, exc_tb)
            self._decrement_ref_count()
            return False

        @classmethod
        def _subscribe_cupti(cls):
            cupti = _get_cupti()
            assert cls._subscriber is None  # noqa: S101
            try:
                cls._subscriber = cupti.subscribe(_callback, None)
            except cupti.cuptiError as e:
                if "MULTIPLE_SUBSCRIBERS" in str(e):
                    raise RuntimeError(
                        "CUPTI subscriber already exists. Only one CUPTI callback "
                        "subscriber is allowed at a time. This can happen if:\n"
                        "  - A profiler (nsys, ncu, nvprof) is attached\n"
                        "  - PyTorch profiler is running\n"
                        "  - Another tool is using CUPTI callbacks"
                    ) from e
                raise
            cupti.enable_domain(1, cls._subscriber, cupti.CallbackDomain.RUNTIME_API)
            cupti.enable_domain(1, cls._subscriber, cupti.CallbackDomain.DRIVER_API)

        @classmethod
        def _unsubscribe_cupti(cls):
            cupti = _get_cupti()
            assert cls._subscriber is not None  # noqa: S101
            cupti.unsubscribe(cls._subscriber)
            cls._subscriber = None  # pyrefly: ignore [bad-assignment]

    _NULLStreamUseWarningMode = _NULLStreamUseWarningMode  # noqa: PLW0127
    return _NULLStreamUseWarningMode


def warn_on_null_stream_use():
    """Context manager to warn when operations are launched on the NULL CUDA stream.

    The NULL stream (stream 0) in CUDA causes implicit synchronization with all
    other streams on the device, which can significantly degrade performance.
    This tool helps identify code paths that accidentally use the NULL stream.

    Example::

        with torch.cuda.warn_on_null_stream_use():
            # This will warn because default_stream() is the NULL stream
            with torch.cuda.stream(torch.cuda.default_stream()):
                x = torch.randn(10, device="cuda")

            # This will not warn
            s = torch.cuda.Stream()
            with torch.cuda.stream(s):
                y = torch.randn(10, device="cuda")

    Note:
        This tool requires the ``cupti-python`` package to be installed.
        Only one CUPTI subscriber can be active at a time, so this tool
        cannot be used simultaneously with profilers like nsys or the
        PyTorch profiler.
    """
    return _get_mode_class()()
