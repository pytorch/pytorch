# mypy: allow-untyped-defs
r"""Utilities to warn about work launched on NULL CUDA streams."""

from __future__ import annotations

import ctypes
import warnings
from typing import Optional

from torch.cuda import current_stream, set_stream, Stream


__all__ = ["warn_on_null_stream_use"]

# Lazy import cupti to avoid import-time warning
_cupti: Optional[object] = None


def _get_cupti():
    global _cupti
    if _cupti is not None:
        return _cupti
    try:
        from cupti import cupti

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

    cupti = _get_cupti()
    if cupti is None:
        _RUNTIME_STREAM_MAP = {}
        _DRIVER_STREAM_MAP = {}
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
        ("cudaMemsetAsync_v3020", 3),
        ("cudaMemsetAsync_ptsz_v7000", 3),
        ("cudaMemset2DAsync_v3020", 4),
        ("cudaMemset2DAsync_ptsz_v7000", 4),
        ("cudaMemset3DAsync_v3020", 1),
        ("cudaMemset3DAsync_ptsz_v7000", 1),
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

    _RUNTIME_STREAM_MAP = {}
    for api_name, stream_idx in runtime_apis:
        try:
            cbid = getattr(cupti.runtime_api_trace_cbid, api_name)
            _RUNTIME_STREAM_MAP[cbid] = stream_idx
        except AttributeError:
            pass

    _DRIVER_STREAM_MAP = {}
    for api_name, stream_idx in driver_apis:
        try:
            cbid = getattr(cupti.driver_api_trace_cbid, api_name)
            _DRIVER_STREAM_MAP[cbid] = stream_idx
        except AttributeError:
            pass


def _is_null_stream_ptr(stream_ptr):
    return stream_ptr == 0 or stream_ptr is None


def _check_null_stream(cbdata, stream_idx):
    params_ptr = int(cbdata._data[0]["function_params"])
    params = ctypes.cast(params_ptr, ctypes.POINTER(ctypes.c_void_p))
    stream = params[stream_idx]
    if _is_null_stream_ptr(stream):
        warnings.warn("Operation launched on NULL stream", UserWarning, stacklevel=4)


def _callback(_, domain, cbid, cbdata):
    cupti = _get_cupti()
    if cbdata.callback_site != cupti.ApiCallbackSite.API_ENTER:
        return
    if domain == cupti.CallbackDomain.RUNTIME_API:
        if cbid in _RUNTIME_STREAM_MAP:
            _check_null_stream(cbdata, _RUNTIME_STREAM_MAP[cbid])
    elif domain == cupti.CallbackDomain.DRIVER_API:
        if cbid in _DRIVER_STREAM_MAP:
            _check_null_stream(cbdata, _DRIVER_STREAM_MAP[cbid])


class _NULLStreamUseWarning:
    def __init__(self):
        self._subscriber = None
        self._orig_stream = None
        self._new_stream = None

    def __enter__(self):
        cupti = _get_cupti()
        if cupti is None:
            return self

        _init_api_maps()
        self._orig_stream = current_stream()
        # If the current stream in PyTorch is a NULL stream, move it away when
        # the tool is enabled. This helps isolate use of the NULL stream which is
        # dangerous since it could cause unexpected stream synchronizations and
        # degrade performance.
        if _is_null_stream_ptr(self._orig_stream.cuda_stream):
            self._new_stream = Stream()
            set_stream(self._new_stream)
        self._enable_tool()
        return self

    def __exit__(self, *args, **kwargs):
        cupti = _get_cupti()
        if cupti is None:
            return False
        if self._new_stream is not None and self._orig_stream is not None:
            # User could have potentially updated the current stream themselves.
            # In that case, we should not override it.
            if current_stream().cuda_stream == self._new_stream.cuda_stream:
                set_stream(self._orig_stream)
        self._disable_tool()
        return False

    def _enable_tool(self):
        cupti = _get_cupti()
        if self._subscriber is not None:
            return
        try:
            self._subscriber = cupti.subscribe(_callback, None)
        except cupti.cuptiError as e:
            if "MULTIPLE_SUBSCRIBERS" in str(e):
                raise RuntimeError(
                    "CUPTI subscriber already exists. Only one CUPTI callback "
                    "subscriber is allowed at a time. This can happen if:\n"
                    "  - Another warn_on_null_stream_use() context is active\n"
                    "  - A profiler (nsys, ncu, nvprof) is attached\n"
                    "  - PyTorch profiler is running\n"
                    "  - Another tool is using CUPTI callbacks"
                ) from e
            raise
        cupti.enable_domain(1, self._subscriber, cupti.CallbackDomain.RUNTIME_API)
        cupti.enable_domain(1, self._subscriber, cupti.CallbackDomain.DRIVER_API)

    def _disable_tool(self):
        cupti = _get_cupti()
        if self._subscriber is None:
            return
        cupti.unsubscribe(self._subscriber)
        self._subscriber = None


def warn_on_null_stream_use():
    """Context manager to warn when operations are launched on the NULL CUDA stream.

    The NULL stream (stream 0) in CUDA causes implicit synchronization with all
    other streams on the device, which can significantly degrade performance.
    This tool helps identify code paths that accidentally use the NULL stream.

    Example::

        with torch.cuda.debug_streams.warn_on_null_stream_use():
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
    return _NULLStreamUseWarning()
