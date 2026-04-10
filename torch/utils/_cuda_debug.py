# mypy: allow-untyped-defs
"""Debug tools for CUDA streams and graphs."""

from __future__ import annotations

import ctypes
import threading
import traceback
import warnings

import torch._C
import torch.cuda
from torch.utils._cuda_structures import (
    CudaExternalSemaphoresAsyncParams,
    CudaGraphicsMapParams,
    CudaLaunchKernelParams,
    CudaMemPrefetchAsyncV12020Params,
    CudaMemset2DAsyncParams,
    CudaMemset3DAsyncParams,
    CuExternalSemaphoresAsyncParams,
    CuLaunchKernelParams,
    CuMemPrefetchAsyncV2Params,
    extract_stream_cu_launch_config,
    extract_stream_cuda_launch_config,
    StreamAtSlot0,
    StreamAtSlot1,
    StreamAtSlot2,
    StreamAtSlot3,
    StreamAtSlot4,
    StreamAtSlot5,
    StreamAtSlot7,
)
from torch.utils._python_dispatch import TorchDispatchMode


def _get_cupti():
    try:
        from cupti import cupti  # type: ignore[import]
    except ImportError:
        raise ImportError(
            "cupti-python package is required for warn_on_null_stream_use(). "
            "Install with: pip install cupti-python"
        ) from None
    return cupti


_RUNTIME_STREAM_MAP: dict | None = None
_DRIVER_STREAM_MAP: dict | None = None


def _init_api_maps():
    global _RUNTIME_STREAM_MAP, _DRIVER_STREAM_MAP

    if _RUNTIME_STREAM_MAP is not None:
        return

    _RUNTIME_STREAM_MAP = {}
    _DRIVER_STREAM_MAP = {}

    cupti = _get_cupti()

    runtime_apis = [
        # Kernel launches
        ("cudaLaunchKernel_v7000", CudaLaunchKernelParams),
        ("cudaLaunchKernel_ptsz_v7000", CudaLaunchKernelParams),
        ("cudaLaunchCooperativeKernel_v9000", CudaLaunchKernelParams),
        ("cudaLaunchCooperativeKernel_ptsz_v9000", CudaLaunchKernelParams),
        ("cudaLaunchKernelExC_v11060", extract_stream_cuda_launch_config),
        ("cudaLaunchKernelExC_ptsz_v11060", extract_stream_cuda_launch_config),
        ("cudaLaunchCooperativeKernelMultiDevice_v9000", CudaLaunchKernelParams),
        ("cudaLaunchHostFunc_v10000", StreamAtSlot0),
        ("cudaLaunchHostFunc_ptsz_v10000", StreamAtSlot0),
        # Graph launch/upload
        ("cudaGraphLaunch_v10000", StreamAtSlot1),
        ("cudaGraphLaunch_ptsz_v10000", StreamAtSlot1),
        ("cudaGraphUpload_v10000", StreamAtSlot1),
        ("cudaGraphUpload_ptsz_v10000", StreamAtSlot1),
        # Async memcpy
        ("cudaMemcpyAsync_v3020", StreamAtSlot4),
        ("cudaMemcpyAsync_ptsz_v7000", StreamAtSlot4),
        ("cudaMemcpy2DAsync_v3020", StreamAtSlot5),
        ("cudaMemcpy2DAsync_ptsz_v7000", StreamAtSlot5),
        ("cudaMemcpy3DAsync_v3020", StreamAtSlot1),
        ("cudaMemcpy3DAsync_ptsz_v7000", StreamAtSlot1),
        ("cudaMemcpyFromSymbolAsync_v3020", StreamAtSlot5),
        ("cudaMemcpyFromSymbolAsync_ptsz_v7000", StreamAtSlot5),
        ("cudaMemcpyToSymbolAsync_v3020", StreamAtSlot5),
        ("cudaMemcpyToSymbolAsync_ptsz_v7000", StreamAtSlot5),
        ("cudaMemcpyPeerAsync_v4000", StreamAtSlot5),
        ("cudaMemcpy3DPeerAsync_v4000", StreamAtSlot1),
        ("cudaMemcpy3DPeerAsync_ptsz_v7000", StreamAtSlot1),
        ("cudaMemcpy2DFromArrayAsync_v3020", StreamAtSlot5),
        ("cudaMemcpy2DFromArrayAsync_ptsz_v7000", StreamAtSlot5),
        ("cudaMemcpy2DToArrayAsync_v3020", StreamAtSlot5),
        ("cudaMemcpy2DToArrayAsync_ptsz_v7000", StreamAtSlot5),
        ("cudaMemcpyToArrayAsync_v3020", StreamAtSlot5),
        ("cudaMemcpyToArrayAsync_ptsz_v7000", StreamAtSlot5),
        ("cudaMemcpyFromArrayAsync_v3020", StreamAtSlot5),
        ("cudaMemcpyFromArrayAsync_ptsz_v7000", StreamAtSlot5),
        # Sync memcpy (legacy/NULL stream)
        ("cudaMemcpy_v3020", None),
        ("cudaMemcpy2D_v3020", None),
        ("cudaMemcpy3D_v3020", None),
        ("cudaMemcpy2DFromArray_v3020", None),
        ("cudaMemcpy2DToArray_v3020", None),
        ("cudaMemcpyFromArray_v3020", None),
        ("cudaMemcpyToArray_v3020", None),
        ("cudaMemcpyFromSymbol_v3020", None),
        ("cudaMemcpyToSymbol_v3020", None),
        ("cudaMemcpyPeer_v4000", None),
        ("cudaMemcpy3DPeer_v4000", None),
        ("cudaMemcpyArrayToArray_v3020", None),
        ("cudaMemcpy2DArrayToArray_v3020", None),
        # Batch memcpy (CUDA 12.0+)
        ("cudaMemcpyBatchAsync_v12080", StreamAtSlot7),
        ("cudaMemcpyBatchAsync_ptsz_v12080", StreamAtSlot7),
        ("cudaMemcpyBatchAsync_v13000", StreamAtSlot7),
        ("cudaMemcpyBatchAsync_ptsz_v13000", StreamAtSlot7),
        ("cudaMemcpy3DBatchAsync_v12080", StreamAtSlot3),
        ("cudaMemcpy3DBatchAsync_ptsz_v12080", StreamAtSlot3),
        ("cudaMemcpy3DBatchAsync_v13000", StreamAtSlot3),
        ("cudaMemcpy3DBatchAsync_ptsz_v13000", StreamAtSlot3),
        # Async memset
        ("cudaMemsetAsync_v3020", StreamAtSlot3),
        ("cudaMemsetAsync_ptsz_v7000", StreamAtSlot3),
        ("cudaMemset2DAsync_v3020", CudaMemset2DAsyncParams),
        ("cudaMemset2DAsync_ptsz_v7000", CudaMemset2DAsyncParams),
        ("cudaMemset3DAsync_v3020", CudaMemset3DAsyncParams),
        ("cudaMemset3DAsync_ptsz_v7000", CudaMemset3DAsyncParams),
        # Sync memset (legacy/NULL stream)
        ("cudaMemset_v3020", None),
        ("cudaMemset2D_v3020", None),
        ("cudaMemset3D_v3020", None),
        # Async malloc/free
        ("cudaMallocAsync_v11020", StreamAtSlot2),
        ("cudaMallocAsync_ptsz_v11020", StreamAtSlot2),
        ("cudaFreeAsync_v11020", StreamAtSlot1),
        ("cudaFreeAsync_ptsz_v11020", StreamAtSlot1),
        ("cudaMallocFromPoolAsync_v11020", StreamAtSlot3),
        ("cudaMallocFromPoolAsync_ptsz_v11020", StreamAtSlot3),
        # Prefetch
        ("cudaMemPrefetchAsync_v8000", StreamAtSlot3),
        ("cudaMemPrefetchAsync_ptsz_v8000", StreamAtSlot3),
        ("cudaMemPrefetchAsync_v12020", CudaMemPrefetchAsyncV12020Params),
        ("cudaMemPrefetchAsync_ptsz_v12020", CudaMemPrefetchAsyncV12020Params),
        # Discard/Prefetch batch (CUDA 13.0+)
        ("cudaMemDiscardBatchAsync_v13000", StreamAtSlot4),
        ("cudaMemDiscardBatchAsync_ptsz_v13000", StreamAtSlot4),
        ("cudaMemDiscardAndPrefetchBatchAsync_v13000", StreamAtSlot7),
        ("cudaMemDiscardAndPrefetchBatchAsync_ptsz_v13000", StreamAtSlot7),
        ("cudaMemPrefetchBatchAsync_v13000", StreamAtSlot7),
        ("cudaMemPrefetchBatchAsync_ptsz_v13000", StreamAtSlot7),
        # Events (enqueue a record into the stream)
        ("cudaEventRecord_v3020", StreamAtSlot1),
        ("cudaEventRecord_ptsz_v7000", StreamAtSlot1),
        ("cudaEventRecordWithFlags_v11010", StreamAtSlot1),
        ("cudaEventRecordWithFlags_ptsz_v11010", StreamAtSlot1),
        # Stream ops that enqueue work
        ("cudaStreamWaitEvent_v3020", StreamAtSlot0),
        ("cudaStreamWaitEvent_ptsz_v7000", StreamAtSlot0),
        ("cudaStreamAddCallback_v5000", StreamAtSlot0),
        ("cudaStreamAddCallback_ptsz_v7000", StreamAtSlot0),
        ("cudaStreamAttachMemAsync_v6000", StreamAtSlot0),
        ("cudaStreamAttachMemAsync_ptsz_v7000", StreamAtSlot0),
        # Stream synchronize
        ("cudaStreamSynchronize_v3020", StreamAtSlot0),
        ("cudaStreamSynchronize_ptsz_v7000", StreamAtSlot0),
        # External semaphores
        ("cudaSignalExternalSemaphoresAsync_v10000", CudaExternalSemaphoresAsyncParams),
        (
            "cudaSignalExternalSemaphoresAsync_ptsz_v10000",
            CudaExternalSemaphoresAsyncParams,
        ),
        ("cudaWaitExternalSemaphoresAsync_v10000", CudaExternalSemaphoresAsyncParams),
        (
            "cudaWaitExternalSemaphoresAsync_ptsz_v10000",
            CudaExternalSemaphoresAsyncParams,
        ),
        # Graphics interop
        ("cudaGraphicsMapResources_v3020", CudaGraphicsMapParams),
        ("cudaGraphicsUnmapResources_v3020", CudaGraphicsMapParams),
    ]

    driver_apis = [
        # Kernel launches
        ("cuLaunchKernel", CuLaunchKernelParams),
        ("cuLaunchKernel_ptsz", CuLaunchKernelParams),
        ("cuLaunchCooperativeKernel", CuLaunchKernelParams),
        ("cuLaunchCooperativeKernel_ptsz", CuLaunchKernelParams),
        ("cuLaunchKernelEx", extract_stream_cu_launch_config),
        ("cuLaunchKernelEx_ptsz", extract_stream_cu_launch_config),
        ("cuLaunchHostFunc", StreamAtSlot0),
        ("cuLaunchHostFunc_ptsz", StreamAtSlot0),
        # Graph launch/upload
        ("cuGraphLaunch", StreamAtSlot1),
        ("cuGraphLaunch_ptsz", StreamAtSlot1),
        ("cuGraphUpload", StreamAtSlot1),
        ("cuGraphUpload_ptsz", StreamAtSlot1),
        # Async memcpy
        ("cuMemcpyAsync", StreamAtSlot3),
        ("cuMemcpyAsync_ptsz", StreamAtSlot3),
        ("cuMemcpy2DAsync_v2", StreamAtSlot1),
        ("cuMemcpy2DAsync_v2_ptsz", StreamAtSlot1),
        ("cuMemcpy3DAsync_v2", StreamAtSlot1),
        ("cuMemcpy3DAsync_v2_ptsz", StreamAtSlot1),
        ("cuMemcpyDtoDAsync_v2", StreamAtSlot3),
        ("cuMemcpyDtoDAsync_v2_ptsz", StreamAtSlot3),
        ("cuMemcpyDtoHAsync_v2", StreamAtSlot3),
        ("cuMemcpyDtoHAsync_v2_ptsz", StreamAtSlot3),
        ("cuMemcpyHtoDAsync_v2", StreamAtSlot3),
        ("cuMemcpyHtoDAsync_v2_ptsz", StreamAtSlot3),
        ("cuMemcpyPeerAsync", StreamAtSlot5),
        ("cuMemcpyPeerAsync_ptsz", StreamAtSlot5),
        ("cuMemcpy3DPeerAsync", StreamAtSlot1),
        ("cuMemcpy3DPeerAsync_ptsz", StreamAtSlot1),
        ("cuMemcpyHtoAAsync", StreamAtSlot4),
        ("cuMemcpyHtoAAsync_v2", StreamAtSlot4),
        ("cuMemcpyHtoAAsync_v2_ptsz", StreamAtSlot4),
        ("cuMemcpyAtoHAsync", StreamAtSlot4),
        ("cuMemcpyAtoHAsync_v2", StreamAtSlot4),
        ("cuMemcpyAtoHAsync_v2_ptsz", StreamAtSlot4),
        # Sync memcpy (legacy/NULL stream)
        ("cuMemcpy", None),
        ("cuMemcpy_v2", None),
        ("cuMemcpy2D", None),
        ("cuMemcpy2D_v2", None),
        ("cuMemcpy2DUnaligned", None),
        ("cuMemcpy2DUnaligned_v2", None),
        ("cuMemcpy3D", None),
        ("cuMemcpy3D_v2", None),
        ("cuMemcpy3DPeer", None),
        ("cuMemcpyPeer", None),
        ("cuMemcpyAtoA", None),
        ("cuMemcpyAtoA_v2", None),
        ("cuMemcpyAtoD", None),
        ("cuMemcpyAtoD_v2", None),
        ("cuMemcpyAtoH", None),
        ("cuMemcpyAtoH_v2", None),
        ("cuMemcpyDtoA", None),
        ("cuMemcpyDtoA_v2", None),
        ("cuMemcpyDtoD", None),
        ("cuMemcpyDtoD_v2", None),
        ("cuMemcpyDtoH", None),
        ("cuMemcpyDtoH_v2", None),
        ("cuMemcpyHtoA", None),
        ("cuMemcpyHtoA_v2", None),
        ("cuMemcpyHtoD", None),
        ("cuMemcpyHtoD_v2", None),
        ("cu64Memcpy2D", None),
        ("cu64Memcpy2DUnaligned", None),
        ("cu64Memcpy3D", None),
        ("cu64MemcpyAtoD", None),
        ("cu64MemcpyDtoA", None),
        ("cu64MemcpyDtoD", None),
        ("cu64MemcpyDtoH", None),
        ("cu64MemcpyHtoD", None),
        # Batch memcpy
        ("cuMemcpyBatchAsync", StreamAtSlot3),
        ("cuMemcpyBatchAsync_ptsz", StreamAtSlot3),
        ("cuMemcpyBatchAsync_v2", StreamAtSlot3),
        ("cuMemcpyBatchAsync_v2_ptsz", StreamAtSlot3),
        ("cuMemcpy3DBatchAsync", StreamAtSlot3),
        ("cuMemcpy3DBatchAsync_ptsz", StreamAtSlot3),
        ("cuMemcpy3DBatchAsync_v2", StreamAtSlot3),
        ("cuMemcpy3DBatchAsync_v2_ptsz", StreamAtSlot3),
        # Async memset
        ("cuMemsetD8Async", StreamAtSlot3),
        ("cuMemsetD8Async_ptsz", StreamAtSlot3),
        ("cuMemsetD16Async", StreamAtSlot3),
        ("cuMemsetD16Async_ptsz", StreamAtSlot3),
        ("cuMemsetD32Async", StreamAtSlot3),
        ("cuMemsetD32Async_ptsz", StreamAtSlot3),
        ("cuMemsetD2D8Async", StreamAtSlot5),
        ("cuMemsetD2D8Async_ptsz", StreamAtSlot5),
        ("cuMemsetD2D16Async", StreamAtSlot5),
        ("cuMemsetD2D16Async_ptsz", StreamAtSlot5),
        ("cuMemsetD2D32Async", StreamAtSlot5),
        ("cuMemsetD2D32Async_ptsz", StreamAtSlot5),
        # Sync memset (legacy/NULL stream)
        ("cuMemsetD8", None),
        ("cuMemsetD8_v2", None),
        ("cuMemsetD16", None),
        ("cuMemsetD16_v2", None),
        ("cuMemsetD32", None),
        ("cuMemsetD32_v2", None),
        ("cuMemsetD2D8", None),
        ("cuMemsetD2D8_v2", None),
        ("cuMemsetD2D16", None),
        ("cuMemsetD2D16_v2", None),
        ("cuMemsetD2D32", None),
        ("cuMemsetD2D32_v2", None),
        ("cu64MemsetD8", None),
        ("cu64MemsetD16", None),
        ("cu64MemsetD32", None),
        ("cu64MemsetD2D8", None),
        ("cu64MemsetD2D16", None),
        ("cu64MemsetD2D32", None),
        # Async malloc/free
        ("cuMemAllocAsync", StreamAtSlot2),
        ("cuMemAllocAsync_ptsz", StreamAtSlot2),
        ("cuMemFreeAsync", StreamAtSlot1),
        ("cuMemFreeAsync_ptsz", StreamAtSlot1),
        ("cuMemAllocFromPoolAsync", StreamAtSlot3),
        ("cuMemAllocFromPoolAsync_ptsz", StreamAtSlot3),
        # Prefetch
        ("cuMemPrefetchAsync", StreamAtSlot1),
        ("cuMemPrefetchAsync_ptsz", StreamAtSlot1),
        ("cuMemPrefetchAsync_v2", CuMemPrefetchAsyncV2Params),
        ("cuMemPrefetchAsync_v2_ptsz", CuMemPrefetchAsyncV2Params),
        # Map/decompress/discard/prefetch batch
        ("cuMemMapArrayAsync", StreamAtSlot2),
        ("cuMemMapArrayAsync_ptsz", StreamAtSlot2),
        ("cuMemBatchDecompressAsync", StreamAtSlot4),
        ("cuMemBatchDecompressAsync_ptsz", StreamAtSlot4),
        ("cuMemDiscardBatchAsync", StreamAtSlot4),
        ("cuMemDiscardBatchAsync_ptsz", StreamAtSlot4),
        ("cuMemDiscardAndPrefetchBatchAsync", StreamAtSlot7),
        ("cuMemDiscardAndPrefetchBatchAsync_ptsz", StreamAtSlot7),
        ("cuMemPrefetchBatchAsync", StreamAtSlot7),
        ("cuMemPrefetchBatchAsync_ptsz", StreamAtSlot7),
        # Events (enqueue a record into the stream)
        ("cuEventRecord", StreamAtSlot1),
        ("cuEventRecord_ptsz", StreamAtSlot1),
        ("cuEventRecordWithFlags", StreamAtSlot1),
        ("cuEventRecordWithFlags_ptsz", StreamAtSlot1),
        # Stream ops that enqueue work
        ("cuStreamWaitEvent", StreamAtSlot0),
        ("cuStreamWaitEvent_ptsz", StreamAtSlot0),
        ("cuStreamAddCallback", StreamAtSlot0),
        ("cuStreamAddCallback_ptsz", StreamAtSlot0),
        ("cuStreamAttachMemAsync", StreamAtSlot0),
        ("cuStreamAttachMemAsync_ptsz", StreamAtSlot0),
        # Stream synchronize
        ("cuStreamSynchronize", StreamAtSlot0),
        ("cuStreamSynchronize_ptsz", StreamAtSlot0),
        # Stream batch/value ops
        ("cuStreamBatchMemOp", StreamAtSlot0),
        ("cuStreamBatchMemOp_ptsz", StreamAtSlot0),
        ("cuStreamBatchMemOp_v2", StreamAtSlot0),
        ("cuStreamBatchMemOp_v2_ptsz", StreamAtSlot0),
        ("cuStreamWaitValue32", StreamAtSlot0),
        ("cuStreamWaitValue32_ptsz", StreamAtSlot0),
        ("cuStreamWaitValue32_v2", StreamAtSlot0),
        ("cuStreamWaitValue32_v2_ptsz", StreamAtSlot0),
        ("cuStreamWaitValue64", StreamAtSlot0),
        ("cuStreamWaitValue64_ptsz", StreamAtSlot0),
        ("cuStreamWaitValue64_v2", StreamAtSlot0),
        ("cuStreamWaitValue64_v2_ptsz", StreamAtSlot0),
        ("cuStreamWriteValue32", StreamAtSlot0),
        ("cuStreamWriteValue32_ptsz", StreamAtSlot0),
        ("cuStreamWriteValue32_v2", StreamAtSlot0),
        ("cuStreamWriteValue32_v2_ptsz", StreamAtSlot0),
        ("cuStreamWriteValue64", StreamAtSlot0),
        ("cuStreamWriteValue64_ptsz", StreamAtSlot0),
        ("cuStreamWriteValue64_v2", StreamAtSlot0),
        ("cuStreamWriteValue64_v2_ptsz", StreamAtSlot0),
        # External semaphores
        ("cuSignalExternalSemaphoresAsync", CuExternalSemaphoresAsyncParams),
        ("cuSignalExternalSemaphoresAsync_ptsz", CuExternalSemaphoresAsyncParams),
        ("cuWaitExternalSemaphoresAsync", CuExternalSemaphoresAsyncParams),
        ("cuWaitExternalSemaphoresAsync_ptsz", CuExternalSemaphoresAsyncParams),
    ]

    for api_name, extractor in runtime_apis:
        cbid = getattr(cupti.runtime_api_trace_cbid, api_name, None)
        if cbid is not None:
            _RUNTIME_STREAM_MAP[cbid] = (extractor, api_name)

    for api_name, extractor in driver_apis:
        cbid = getattr(cupti.driver_api_trace_cbid, api_name, None)
        if cbid is not None:
            _DRIVER_STREAM_MAP[cbid] = (extractor, api_name)


_last_warned_correlation_id = -1
_verbosity = 0
_swap_stream_enabled = True


def _is_null_stream_ptr(stream_ptr):
    return stream_ptr == 0 or stream_ptr is None


def _warn_null_stream(func_name, mode):
    msg = f"{mode}::{func_name} launched on NULL stream"
    if _verbosity >= 1:
        msg += f"\nTraceback:\n{''.join(traceback.format_stack()[:-1])}"
    if _verbosity >= 2:
        msg += f"\nC++ Traceback:\n{torch._C._get_cpp_backtrace(0, 64)}"
    warnings.warn(msg, UserWarning, stacklevel=2)


def _check_null_stream(cbdata, extractor, func_name, mode):
    global _last_warned_correlation_id
    if cbdata.correlation_id == _last_warned_correlation_id:
        return
    if extractor is None:
        _last_warned_correlation_id = cbdata.correlation_id
        _warn_null_stream(func_name, mode)
        return
    params_ptr = int(cbdata._data[0]["function_params"])
    if callable(extractor) and not isinstance(extractor, type):
        stream = extractor(params_ptr)
    else:
        p = ctypes.cast(params_ptr, ctypes.POINTER(extractor))
        stream = p[0].stream
    if _is_null_stream_ptr(stream):
        _last_warned_correlation_id = cbdata.correlation_id
        _warn_null_stream(func_name, mode)


def _callback(_, domain, cbid, cbdata):
    cupti = _get_cupti()
    if cbdata.callback_site != cupti.ApiCallbackSite.API_ENTER:
        return
    if domain == cupti.CallbackDomain.RUNTIME_API:
        if cbid in _RUNTIME_STREAM_MAP:  # pyrefly: ignore [not-iterable]
            # pyrefly: ignore [unsupported-operation]
            extractor, func_name = _RUNTIME_STREAM_MAP[cbid]
            _check_null_stream(cbdata, extractor, func_name, "Runtime")
    elif domain == cupti.CallbackDomain.DRIVER_API:
        if cbid in _DRIVER_STREAM_MAP:  # pyrefly: ignore [not-iterable]
            # pyrefly: ignore [unsupported-operation]
            extractor, func_name = _DRIVER_STREAM_MAP[cbid]
            _check_null_stream(cbdata, extractor, func_name, "Driver")


class _NULLStreamUseWarningMode(TorchDispatchMode):
    supports_higher_order_operators = True

    _ref_count: int = 0
    _lock = threading.RLock()
    _subscriber = None
    _orig_stream: object | None = None
    _new_stream: object | None = None

    @classmethod
    def _increment_ref_count(cls):
        with cls._lock:
            if cls._subscriber is None and cls._ref_count == 0:
                _init_api_maps()
                cls._orig_stream = torch.cuda.current_stream()
                if _swap_stream_enabled and _is_null_stream_ptr(
                    cls._orig_stream.cuda_stream
                ):
                    cls._new_stream = torch.cuda.Stream()
                    cls._new_stream.wait_stream(cls._orig_stream)
                    torch.cuda.set_stream(cls._new_stream)
                cls._subscribe_cupti()
            cls._ref_count += 1

    @classmethod
    def _decrement_ref_count(cls):
        with cls._lock:
            cls._ref_count -= 1
            if cls._ref_count <= 0:
                cls._unsubscribe_cupti()
                if cls._new_stream is not None and cls._orig_stream is not None:
                    if torch.cuda.current_stream() == cls._new_stream:
                        # pyrefly: ignore [missing-attribute]
                        cls._orig_stream.wait_stream(cls._new_stream)
                        # pyrefly: ignore [bad-argument-type]
                        torch.cuda.set_stream(cls._orig_stream)

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
        params = cupti.SubscriberParams()
        params.struct_size = cupti.SUBSCRIBER_PARAMS_SIZE
        params.subscriber_name = "torch.cuda.warn_on_null_stream_use"
        try:
            cls._subscriber = cupti.subscribe_v2(_callback, None, params.ptr)
        except cupti.cuptiError as e:
            if "MULTIPLE_SUBSCRIBERS" in str(e):
                old_subscriber_name = None
                try:
                    old_subscriber_name = params.old_subscriber_name
                except Exception:
                    pass
                raise RuntimeError(
                    "CUPTI subscriber already exists "
                    f"(existing subscriber: {old_subscriber_name}). Only one "
                    "CUPTI callback subscriber is allowed at a time. This can "
                    "happen if:\n"
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
        cls._subscriber = None


def warn_on_null_stream_use(verbosity: int = 0, swap_stream: bool = True):
    """Context manager to warn when operations are launched on the NULL CUDA stream.

    The NULL stream (stream 0) in CUDA causes implicit synchronization with all
    other streams on the device, which can significantly degrade performance.
    It can also cause a CUDA graph capture to fail. This tool helps identify code
    paths that accidentally use the NULL stream.

    Args:
        verbosity: Controls how much detail is included in warnings.
            ``0`` — API name only (e.g. ``Runtime::cudaLaunchKernel launched on NULL stream``).
            ``1`` — adds a Python traceback.
            ``2`` — adds both Python and C++ tracebacks.
        swap_stream: When ``True`` (default), automatically swap the default
            (NULL) stream to a new non-default stream on entry.  Set to
            ``False`` to leave the current stream in PyTorch unchanged.

    Example::

        with warn_on_null_stream_use():
            # safe: swap_stream=True (default) moved us off the NULL stream
            x = torch.randn(10, device="cuda")

            # warns: explicitly using the NULL stream
            with torch.cuda.stream(torch.cuda.default_stream()):
                y = torch.randn(10, device="cuda")

    Note:
        This tool requires the ``cupti-python`` package to be installed.
        Only one CUPTI subscriber can be active at a time, so this tool
        cannot be used simultaneously with profilers like nsys or the
        PyTorch profiler.
    """
    global _verbosity, _swap_stream_enabled
    _verbosity = verbosity
    _swap_stream_enabled = swap_stream
    return _NULLStreamUseWarningMode()
