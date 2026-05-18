# mypy: allow-untyped-defs
"""CUDA runtime + driver API parameter layouts as ctypes Structures."""

from __future__ import annotations

import ctypes


# ── Primitive types ──────────────────────────────────────────────────────


# struct dim3 { unsigned int x, y, z; }
# https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html
class dim3(ctypes.Structure):
    _fields_ = [("x", ctypes.c_uint), ("y", ctypes.c_uint), ("z", ctypes.c_uint)]


# struct cudaPitchedPtr { void *ptr; size_t pitch, xsize, ysize; }
# https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html
class cudaPitchedPtr(ctypes.Structure):
    _fields_ = [
        ("ptr", ctypes.c_void_p),
        ("pitch", ctypes.c_size_t),
        ("xsize", ctypes.c_size_t),
        ("ysize", ctypes.c_size_t),
    ]


# struct cudaExtent { size_t width, height, depth; }
# https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html
class cudaExtent(ctypes.Structure):
    _fields_ = [
        ("width", ctypes.c_size_t),
        ("height", ctypes.c_size_t),
        ("depth", ctypes.c_size_t),
    ]


# struct cudaMemLocation { enum cudaMemLocationType type; int id; }
# https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html
class cudaMemLocation(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("id", ctypes.c_int)]


# struct CUmemLocation { CUmemLocationType type; int id; }
# https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__UNIFIED.html
class CUmemLocation(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("id", ctypes.c_int)]


# ── Custom structs: APIs with non-pointer-sized fields before stream ─────


# __host__ cudaError_t cudaLaunchKernel(const void* func, dim3 gridDim,  # noqa: B950
#     dim3 blockDim, void** args, size_t sharedMem, cudaStream_t stream)
# https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EXECUTION.html
class CudaLaunchKernelParams(ctypes.Structure):
    _fields_ = [
        ("func", ctypes.c_void_p),
        ("gridDim", dim3),
        ("blockDim", dim3),
        ("args", ctypes.c_void_p),
        ("sharedMem", ctypes.c_size_t),
        ("stream", ctypes.c_void_p),
    ]


# CUresult cuLaunchKernel(CUfunction f, uint gridDimX/Y/Z,  # noqa: B950
#     uint blockDimX/Y/Z, uint sharedMemBytes, CUstream hStream, void** kernelParams, void** extra)
# https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EXEC.html
class CuLaunchKernelParams(ctypes.Structure):
    _fields_ = [
        ("f", ctypes.c_void_p),
        ("gridDimX", ctypes.c_uint),
        ("gridDimY", ctypes.c_uint),
        ("gridDimZ", ctypes.c_uint),
        ("blockDimX", ctypes.c_uint),
        ("blockDimY", ctypes.c_uint),
        ("blockDimZ", ctypes.c_uint),
        ("sharedMemBytes", ctypes.c_uint),
        ("stream", ctypes.c_void_p),
        ("kernelParams", ctypes.c_void_p),
        ("extra", ctypes.c_void_p),
    ]


# struct cudaLaunchConfig_t { dim3 gridDim; dim3 blockDim; size_t dynamicSmemBytes; cudaStream_t stream; ... }
# https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EXECUTION.html
class CudaLaunchConfig(ctypes.Structure):
    _fields_ = [
        ("gridDim", dim3),
        ("blockDim", dim3),
        ("dynamicSmemBytes", ctypes.c_size_t),
        ("stream", ctypes.c_void_p),
        ("attrs", ctypes.c_void_p),
        ("numAttrs", ctypes.c_uint),
    ]


# __host__ cudaError_t cudaLaunchKernelExC(const cudaLaunchConfig_t* config, const void* func, void** args)
# https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EXECUTION.html
class CudaLaunchKernelExCParams(ctypes.Structure):
    _fields_ = [
        ("config", ctypes.c_void_p),
        ("func", ctypes.c_void_p),
        ("args", ctypes.c_void_p),
    ]


# struct CUlaunchConfig { uint gridDimX/Y/Z; uint blockDimX/Y/Z;  # noqa: B950
#     uint sharedMemBytes; CUstream hStream; ... }
# https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EXEC.html
class CuLaunchConfig(ctypes.Structure):
    _fields_ = [
        ("gridDimX", ctypes.c_uint),
        ("gridDimY", ctypes.c_uint),
        ("gridDimZ", ctypes.c_uint),
        ("blockDimX", ctypes.c_uint),
        ("blockDimY", ctypes.c_uint),
        ("blockDimZ", ctypes.c_uint),
        ("sharedMemBytes", ctypes.c_uint),
        ("hStream", ctypes.c_void_p),
        ("attrs", ctypes.c_void_p),
        ("numAttrs", ctypes.c_uint),
    ]


# CUresult cuLaunchKernelEx(const CUlaunchConfig* config, CUfunction f, void** kernelParams, void** extra)
# https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EXEC.html
class CuLaunchKernelExParams(ctypes.Structure):
    _fields_ = [
        ("config", ctypes.c_void_p),
        ("f", ctypes.c_void_p),
        ("kernelParams", ctypes.c_void_p),
        ("extra", ctypes.c_void_p),
    ]


# __host__ cudaError_t cudaMemset2DAsync(void* devPtr, size_t pitch, int value, size_t width, size_t height, cudaStream_t stream)
# https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html
class CudaMemset2DAsyncParams(ctypes.Structure):
    _fields_ = [
        ("devPtr", ctypes.c_void_p),
        ("pitch", ctypes.c_size_t),
        ("value", ctypes.c_int),
        ("_pad0", ctypes.c_int),
        ("width", ctypes.c_size_t),
        ("height", ctypes.c_size_t),
        ("stream", ctypes.c_void_p),
    ]


# __host__ cudaError_t cudaMemset3DAsync(cudaPitchedPtr pitchedDevPtr, int value, cudaExtent extent, cudaStream_t stream)
# https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html
class CudaMemset3DAsyncParams(ctypes.Structure):
    _fields_ = [
        ("pitchedDevPtr", cudaPitchedPtr),
        ("value", ctypes.c_int),
        ("_pad0", ctypes.c_int),
        ("extent", cudaExtent),
        ("stream", ctypes.c_void_p),
    ]


# __host__ cudaError_t cudaMemPrefetchAsync(const void* devPtr,  # noqa: B950
#     size_t count, cudaMemLocation location, unsigned int flags, cudaStream_t stream) [v12020]
# https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html
class CudaMemPrefetchAsyncV12020Params(ctypes.Structure):
    _fields_ = [
        ("devPtr", ctypes.c_void_p),
        ("count", ctypes.c_size_t),
        ("location", cudaMemLocation),
        ("flags", ctypes.c_uint),
        ("_pad0", ctypes.c_uint),
        ("stream", ctypes.c_void_p),
    ]


# CUresult cuMemPrefetchAsync_v2(CUdeviceptr devPtr, size_t count, CUmemLocation location, unsigned int flags, CUstream hStream)
# https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__UNIFIED.html
class CuMemPrefetchAsyncV2Params(ctypes.Structure):
    _fields_ = [
        ("devPtr", ctypes.c_void_p),
        ("count", ctypes.c_size_t),
        ("location", CUmemLocation),
        ("flags", ctypes.c_uint),
        ("_pad0", ctypes.c_uint),
        ("stream", ctypes.c_void_p),
    ]


# __host__ cudaError_t cudaSignalExternalSemaphoresAsync(  # noqa: B950
#     const cudaExternalSemaphore_t* extSemArray,
#     const cudaExternalSemaphoreSignalParams* paramsArray, uint numExtSems, cudaStream_t stream)
# https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EXTRES__INTEROP.html
class CudaExternalSemaphoresAsyncParams(ctypes.Structure):
    _fields_ = [
        ("extSemArray", ctypes.c_void_p),
        ("paramsArray", ctypes.c_void_p),
        ("numExtSems", ctypes.c_uint),
        ("_pad0", ctypes.c_uint),
        ("stream", ctypes.c_void_p),
    ]


# CUresult cuSignalExternalSemaphoresAsync(  # noqa: B950
#     const CUexternalSemaphore* extSemArray,
#     const CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS* paramsArray, uint numExtSems, CUstream stream)
# https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EXTRES__INTEROP.html
class CuExternalSemaphoresAsyncParams(ctypes.Structure):
    _fields_ = [
        ("extSemArray", ctypes.c_void_p),
        ("paramsArray", ctypes.c_void_p),
        ("numExtSems", ctypes.c_uint),
        ("_pad0", ctypes.c_uint),
        ("stream", ctypes.c_void_p),
    ]


# __host__ cudaError_t cudaGraphicsMapResources(int count, cudaGraphicsResource_t* resources, cudaStream_t stream)
# https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__INTEROP.html
class CudaGraphicsMapParams(ctypes.Structure):
    _fields_ = [
        ("count", ctypes.c_int),
        ("_pad0", ctypes.c_int),
        ("resources", ctypes.c_void_p),
        ("stream", ctypes.c_void_p),
    ]


# ── Generic slot structs: all fields are pointer-sized ───────────────────


def _make_stream_at_slot(n):
    fields = [("_" + str(i), ctypes.c_void_p) for i in range(n)]
    fields.append(("stream", ctypes.c_void_p))
    return type(f"_StreamAtSlot{n}", (ctypes.Structure,), {"_fields_": fields})


StreamAtSlot0 = _make_stream_at_slot(0)
StreamAtSlot1 = _make_stream_at_slot(1)
StreamAtSlot2 = _make_stream_at_slot(2)
StreamAtSlot3 = _make_stream_at_slot(3)
StreamAtSlot4 = _make_stream_at_slot(4)
StreamAtSlot5 = _make_stream_at_slot(5)
StreamAtSlot7 = _make_stream_at_slot(7)


# ── Special extractors: stream inside a pointed-to config struct ─────────


def extract_stream_cuda_launch_config(params_ptr):
    p = ctypes.cast(params_ptr, ctypes.POINTER(CudaLaunchKernelExCParams))
    config = ctypes.cast(p[0].config, ctypes.POINTER(CudaLaunchConfig))
    return config[0].stream


def extract_stream_cu_launch_config(params_ptr):
    p = ctypes.cast(params_ptr, ctypes.POINTER(CuLaunchKernelExParams))
    config = ctypes.cast(p[0].config, ctypes.POINTER(CuLaunchConfig))
    return config[0].hStream
