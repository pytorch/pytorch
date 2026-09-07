#pragma once
#include <cuda.h>
#define NVML_NO_UNVERSIONED_FUNC_DEFS
#include <nvml.h>

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAMacros.h>
#include <c10/util/Exception.h>

#define C10_CUDA_DRIVER_CHECK(EXPR)                                           \
  do {                                                                        \
    c10::cuda::CUDAErrorLogCapture __cuda_error_log;                          \
    CUresult __err = EXPR;                                                    \
    if (__err != CUDA_SUCCESS) {                                              \
      const auto __cuda_error_log_message =                                   \
          __cuda_error_log.get_error_log_suffix();                            \
      const char* err_str;                                                    \
      CUresult get_error_str_err [[maybe_unused]] =                           \
          c10::cuda::DriverAPI::get()->cuGetErrorString_(__err, &err_str);    \
      if (get_error_str_err != CUDA_SUCCESS) {                                \
        TORCH_CHECK(                                                          \
            false,                                                            \
            "CUDA driver error: unknown error",                               \
            __cuda_error_log_message);                                        \
      } else {                                                                \
        TORCH_CHECK(                                                          \
            false, "CUDA driver error: ", err_str, __cuda_error_log_message); \
      }                                                                       \
    }                                                                         \
  } while (0)

// clang-format off
#define C10_CUDA_DRIVER_CHECK_MSG(EXPR, ...)                                \
  do {                                                                      \
    c10::cuda::CUDAErrorLogCapture __cuda_error_log;                        \
    CUresult __err = EXPR;                                                  \
    if (__err != CUDA_SUCCESS) {                                            \
      const auto __cuda_error_log_message =                                 \
          __cuda_error_log.get_error_log_suffix();                          \
      const char* err_str;                                                  \
      CUresult get_error_str_err [[maybe_unused]] =                         \
          c10::cuda::DriverAPI::get()->cuGetErrorString_(__err, &err_str);  \
      if (get_error_str_err != CUDA_SUCCESS) {                              \
        TORCH_CHECK(false, "CUDA driver error: unknown error", __VA_ARGS__, __cuda_error_log_message);\
      } else {                                                              \
        TORCH_CHECK(false, "CUDA driver error: ", err_str, __VA_ARGS__, __cuda_error_log_message);\
      }                                                                     \
    }                                                                       \
  } while (0)
// clang-format on

// Warns instead of throwing, for cleanup paths that must keep unwinding. The
// leading context says which step failed, since a bare driver error string does
// not identify the call it came from.
#define C10_CUDA_DRIVER_CHECK_WARN(EXPR, ...)                              \
  do {                                                                     \
    c10::cuda::CUDAErrorLogCapture __cuda_error_log;                       \
    CUresult __err = EXPR;                                                 \
    if (__err != CUDA_SUCCESS) {                                           \
      const auto __cuda_error_log_message =                                \
          __cuda_error_log.get_error_log_suffix();                         \
      const char* err_str;                                                 \
      CUresult get_error_str_err [[maybe_unused]] =                        \
          c10::cuda::DriverAPI::get()->cuGetErrorString_(__err, &err_str); \
      if (get_error_str_err != CUDA_SUCCESS) {                             \
        TORCH_WARN(                                                        \
            __VA_ARGS__,                                                   \
            ": CUDA driver error: unknown error",                          \
            __cuda_error_log_message);                                     \
      } else {                                                             \
        TORCH_WARN(                                                        \
            __VA_ARGS__,                                                   \
            ": CUDA driver error: ",                                       \
            err_str,                                                       \
            __cuda_error_log_message);                                     \
      }                                                                    \
    }                                                                      \
  } while (0)

#define C10_CUDA_DRIVER_CHECK_GOTO(EXPR, NEXT)                                \
  do {                                                                        \
    c10::cuda::CUDAErrorLogCapture __cuda_error_log;                          \
    CUresult __err = EXPR;                                                    \
    if (__err != CUDA_SUCCESS) {                                              \
      const auto __cuda_error_log_message =                                   \
          __cuda_error_log.get_error_log_suffix();                            \
      const char* err_str;                                                    \
      CUresult get_error_str_err [[maybe_unused]] =                           \
          c10::cuda::DriverAPI::get()->cuGetErrorString_(__err, &err_str);    \
      if (get_error_str_err != CUDA_SUCCESS) {                                \
        TORCH_WARN(                                                           \
            "CUDA driver error: unknown error", __cuda_error_log_message);    \
      } else {                                                                \
        TORCH_WARN("CUDA driver error: ", err_str, __cuda_error_log_message); \
      }                                                                       \
      goto NEXT;                                                              \
    }                                                                         \
  } while (0)

// The integer in the second column specifies the requested CUDA Driver API
// version. The dynamic loader will accept a driver with a newer version, but it
// ensures that the requested symbol exists in *at least* the specified version
// or earlier.

// Keep these requested versions as low as possible to maximize compatibility
// across different driver versions.

// Why do we pin to an older version instead of using the latest?
// If a user installs a newer driver, blindly resolving the symbol may bind to a
// newer version of the function with different behavior, potentially breaking
// PyTorch.

#define C10_LIBCUDA_DRIVER_API_REQUIRED(_)         \
  _(cuDeviceGet, 12000)                            \
  _(cuDeviceGetAttribute, 12000)                   \
  _(cuMemGetAddressRange, 12000)                   \
  _(cuMemAddressReserve, 12000)                    \
  _(cuMemRelease, 12000)                           \
  _(cuMemMap, 12000)                               \
  _(cuMemAddressFree, 12000)                       \
  _(cuMemSetAccess, 12000)                         \
  _(cuMemUnmap, 12000)                             \
  _(cuMemCreate, 12000)                            \
  _(cuMemGetAllocationGranularity, 12000)          \
  _(cuMemExportToShareableHandle, 12000)           \
  _(cuMemImportFromShareableHandle, 12000)         \
  _(cuMemRetainAllocationHandle, 12000)            \
  _(cuMemGetAllocationPropertiesFromHandle, 12000) \
  _(cuMemsetD32Async, 12000)                       \
  _(cuStreamWriteValue32, 12000)                   \
  _(cuGetErrorString, 12000)

#if defined(CUDA_VERSION) && (CUDA_VERSION >= 12030)
#define C10_LIBCUDA_DRIVER_API_12_3(_) \
  _(cuMulticastAddDevice, 12030)       \
  _(cuMulticastBindMem, 12030)         \
  _(cuMulticastCreate, 12030)          \
  _(cuMulticastUnbind, 12030)
#else
#define C10_LIBCUDA_DRIVER_API_12_3(_)
#endif

#if defined(CUDA_VERSION) && (CUDA_VERSION >= 12080)
#define C10_LIBCUDA_DRIVER_API_12_8(_)  \
  _(cuCtxFromGreenCtx, 12080)           \
  _(cuCtxGetCurrent, 12080)             \
  _(cuCtxPopCurrent, 12080)             \
  _(cuCtxPushCurrent, 12080)            \
  _(cuCtxSetCurrent, 12080)             \
  _(cuGreenCtxCreate, 12080)            \
  _(cuGreenCtxDestroy, 12080)           \
  _(cuGreenCtxStreamCreate, 12080)      \
  _(cuDevSmResourceSplitByCount, 12080) \
  _(cuDeviceGetDevResource, 12080)      \
  _(cuDevResourceGenerateDesc, 12080)
#else
#define C10_LIBCUDA_DRIVER_API_12_8(_)
#endif

#if defined(CUDA_VERSION) && (CUDA_VERSION >= 12090)
#define C10_LIBCUDA_DRIVER_API_12_9(_) \
  _(cuLogsRegisterCallback, 12090)     \
  _(cuLogsUnregisterCallback, 12090)
#else
#define C10_LIBCUDA_DRIVER_API_12_9(_)
#endif

#define C10_LIBCUDA_DRIVER_API_OPTIONAL(_) \
  C10_LIBCUDA_DRIVER_API_12_3(_)           \
  C10_LIBCUDA_DRIVER_API_12_8(_)           \
  C10_LIBCUDA_DRIVER_API_12_9(_)

#define C10_NVML_DRIVER_API(_)            \
  _(nvmlInit_v2)                          \
  _(nvmlDeviceGetHandleByPciBusId_v2)     \
  _(nvmlDeviceGetNvLinkRemoteDeviceType)  \
  _(nvmlDeviceGetNvLinkRemotePciInfo_v2)  \
  _(nvmlDeviceGetComputeRunningProcesses) \
  _(nvmlSystemGetCudaDriverVersion_v2)

#if defined(CUDA_VERSION) && (CUDA_VERSION >= 12040)
#define C10_NVML_DRIVER_API_12_4(_) _(nvmlDeviceGetGpuFabricInfoV)
#else
#define C10_NVML_DRIVER_API_12_4(_)
#endif

#define C10_NVML_DRIVER_API_OPTIONAL(_) C10_NVML_DRIVER_API_12_4(_)

namespace c10::cuda {

struct DriverAPI {
#define CREATE_MEMBER_VERSIONED(name, version) decltype(&name) name##_;
#define CREATE_MEMBER(name) decltype(&name) name##_;
  C10_LIBCUDA_DRIVER_API_REQUIRED(CREATE_MEMBER_VERSIONED)
  C10_LIBCUDA_DRIVER_API_OPTIONAL(CREATE_MEMBER_VERSIONED)
  C10_NVML_DRIVER_API(CREATE_MEMBER)
  C10_NVML_DRIVER_API_OPTIONAL(CREATE_MEMBER)
#undef CREATE_MEMBER_VERSIONED
#undef CREATE_MEMBER

  static C10_CUDA_API DriverAPI* get();
  static void* get_nvml_handle();
};

} // namespace c10::cuda
