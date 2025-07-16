#pragma once
#include <cuda.h>
#define NVML_NO_UNVERSIONED_FUNC_DEFS
#include <nvml.h>

#include <c10/util/Exception.h>

#define C10_CUDA_DRIVER_CHECK(EXPR)                                        \
  do {                                                                     \
    CUresult __err = EXPR;                                                 \
    if (__err != CUDA_SUCCESS) {                                           \
      const char* err_str;                                                 \
      CUresult get_error_str_err [[maybe_unused]] =                        \
          c10::cuda::DriverAPI::get()->cuGetErrorString_(__err, &err_str); \
      if (get_error_str_err != CUDA_SUCCESS) {                             \
        TORCH_CHECK(false, "CUDA driver error: unknown error");            \
      } else {                                                             \
        TORCH_CHECK(false, "CUDA driver error: ", err_str);                \
      }                                                                    \
    }                                                                      \
  } while (0)

#define C10_LIBCUDA_DRIVER_API(_)   \
  _(cuDeviceGetAttribute)           \
  _(cuMemAddressReserve)            \
  _(cuMemRelease)                   \
  _(cuMemMap)                       \
  _(cuMemAddressFree)               \
  _(cuMemSetAccess)                 \
  _(cuMemUnmap)                     \
  _(cuMemCreate)                    \
  _(cuMemGetAllocationGranularity)  \
  _(cuMemExportToShareableHandle)   \
  _(cuMemImportFromShareableHandle) \
  _(cuMemsetD32Async)               \
  _(cuStreamWriteValue32)           \
  _(cuGetErrorString)

#if defined(CUDA_VERSION) && (CUDA_VERSION >= 12030)
#define C10_LIBCUDA_DRIVER_API_12030(_) \
  _(cuMulticastAddDevice)               \
  _(cuMulticastBindMem)                 \
  _(cuMulticastCreate)
#else
#define C10_LIBCUDA_DRIVER_API_12030(_)
#endif

#define C10_NVML_DRIVER_API(_)            \
  _(nvmlInit_v2)                          \
  _(nvmlDeviceGetHandleByPciBusId_v2)     \
  _(nvmlDeviceGetNvLinkRemoteDeviceType)  \
  _(nvmlDeviceGetNvLinkRemotePciInfo_v2)  \
  _(nvmlDeviceGetComputeRunningProcesses) \
  _(nvmlSystemGetCudaDriverVersion_v2)

namespace c10::cuda {

struct DriverAPI {
#define CREATE_MEMBER(name) decltype(&name) name##_;
  C10_LIBCUDA_DRIVER_API(CREATE_MEMBER)
  C10_LIBCUDA_DRIVER_API_12030(CREATE_MEMBER)
  C10_NVML_DRIVER_API(CREATE_MEMBER)
#undef CREATE_MEMBER
  static DriverAPI* get();
  static void* get_nvml_handle();
};

} // namespace c10::cuda
