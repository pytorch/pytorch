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

#define C10_LIBCUDA_DRIVER_API(_)          \
  _(cuDeviceGetAttribute, 12000)           \
  _(cuMemAddressReserve, 12000)            \
  _(cuMemRelease, 12000)                   \
  _(cuMemMap, 12000)                       \
  _(cuMemAddressFree, 12000)               \
  _(cuMemSetAccess, 12000)                 \
  _(cuMemUnmap, 12000)                     \
  _(cuMemCreate, 12000)                    \
  _(cuMemGetAllocationGranularity, 12000)  \
  _(cuMemExportToShareableHandle, 12000)   \
  _(cuMemImportFromShareableHandle, 12000) \
  _(cuMemsetD32Async, 12000)               \
  _(cuStreamWriteValue32, 12000)           \
  _(cuGetErrorString, 12000)               \
  _(cuMulticastAddDevice, 12030)           \
  _(cuMulticastBindMem, 12030)             \
  _(cuMulticastCreate, 12030)

#define C10_NVML_DRIVER_API(_)            \
  _(nvmlInit_v2)                          \
  _(nvmlDeviceGetHandleByPciBusId_v2)     \
  _(nvmlDeviceGetNvLinkRemoteDeviceType)  \
  _(nvmlDeviceGetNvLinkRemotePciInfo_v2)  \
  _(nvmlDeviceGetComputeRunningProcesses) \
  _(nvmlSystemGetCudaDriverVersion_v2)

namespace c10::cuda {

struct DriverAPI {
#define CREATE_MEMBER_VERSIONED(name, version) decltype(&name) name##_;
#define CREATE_MEMBER(name) decltype(&name) name##_;
  C10_LIBCUDA_DRIVER_API(CREATE_MEMBER_VERSIONED)
  C10_NVML_DRIVER_API(CREATE_MEMBER)
#undef CREATE_MEMBER_VERSIONED
#undef CREATE_MEMBER

  static DriverAPI* get();
  static void* get_nvml_handle();
};

} // namespace c10::cuda
