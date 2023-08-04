#pragma once
#include <cuda.h>
#define NVML_NO_UNVERSIONED_FUNC_DEFS
#include <nvml.h>

#define C10_CUDA_DRIVER_CHECK(EXPR)                                        \
  do {                                                                     \
    CUresult __err = EXPR;                                                 \
    if (__err != CUDA_SUCCESS) {                                           \
      const char* err_str;                                                 \
      CUresult get_error_str_err C10_UNUSED =                              \
          c10::cuda::DriverAPI::get()->cuGetErrorString_(__err, &err_str); \
      if (get_error_str_err != CUDA_SUCCESS) {                             \
        AT_ERROR("CUDA driver error: unknown error");                      \
      } else {                                                             \
        AT_ERROR("CUDA driver error: ", err_str);                          \
      }                                                                    \
    }                                                                      \
  } while (0)

#define C10_FORALL_DRIVER_LIBRARIES(_) \
  _("libcuda.so", 0)                   \
  _("libnvidia-ml.so.1", 1)

#define C10_FORALL_DRIVER_API(_)         \
  _(cuMemAddressReserve, 0)              \
  _(cuMemRelease, 0)                     \
  _(cuMemMap, 0)                         \
  _(cuMemAddressFree, 0)                 \
  _(cuMemSetAccess, 0)                   \
  _(cuMemUnmap, 0)                       \
  _(cuMemCreate, 0)                      \
  _(cuGetErrorString, 0)                 \
  _(nvmlInit_v2, 1)                      \
  _(nvmlDeviceGetHandleByPciBusId_v2, 1) \
  _(nvmlDeviceGetComputeRunningProcesses, 1)

namespace c10 {
namespace cuda {

struct DriverAPI {
#define CREATE_MEMBER(name, n) decltype(&name) name##_;
  C10_FORALL_DRIVER_API(CREATE_MEMBER)
#undef CREATE_MEMBER
  static DriverAPI* get();
};

} // namespace cuda
} // namespace c10
