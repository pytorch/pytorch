#pragma once

#include <torch/csrc/inductor/aoti_runtime/utils_cuda.h>
#include <torch/csrc/inductor/aoti_torch/generated/c_shim_cuda.h>

#if defined(USE_CUDA) && !defined(USE_ROCM)
namespace torch::aot_inductor {

// CUDA 13 cuda.h values, spelled out so older toolkits still build:
// CU_DEVICE_ATTRIBUTE_MAX_OVERSIZED_SHARED_MEMORY_PER_BLOCK,
// CU_FUNC_ATTRIBUTE_SHARED_MEMORY_MODE and
// CU_SHARED_MEMORY_MODE_ALLOW_OVERSIZED_SHARED_MEMORY.
constexpr int kDeviceAttrMaxOversizedSharedMemoryPerBlock = 150;
constexpr int kFuncAttrSharedMemoryMode = 17;
constexpr int kSharedMemoryModeAllowOversized = 3;

// Max static + dynamic shared memory per block for a kernel needing
// sharedNeeded bytes. Some devices (sm_107) allow more than the opt-in limit,
// but only in the oversized shared memory mode, so a result above sharedOptin
// means the kernel must use that mode. Drivers that don't report the oversized
// limit keep the opt-in limit. deviceGetAttribute is a parameter because the
// static launcher reaches the driver through nvrtc().
inline int sharedMemoryLimit(
    uint32_t sharedNeeded,
    int sharedOptin,
    CUdevice device,
    decltype(&cuDeviceGetAttribute) deviceGetAttribute) {
  int sharedOversized = 0;
  if (sharedNeeded > static_cast<uint32_t>(sharedOptin) &&
      deviceGetAttribute(
          &sharedOversized,
          static_cast<CUdevice_attribute>(
              kDeviceAttrMaxOversizedSharedMemoryPerBlock),
          device) == CUDA_SUCCESS &&
      sharedOversized > sharedOptin) {
    return sharedOversized;
  }
  return sharedOptin;
}

} // namespace torch::aot_inductor
#endif
