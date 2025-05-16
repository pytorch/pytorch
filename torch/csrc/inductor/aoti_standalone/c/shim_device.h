#pragma once

// This header mimics APIs in aoti_torch/c/shim.h in a standalone way

// TODO: Move DeviceType to a header-only directory
#include <c10/core/DeviceType.h>

#ifdef __cplusplus
extern "C" {
#endif

#define AOTI_TORCH_DEVICE_TYPE_IMPL(device_str, device_type) \
  inline int32_t aoti_torch_device_type_##device_str() {     \
    return (int32_t)c10::DeviceType::device_type;            \
  }

AOTI_TORCH_DEVICE_TYPE_IMPL(cpu, CPU)
AOTI_TORCH_DEVICE_TYPE_IMPL(cuda, CUDA)
AOTI_TORCH_DEVICE_TYPE_IMPL(mps, MPS)
AOTI_TORCH_DEVICE_TYPE_IMPL(xpu, XPU)
#undef AOTI_TORCH_DEVICE_TYPE_IMPL

#ifdef __cplusplus
} // extern "C"
#endif
