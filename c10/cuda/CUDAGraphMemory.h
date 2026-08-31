#pragma once

#include <c10/core/Device.h>
#include <c10/cuda/CUDAGraphsC10Utils.h>
#include <c10/cuda/CUDAMacros.h>

#include <cstddef>
#include <optional>

namespace c10::cuda::CUDAGraphMemory {

struct CaptureRegistration {
  CaptureId_t capture_id;
  std::optional<CaptureId_t> parent_capture_id;
};

// Graph-aware capture lifecycle hooks. The native allocator receives the full
// registration; other allocator backends retain their legacy capture hooks.
C10_CUDA_API void markCaptureBegin(
    c10::DeviceIndex device,
    const CaptureRegistration& registration);
C10_CUDA_API size_t
markCaptureEnd(c10::DeviceIndex device, CaptureId_t capture_id);

} // namespace c10::cuda::CUDAGraphMemory
