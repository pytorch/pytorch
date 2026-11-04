#pragma once

#include <c10/core/Device.h>
#include <c10/cuda/CUDAGraphsC10Utils.h>
#include <c10/cuda/CUDAMacros.h>

#include <cstddef>
#include <optional>

namespace c10::cuda::CUDAGraphMemory {

struct CaptureRegistration {
  CaptureId_t capture_id{};
  std::optional<CaptureId_t> parent_capture_id;
};

// Notify the active allocator of capture lifecycle changes. End returns the
// number of invalid conditional-scope frees observed during the capture.
C10_CUDA_API void markCaptureBegin(
    c10::DeviceIndex device,
    const CaptureRegistration& registration);
C10_CUDA_API size_t
markCaptureEnd(c10::DeviceIndex device, CaptureId_t capture_id);

} // namespace c10::cuda::CUDAGraphMemory
