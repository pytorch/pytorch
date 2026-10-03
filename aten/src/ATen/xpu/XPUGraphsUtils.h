#pragma once

#include <c10/xpu/XPUGraphsC10Utils.h>
#include <optional>

namespace at::xpu {

inline std::optional<size_t> currentStreamCaptureId() {
  auto& queue = c10::xpu::getCurrentXPUStream().queue();
  if (queue.ext_oneapi_get_state() == queue_state::recording) {
#if SYCL_COMPILER_VERSION >= 20260101
    return queue.ext_oneapi_get_graph().get_id();
#else
    TORCH_CHECK_NOT_IMPLEMENTED(
        false,
        "XPU graph id requires PyTorch to be compiled with oneAPI 2026.1.1 or newer. ",
        "Please rebuild PyTorch with a supported SYCL compiler.");
#endif
  }
  return std::nullopt;
}

inline CaptureStatus currentStreamCaptureStatus() {
  return c10::xpu::currentStreamCaptureStatusMayInitCtx();
}

inline void assertNotCapturing(const std::string& attempt) {
  auto status = currentStreamCaptureStatus();
  TORCH_CHECK(
      status == CaptureStatus::Executing,
      attempt,
      " during XPU graph capture. If you need this call to be captured, "
      "please file an issue. "
      "Current xpuStreamCaptureStatus: ",
      status);
}

} // namespace at::xpu
