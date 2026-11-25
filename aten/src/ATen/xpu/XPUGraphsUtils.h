#pragma once

#include <c10/xpu/XPUGraphsC10Utils.h>
#include <optional>

namespace at::xpu {

inline std::optional<size_t> currentStreamCaptureId() {
  auto& queue = c10::xpu::getCurrentXPUStream().queue();
  if (queue.ext_oneapi_get_state() == queue_state::recording) {
    return queue.ext_oneapi_get_graph().get_id();
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
