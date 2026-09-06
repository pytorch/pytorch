#include <c10/cuda/impl/CUDAGraphMemory.h>

#include <c10/util/Exception.h>

namespace c10::cuda::CUDAGraphMemory {

void CaptureTracker::captureBegin() {
  ++active_captures_;
}

void CaptureTracker::captureEnd() {
  TORCH_INTERNAL_ASSERT(
      active_captures_ > 0, "captureEnd called with no active capture");
  --active_captures_;
}

} // namespace c10::cuda::CUDAGraphMemory
