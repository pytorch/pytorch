#include <c10/cuda/CUDAGraphsC10Utils.h>
#include <c10/cuda/impl/CUDAGuardImpl.h>

namespace c10::cuda::impl {

uint64_t CUDAGuardImpl::getStreamCaptureId(const Stream& stream) const {
  const auto info = captureInfoMayInitCtx(CUDAStream(stream));
  TORCH_CHECK(
      info.status != CaptureStatus::Invalidated,
      "Cannot query the capture ID of an invalidated CUDA graph capture");
  return info.status == CaptureStatus::Active ? info.id : 0;
}

C10_REGISTER_GUARD_IMPL(CUDA, CUDAGuardImpl)

} // namespace c10::cuda::impl
