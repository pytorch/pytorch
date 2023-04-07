#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/impl/CUDAGuardImpl.h>

namespace c10 {
namespace cuda {
namespace impl {

void CUDAGuardImpl::recordDataPtrOnStream(
    const c10::DataPtr& data_ptr,
    const Stream& stream) const {
  CUDAStream cuda_stream{stream};
  CUDACachingAllocator::recordStream(data_ptr, cuda_stream);
}

constexpr DeviceType CUDAGuardImpl::static_type;

C10_REGISTER_GUARD_IMPL(CUDA, CUDAGuardImpl);

} // namespace impl
} // namespace cuda
} // namespace c10
