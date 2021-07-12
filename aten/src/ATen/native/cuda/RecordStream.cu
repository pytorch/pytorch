#include <ATen/ATen.h>
#include <c10/cuda/CUDACachingAllocator.h>
namespace at { namespace native {
void record_stream_cuda(Tensor& self, c10::Stream stream) {
  c10::cuda::CUDACachingAllocator::recordStream(self.storage().data_ptr(), at::cuda::CUDAStream::unpack(stream.pack()));
}
}}  // namespace at::native
