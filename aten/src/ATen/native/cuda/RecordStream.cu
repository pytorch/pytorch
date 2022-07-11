#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <c10/cuda/CUDACachingAllocator.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/record_stream_native.h>
#endif

namespace at { namespace native {
void record_stream_cuda(Tensor& self, c10::Stream stream) {
  c10::cuda::CUDACachingAllocator::recordStream(self.storage().data_ptr(), at::cuda::CUDAStream::unpack(stream.pack()));
}
}}  // namespace at::native
