#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <c10/cuda/CUDACachingAllocator.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/record_stream_native.h>
#endif

namespace at::native {
void record_stream_cuda(Tensor& self, c10::Stream stream) {
  struct c10::StreamData3 data = stream.pack3();
  auto cuda_stream = at::cuda::CUDAStream::unpack3(data.stream_id, data.device_index, data.device_type);
  const auto& storage = self.storage();
  c10::cuda::CUDACachingAllocator::recordStream(storage.data_ptr(), cuda_stream);
  // Uses of the tensor before it was materialized from a lazy clone read the
  // data it was materialized from.
  if (const auto* data_ptr = storage.unsafeGetStorageImpl()->cow_materialized_from()) {
    c10::cuda::CUDACachingAllocator::recordStream(*data_ptr, cuda_stream);
  }
}
}  // namespace at::native
