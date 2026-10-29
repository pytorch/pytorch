#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/mps/MPSAllocatorInterface.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/record_stream_native.h>
#endif

namespace at::native {
void record_stream_mps(Tensor& self, c10::Stream stream) {
  at::mps::getIMPSAllocator()->recordStream(self.storage().data_ptr(), stream);
}
} // namespace at::native
