#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/xpu/TensorFactories.h>

#include <c10/xpu/XPUFunctions.h>
#include <c10/xpu/XPUStream.h>

#include <ATen/ops/fill.h>
#include <ATen/ops/zero_native.h>

namespace at {
namespace native {

Tensor& zero_xpu_(Tensor& self) {
  void* const ptr = self.mutable_data_ptr();
  if (ptr != nullptr && self.is_non_overlapping_and_dense()) {
    c10::xpu::getCurrentXPUStream(self.device().index())
        .queue()
        .memset(ptr, 0, self.numel() * self.dtype().itemsize());
    return self;
  }
  return self.fill_(0);
}

} // namespace native
} // namespace at