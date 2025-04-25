#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/Dispatch.h>
#include <ATen/native/cuda/Loops.cuh>

namespace at::native {

// These functions are needed to fix the undefined reference errors
// They call into the existing CPU implementation which will dispatch
// to the CUDA kernel we just implemented through div_ceil_stub

Tensor ceiling_divide(const Tensor& self, const Tensor& other) {
  Tensor result = at::empty({0}, self.options());
  auto iter = at::TensorIteratorConfig()
      .add_output(result)
      .add_input(self)
      .add_input(other)
      .build();
  div_ceil_stub(iter.device_type(), iter);
  return result;
}

Tensor& ceiling_divide_(Tensor& self, const Tensor& other) {
  return ceiling_divide_out(self, other, self);
}

Tensor& ceiling_divide_out(const Tensor& self, const Tensor& other, Tensor& result) {
  auto iter = at::TensorIteratorConfig()
      .add_output(result)
      .add_input(self)
      .add_input(other)
      .build();
  div_ceil_stub(iter.device_type(), iter);
  return result;
}

} // namespace at::native 