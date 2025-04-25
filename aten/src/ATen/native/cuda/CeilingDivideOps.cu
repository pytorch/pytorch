#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/native/BinaryOps.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/Dispatch.h>
#include <ATen/native/cuda/Loops.cuh>

namespace at::native {

// These functions are needed to fix the undefined reference errors
// They call into the existing CPU implementation which will dispatch
// to the CUDA kernel we just implemented through div_ceil_stub

Tensor ceiling_divide(const Tensor& self, const Tensor& other) {
  Tensor result;
  auto iter = TensorIterator::binary_op(result, self, other);
  div_ceil_stub(iter.device_type(), iter);
  return iter.output();
}

Tensor& ceiling_divide_(Tensor& self, const Tensor& other) {
  return ceiling_divide_out(self, other, self);
}

Tensor& ceiling_divide_out(const Tensor& self, const Tensor& other, Tensor& result) {
  auto iter = TensorIterator::binary_op(result, self, other);
  div_ceil_stub(iter.device_type(), iter);
  if (!result.defined()) {
    result = iter.output();
  }
  return result;
}

} // namespace at::native 