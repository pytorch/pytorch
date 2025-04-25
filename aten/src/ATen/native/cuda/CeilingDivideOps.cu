#define TORCH_ASSERT_NO_OPERATORS
// Use lower-level headers to avoid dependencies on native_functions.yaml
#include <c10/core/Scalar.h>
#include <c10/core/TensorOptions.h>
#include <ATen/TensorBase.h>
#include <ATen/EmptyTensor.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/Dispatch.h>
#include <ATen/native/cuda/Loops.cuh>

namespace at::native {

// These functions are needed to fix the undefined reference errors
// They call into the existing CPU implementation which will dispatch
// to the CUDA kernel we just implemented through div_ceil_stub

Tensor ceiling_divide(const Tensor& self, const Tensor& other) {
  // Create empty output tensor
  auto result = at::empty_like(self);
  
  // Set up the iterator
  auto iter = at::TensorIteratorConfig()
      .add_output(result)
      .add_input(self)
      .add_input(other)
      .build();
      
  // Dispatch to the appropriate kernel
  div_ceil_stub(iter.device_type(), iter);
  return result;
}

Tensor& ceiling_divide_(Tensor& self, const Tensor& other) {
  return ceiling_divide_out(self, other, self);
}

Tensor& ceiling_divide_out(const Tensor& self, const Tensor& other, Tensor& result) {
  // Set up the iterator
  auto iter = at::TensorIteratorConfig()
      .add_output(result)
      .add_input(self)
      .add_input(other)
      .build();
      
  // Dispatch to the appropriate kernel
  div_ceil_stub(iter.device_type(), iter);
  return result;
}

} // namespace at::native 