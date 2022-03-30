// Copyright 2004-present Facebook. All Rights Reserved.
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>

#include <ATen/Dispatch.h>
#include <ATen/native/cpu/StackKernel.h>
#include <ATen/native/cpu/SerialStackImpl.h>

namespace at {
namespace native {

namespace {

void stack_serial_kernel(Tensor& result, TensorList tensors, int64_t dim) {
  AT_DISPATCH_FLOATING_TYPES(
      result.scalar_type(), "stack_serial_kernel", [&]() {
        detail::stack_serial_kernel_impl<scalar_t, TensorList>(result, tensors, dim);
      });
}

} // anonymous namespace

REGISTER_DISPATCH(stack_serial_stub, &stack_serial_kernel);

} // namespace native
} // namespace at
