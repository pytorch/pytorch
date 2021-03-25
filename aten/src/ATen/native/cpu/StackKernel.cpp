// Copyright 2004-present Facebook. All Rights Reserved.

#include <ATen/ATen.h>

#include <ATen/Dispatch.h>
#include <ATen/cpu/vec256/functional.h>
#include <ATen/cpu/vec256/vec256.h>
#include <ATen/native/cpu/StackKernel.h>

namespace at {
namespace native {

namespace {

struct InputMeta {
  void* data_ptr;
  int64_t inner_size;

  InputMeta(const Tensor& t, int64_t dim, int64_t inner)
      : data_ptr(t.data_ptr()), inner_size(t.sizes()[dim] * inner) {}
};

template <typename scalar_t>
void stack_serial_kernel_impl(Tensor& result, TensorList tensors, int64_t dim) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      dim >= 0 && dim <= result.dim(),
      "dim out of range in stack_serial_kernel_impl");
  int64_t outer =
      result.numel() / (result.sizes()[dim] * result.strides()[dim]);
  scalar_t* result_data = result.data_ptr<scalar_t>();
  int64_t ninputs = tensors.size();
  std::vector<InputMeta> inputs;
  inputs.reserve(ninputs);
  for (auto const& tensor : tensors) {
    inputs.emplace_back(tensor, dim, tensor.strides()[dim]);
  }

  using Vec = vec256::Vec256<scalar_t>;
  scalar_t* result_ptr = result_data;
  for (int64_t i = 0; i < outer; ++i) {
    for (int64_t j = 0; j < ninputs; j++) {
      int64_t local_inner = inputs[j].inner_size;
      scalar_t* input_ptr = (scalar_t*)(inputs[j].data_ptr) + i * local_inner;

      if (local_inner < Vec::size()) {
#if !defined(_MSC_VER) && !defined(COMPILING_FOR_MIN_SIZE)
#pragma unroll
#endif
        for (int64_t k = 0; k < local_inner; k++) {
          result_ptr[k] = input_ptr[k];
        }
      } else {
        vec256::map(
            [](Vec x) { return x; }, result_ptr, input_ptr, local_inner);
      }
      result_ptr += local_inner;
    }
  }
}

void stack_serial_kernel(Tensor& result, TensorList tensors, int64_t dim) {
  AT_DISPATCH_FLOATING_TYPES(
      result.scalar_type(), "stack_serial_kernel", [&]() {
        stack_serial_kernel_impl<scalar_t>(result, tensors, dim);
      });
}

} // anonymous namespace

REGISTER_DISPATCH(stack_serial_stub, &stack_serial_kernel);

} // namespace native
} // namespace at
