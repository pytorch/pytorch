#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/cuda/Activation.h>

#include <ATen/core/DimVector.h>
#include <ATen/core/Tensor.h>
#include <ATen/TensorIterator.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/native/Resize.h>
#include <c10/util/irange.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/gelu_backward_native.h>
#include <ATen/ops/gelu_native.h>
#include <ATen/ops/glu_backward_native.h>
#include <ATen/ops/log_sigmoid_forward_native.h>
#endif

namespace at::native {

// -----------------------------------
// glu backward
// -----------------------------------

Tensor& glu_backward_cuda_out(const Tensor& grad_output, const Tensor& input,
                              int64_t dim, Tensor& grad_input) {
  TORCH_CHECK(input.dim() > 0, "glu does not support 0-dimensional tensors");
  auto wrap_dim = maybe_wrap_dim(dim, input.dim());
  auto input_sizes = input.sizes();
  const int64_t nIn = input_sizes[wrap_dim];
  TORCH_CHECK(nIn % 2 == 0, "Halving dimension must be even, but dimension ",
              wrap_dim, " is size ", nIn);

  resize_output(grad_input, input_sizes);

  DimVector iter_shape(input_sizes);
  const auto dim_size = nIn / 2;
  iter_shape[wrap_dim] = dim_size;
  TORCH_CHECK(grad_output.sizes() == IntArrayRef{iter_shape});

  const auto iter = at::TensorIteratorConfig()
    .add_output(grad_input)
    .add_input(input)
    .add_input(grad_output)
    .resize_outputs(false)
    .declare_static_shape(iter_shape)
    .build();

  if (iter.numel() == 0) {
    return grad_input;
  }

  const auto I_stride = input.strides()[wrap_dim] * dim_size;
  const auto gI_stride = grad_input.strides()[wrap_dim] * dim_size;

  if (iter.can_use_32bit_indexing()) {
    launch_glu_backward_kernel(iter, gI_stride, I_stride);
  } else {
    for (const auto& sub_iter: iter.with_32bit_indexing()) {
      launch_glu_backward_kernel(sub_iter, gI_stride, I_stride);
    }
  }
  return grad_input;
}

Tensor glu_backward_cuda(const Tensor& grad_output, const Tensor& input, int64_t dim) {
  auto grad_input = at::empty({0}, input.options());
  return glu_backward_cuda_out(grad_output, input, dim, grad_input);
}

// -----------------------------------
// log_sigmoid forward
// -----------------------------------

std::tuple<Tensor&, Tensor&> log_sigmoid_forward_out_cuda(const Tensor& input, Tensor& result, Tensor& buffer) {
  // NOTE: buffer is only used by CPU dispatch, we just ignore it here
  auto iter = TensorIteratorConfig()
    .add_output(result)
    .add_input(input)
    .build();
  launch_log_sigmoid_forward_kernel(iter);
  return std::forward_as_tuple(result, buffer);
}

std::tuple<Tensor, Tensor> log_sigmoid_forward_cuda(const Tensor& input) {
  auto result = at::empty_like(input);
  auto buffer = at::empty({0}, input.options());
  log_sigmoid_forward_out_cuda(input, result, buffer);
  return std::forward_as_tuple(result, buffer);
}

TORCH_IMPL_FUNC(gelu_out_cuda) (
  const Tensor& /*self*/, c10::string_view approximate, const Tensor& /*result*/
) {
  GeluCUDAKernelImpl(*this, get_gelutype_enum(approximate));
}

TORCH_IMPL_FUNC(gelu_backward_out_cuda) (
  const Tensor& /*grad*/, const Tensor& /*self*/, c10::string_view approximate, const Tensor& /*grad_input*/
) {
  GeluBackwardCUDAKernelImpl(*this, get_gelutype_enum(approximate));
}

}  // namespace at::native
