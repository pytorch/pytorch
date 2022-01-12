#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/cuda/Activation.h>

#include <ATen/core/DimVector.h>
#include <ATen/core/Tensor.h>
#include <ATen/TensorIterator.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/native/Resize.h>

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
#include <ATen/ops/prelu_backward_native.h>
#include <ATen/ops/prelu_native.h>
#endif

namespace at { namespace native {

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

// -----------------------------------
// prelu forward
// -----------------------------------

Tensor prelu_cuda(const Tensor& self, const Tensor& weight_) {
  TORCH_CHECK(self.is_cuda());
  TORCH_CHECK(weight_.is_cuda());

  auto input = self.contiguous();
  auto weight = weight_.contiguous();

  TORCH_CHECK(input.is_contiguous());
  TORCH_CHECK(weight.is_contiguous());

  int64_t weight_num = weight.numel();
  Tensor result = at::empty_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);

  // case1: shared weight for all channels
  if (weight_num == 1) {
    auto iter = TensorIterator::unary_op(result, input);
    launch_prelu_cuda_kernel_share_weights(iter, weight);
  }
  else { // case2: multiple weights, one for each channel
    launch_prelu_cuda_kernel_multi_weights(result, input, weight);
  }
  return result;
}

// -----------------------------------
// prelu backward
// -----------------------------------

std::tuple<Tensor, Tensor> prelu_backward_cuda(const Tensor& grad_out_, const Tensor& self, const Tensor& weight_) {
  TORCH_CHECK(grad_out_.is_cuda());
  TORCH_CHECK(self.is_cuda());
  TORCH_CHECK(weight_.is_cuda());

  auto input = self.contiguous();
  auto grad_out = grad_out_.contiguous();
  auto weight = weight_.contiguous();

  TORCH_CHECK(input.is_contiguous());
  TORCH_CHECK(weight.is_contiguous());
  TORCH_CHECK(grad_out.is_contiguous());

  int64_t weight_num = weight.numel();
  auto dims = input.dim();
  Tensor input_grad = at::empty_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  Tensor weight_grad = at::empty_like(weight, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  Tensor weight_grad_collector = at::empty_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  // case1: shared parameter for all channels
  if (weight_num == 1) {
    at::TensorIterator iter = TensorIteratorConfig()
        .add_output(input_grad)
        .add_output(weight_grad_collector)
        .add_input(input)
        .add_input(grad_out)
        .build();

    launch_prelu_cuda_backward_kernel_share_weights(iter, weight);
    weight_grad.fill_(weight_grad_collector.sum());
  }
  else { // case2: multiple parameters, one for each channel
    launch_prelu_cuda_backward_kernel_multi_weights(
        input, weight, grad_out, input_grad, weight_grad_collector);
    // update weight_grad
    std::vector<int64_t> reduce_dims;
    reduce_dims.push_back(0);
    if (dims > 2) {
      for(int64_t i = 2; i < dims; i++) reduce_dims.push_back(i);
    }
    weight_grad = weight_grad_collector.sum(reduce_dims);
  }
  return std::tuple<Tensor, Tensor>{input_grad, weight_grad};
}

TORCH_IMPL_FUNC(gelu_out_cuda) (
    const Tensor& /*self*/, const Tensor& /*result*/
  ) {
  GeluCUDAKernelImpl(*this);
}

TORCH_IMPL_FUNC(gelu_backward_out_cuda) (
    const Tensor& /*grad*/, const Tensor& /*self*/, const Tensor& /*grad_input*/
  ) {
  GeluBackwardCUDAKernelImpl(*this);
}

}}  // namespace at::native
