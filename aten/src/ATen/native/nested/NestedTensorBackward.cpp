#include <ATen/native/nested/NestedTensorMath.h>

#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/core/op_registration/op_registration.h>
#include <ATen/native/layer_norm.h>
#include <ATen/NestedTensorImpl.h>
#include <c10/core/DispatchKey.h>
#include <ATen/native/nested/NestedTensorUtils.h>

namespace at {
namespace native {

// See Note [nested tensor matmul] in NestedTensorMath.cpp
std::tuple<Tensor, Tensor> matmul_backward_nested(
    const Tensor& grad,
    const Tensor& self,
    const Tensor& other,
    std::array<bool, 2> grad_input_mask) {
  if (!grad.defined()) {
    return std::make_tuple(Tensor(), Tensor());
  }
  Tensor grad_self, grad_other;
  if (grad_input_mask[0]) {
    grad_self = at::matmul(grad, other.transpose(-1, -2));
  }
  if (grad_input_mask[1]) {
    grad_other = at::matmul(self.transpose(-1, -2), grad);
  }
  return std::make_tuple(grad_self, grad_other);
}

std::tuple<Tensor, Tensor, Tensor> nested_linear_backward(
    const Tensor& input,
    const Tensor& grad_output,
    const Tensor& weight,
    std::array<bool, 3> output_mask) {
  if (!grad_output.defined()) {
    return std::tuple<Tensor, Tensor, Tensor>{Tensor(), Tensor(), Tensor()};
  }
  Tensor grad_input, grad_weight, grad_bias;
  auto* nt_grad_output = get_nested_tensor_impl(grad_output);
  auto* nt_input = get_nested_tensor_impl(input);
  TORCH_INTERNAL_ASSERT(nt_grad_output != nullptr);
  TORCH_INTERNAL_ASSERT(nt_input != nullptr);
  TORCH_CHECK(nested_tensor_impl_is_contiguous(nt_grad_output));
  auto grad_ouput_buffer = nt_grad_output->get_buffer();
  auto input_buffer = nt_input->get_buffer();

  auto reshaped_grad = grad_ouput_buffer.reshape({-1, weight.size(0)});

  if (output_mask[0]) {
    auto grad_input_buffer = at::mm(reshaped_grad, weight).view({-1});
    auto grad_input_nt_size = nt_input->get_nested_size_tensor().clone();
    grad_input = wrap_buffer(grad_input_buffer, grad_input_nt_size);
  }
  if (output_mask[1]) {
    grad_weight =
        at::mm(reshaped_grad.t(), input_buffer.reshape({-1, weight.size(1)}));
  }
  if (output_mask[2]) {
    grad_bias = reshaped_grad.sum(0);
  }
  return std::tuple<Tensor, Tensor, Tensor>{grad_input, grad_weight, grad_bias};
}

Tensor nested_softmax_backward(
    const Tensor& grad,
    const Tensor& output,
    int64_t dim,
    ScalarType input_dtype) {
  TORCH_INTERNAL_ASSERT(grad.is_nested(), "Should be nested grad")
  TORCH_INTERNAL_ASSERT(output.is_nested(), "Should be nested output")

  auto output_ptr = get_nested_tensor_impl(output);
  auto grad_ptr = get_nested_tensor_impl(grad);
  int64_t ntensors = output_ptr->size(0);
  if (ntensors == 0) {
    return grad.clone();
  }
  int64_t positive_dim = at::maybe_wrap_dim(dim, output_ptr->dim());

  //  Get the info about the output
  const Tensor &output_buffer = output_ptr->get_buffer(),
               &output_sizemat = output_ptr->get_nested_size_tensor();

  //  Get the info about the grad
  const Tensor &grad_sizemat = grad_ptr->get_nested_size_tensor();

  TORCH_INTERNAL_ASSERT(output_sizemat.equal(grad_sizemat));
  Tensor grad_output =
      wrap_buffer(at::empty_like(output_buffer), output_sizemat.clone());

  // Unbind nt into individual tensor slices for calculating the derivative
  std::vector<Tensor> grad_output_unbind{grad_output.unbind()},
      grad_unbind{grad.unbind()}, output_unbind{output.unbind()};

  for(const auto i: c10::irange(ntensors)) {
    at::_softmax_backward_data_out(
        grad_output_unbind[i],
        grad_unbind[i],
        output_unbind[i],
        positive_dim - 1,
        input_dtype);
  }
  return grad_output;

}

// Rudimentary sum backward assuming the conditions in #82387
Tensor _nested_sum_backward_cpu(
  const Tensor& grad,
  const Tensor& nested_self,
  OptionalIntArrayRef opt_dims,
  bool keepdim) {
  auto nt_self = get_nested_tensor_impl(nested_self);
  auto nt_grad = get_nested_tensor_impl(grad);
  const Tensor& grad_buffer = nt_grad->get_buffer();
  const Tensor& self_buffer = nt_self->get_buffer();
  auto grad_sizes = nt_grad->get_nested_size_tensor();
  auto self_sizes = nt_self->get_nested_size_tensor();
  int64_t ntensors = nt_self->size(0);
  const Tensor& self_grad_buffer = self_buffer.new_empty(self_buffer.sizes());

  auto num_segments = at::prod(grad_sizes, -1);
  auto segment_lengths = self_sizes.select(1, -1);

  // This logic assumes for now that
  // (1) all the gradient nested tensors are contiguous
  // (2) the gradient nested tensors are stored contiguously in the buffer
  AT_DISPATCH_ALL_TYPES_AND2(
    ScalarType::Half, ScalarType::BFloat16, self_grad_buffer.scalar_type(), "nested_sum_dim_cpu", [&]() {
    auto* self_grad_data = self_grad_buffer.data_ptr<scalar_t>();
    const auto* output_grad_data = grad_buffer.data_ptr<scalar_t>();
    int64_t out_idx = 0, in_idx = 0;
    for (const auto i : c10::irange(ntensors)) {
      int64_t segments = num_segments[i].item<int64_t>();
      int64_t segment_length = segment_lengths[i].item<int64_t>();
      for (auto j = 0; j < segments; j++) {
        scalar_t output_grad = output_grad_data[out_idx];
        for (auto k = 0; k < segment_length; k++) {
          self_grad_data[in_idx] = output_grad;
          in_idx += 1;
        }
        out_idx += 1;
      }
    }
  });

  return wrap_buffer(self_grad_buffer, self_sizes);

}


Tensor _nested_select_backward(
  const Tensor& grad,
  const Tensor& nested_self,
  int64_t dim,
  int64_t index) {
  auto nt_self = get_nested_tensor_impl(nested_self);
  const Tensor& self_buffer = nt_self->get_buffer();
  const auto self_sizes = nt_self->get_nested_size_tensor();
  const Tensor& self_grad_buffer = self_buffer.new_zeros(self_buffer.sizes());

  auto nt_grad = wrap_buffer(self_grad_buffer, self_sizes);
  nt_grad.select(dim, index).copy_(grad);

  return nt_grad;
}

} // namespace native
} // namespace at
