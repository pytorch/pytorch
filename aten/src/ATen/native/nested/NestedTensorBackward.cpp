#include <ATen/native/nested/NestedTensorMath.h>

#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/core/op_registration/op_registration.h>
#include <ATen/native/layer_norm.h>
#include <ATen/NestedTensorImpl.h>
#include <c10/core/DispatchKey.h>
#include <ATen/native/nested/NestedTensorMath.h>

namespace at {
namespace native {

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

Tensor _reshape_nested_backward(const Tensor& self, const Tensor& grad) {
  auto self_ptr = get_nested_tensor_impl(self);
  // TODO: this is to reproduce self_ptr->opt_sizes_
  //       if an accessor is provided in the future, can replace this
  std::vector<int64_t> sizes;
  for (int64_t i = 0; i < self_ptr->dim(); i++) {
    c10::optional<int64_t> opt_size = self_ptr->opt_size(i);
    if (opt_size.has_value()) {
      sizes.push_back(*opt_size);
    }
    else {
      sizes.push_back(-1);
    }
  }
  return grad.reshape(sizes);
}

} // namespace native
} // namespace at
