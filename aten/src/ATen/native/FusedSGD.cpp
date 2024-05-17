#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/FusedSGD.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_fused_sgd.h>
#include <ATen/ops/_fused_sgd_native.h>
#endif
namespace at {

namespace native {


void _fused_sgd_kernel_cpu_(
    at::TensorList params,
    at::TensorList grads,
    at::TensorList momentum_buffer_list,
    const double weight_decay,
    const double momentum,
    const double lr,
    const double dampening,
    const bool nesterov,
    const bool maximize,
    const bool is_first_step,
    const std::optional<at::Tensor>& grad_scale,
    const std::optional<at::Tensor>& found_inf) {
  const float* grad_scale_ptr =
      grad_scale.has_value() ? grad_scale->data_ptr<float>() : nullptr;
  const float* found_inf_ptr =
      found_inf.has_value() ? found_inf->data_ptr<float>() : nullptr;
  if (found_inf_ptr && *found_inf_ptr == 1.0) {
      return;
  }
  size_t n_tensors = params.size();
  TORCH_CHECK(grads.size() == n_tensors);
  bool no_momentum_buffer = momentum == 0.0;
  if (no_momentum_buffer) {
    TORCH_CHECK(momentum_buffer_list.size() == 0);
  } else {
    TORCH_CHECK(momentum_buffer_list.size() == n_tensors);
  }
  for (size_t i = 0; i < n_tensors; i++){
    fused_sgd_stub(
      kCPU,
      params[i],
      grads[i],
      no_momentum_buffer ? Tensor() : momentum_buffer_list[i],
      weight_decay,
      momentum,
      lr,
      dampening,
      nesterov,
      maximize,
      is_first_step,
      grad_scale_ptr);
  }
}

void _fused_sgd_kernel_cpu_(
    at::TensorList params,
    at::TensorList grads,
    at::TensorList momentum_buffer_list,
    const double weight_decay,
    const double momentum,
    const at::Tensor& lr,
    const double dampening,
    const bool nesterov,
    const bool maximize,
    const bool is_first_step,
    const std::optional<at::Tensor>& grad_scale,
    const std::optional<at::Tensor>& found_inf) {
    _fused_sgd_kernel_cpu_(
        params, grads, momentum_buffer_list, weight_decay,
        momentum, lr.item<double>(), dampening, nesterov,
        maximize, is_first_step, grad_scale, found_inf
    );
}

DEFINE_DISPATCH(fused_sgd_stub);

}
}
