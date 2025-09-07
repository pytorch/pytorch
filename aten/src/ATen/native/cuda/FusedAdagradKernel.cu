#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/TypeDefault.h>
#include <ATen/native/ForeachUtils.h>
#include <ATen/native/cuda/fused_adagrad_impl.cuh>

namespace at::native {

void _fused_adagrad_kernel_cuda_(
    at::TensorList params,
    at::TensorList grads,
    at::TensorList state_sums,
    at::TensorList state_steps,
    const double lr,
    const double lr_decay,
    const double weight_decay,
    const double eps,
    const bool maximize,
    const std::optional<at::Tensor>& grad_scale,
    const std::optional<at::Tensor>& found_inf) {
  TORCH_CHECK(
      at::native::check_fast_path_restrictions({params, grads, state_sums}),
      "params, grads, and state_sums must have same dtype, device, and layout");
  _fused_adagrad_cuda_impl_(
      params,
      grads,
      state_sums,
      state_steps,
      lr,
      lr_decay,
      weight_decay,
      eps,
      maximize,
      grad_scale,
      found_inf);
}

void _fused_adagrad_kernel_cuda_(
    at::TensorList params,
    at::TensorList grads,
    at::TensorList state_sums,
    at::TensorList state_steps,
    const at::Tensor& lr,
    const double lr_decay,
    const double weight_decay,
    const double eps,
    const bool maximize,
    const std::optional<at::Tensor>& grad_scale,
    const std::optional<at::Tensor>& found_inf) {
  if (lr.is_cpu()) {
    _fused_adagrad_kernel_cuda_(
        params,
        grads,
        state_sums,
        state_steps,
        lr.item<double>(),
        lr_decay,
        weight_decay,
        eps,
        maximize,
        grad_scale,
        found_inf);
    return;
  }

  // Manually check devices since we specify no device check in
  // native_functions.yaml
  Device param_device = params[0].device();
  if (grad_scale.has_value()) {
    TORCH_CHECK(
        grad_scale->device() == param_device,
        "grad_scale must be on the same GPU device as the params");
  }
  if (found_inf.has_value()) {
    TORCH_CHECK(
        found_inf->device() == param_device,
        "found_inf must be on the same GPU device as the params");
  }
  TORCH_CHECK(
      lr.device() == param_device,
      "lr must be on the same GPU device as the params");

  TORCH_CHECK(
      at::native::check_fast_path_restrictions({params, grads, state_sums}),
      "params and grads must have same dtype, device, and layout");
  _fused_adagrad_cuda_impl_(
      params,
      grads,
      state_sums,
      state_steps,
      lr,
      lr_decay,
      weight_decay,
      eps,
      maximize,
      grad_scale,
      found_inf);
}

} // namespace at::native