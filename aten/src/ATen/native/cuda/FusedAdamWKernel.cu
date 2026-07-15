#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/TypeDefault.h>
#include <ATen/native/ForeachUtils.h>
#include <c10/util/Exception.h>
#include <ATen/native/cuda/fused_adamw_amsgrad_impl.cuh>
#include <ATen/native/cuda/fused_adamw_impl.cuh>

namespace at::native {

// note(crcrpar): To observe the CI rules, i.e. 20 minutes per file to compile,
// defensively split instantiations into _impl files. this is only for CUDA 11.3
// for which it took about 20 minutes and 28 minutes in my workstation and CI,
// respectively. As a data point, it took about 20 seconds for CUDA 11.7
// installed in my environment. See
// https://github.com/pytorch/pytorch/pull/81705 for details.
void _fused_adamw_kernel_cuda_(
    at::TensorList params,
    at::TensorList grads,
    at::TensorList exp_avgs,
    at::TensorList exp_avg_sqs,
    at::TensorList max_exp_avg_sqs,
    at::TensorList state_steps,
    const double lr,
    const double beta1,
    const double beta2,
    const double weight_decay,
    const double eps,
    const bool amsgrad,
    const bool maximize,
    const std::optional<at::Tensor>& grad_scale,
    const std::optional<at::Tensor>& found_inf) {
  if (amsgrad) {
    TORCH_CHECK(
        at::native::check_fast_path_restrictions(
            {params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs}),
        "params, grads, exp_avgs, exp_avg_sqs, and max_exp_avg_sqs must have same dtype, device, and layout");
    _fused_adamw_amsgrad_cuda_impl_(
        params,
        grads,
        exp_avgs,
        exp_avg_sqs,
        max_exp_avg_sqs,
        state_steps,
        lr,
        beta1,
        beta2,
        weight_decay,
        eps,
        maximize,
        grad_scale,
        found_inf);
  } else {
    TORCH_CHECK(
        at::native::check_fast_path_restrictions(
            {params, grads, exp_avgs, exp_avg_sqs}),
        "params, grads, exp_avgs, and exp_avg_sqs must have same dtype, device, and layout");
    _fused_adamw_cuda_impl_(
        params,
        grads,
        exp_avgs,
        exp_avg_sqs,
        state_steps,
        lr,
        beta1,
        beta2,
        weight_decay,
        eps,
        maximize,
        grad_scale,
        found_inf);
  }
}

// The following overload simply has a Tensor lr
void _fused_adamw_kernel_cuda_(
    at::TensorList params,
    at::TensorList grads,
    at::TensorList exp_avgs,
    at::TensorList exp_avg_sqs,
    at::TensorList max_exp_avg_sqs,
    at::TensorList state_steps,
    const at::Tensor& lr,
    const double beta1,
    const double beta2,
    const double weight_decay,
    const double eps,
    const bool amsgrad,
    const bool maximize,
    const std::optional<at::Tensor>& grad_scale,
    const std::optional<at::Tensor>& found_inf) {
  if (lr.is_cpu()) {
    _fused_adamw_kernel_cuda_(
        params,
        grads,
        exp_avgs,
        exp_avg_sqs,
        max_exp_avg_sqs,
        state_steps,
        lr.item<double>(),
        beta1,
        beta2,
        weight_decay,
        eps,
        amsgrad,
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

  if (amsgrad) {
    TORCH_CHECK(
        at::native::check_fast_path_restrictions(
            {params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs}),
        "params, grads, exp_avgs, exp_avg_sqs, and max_exp_avg_sqs must have same dtype, device, and layout");
    _fused_adamw_amsgrad_cuda_impl_(
        params,
        grads,
        exp_avgs,
        exp_avg_sqs,
        max_exp_avg_sqs,
        state_steps,
        lr,
        beta1,
        beta2,
        weight_decay,
        eps,
        maximize,
        grad_scale,
        found_inf);
  } else {
    TORCH_CHECK(
        at::native::check_fast_path_restrictions(
            {params, grads, exp_avgs, exp_avg_sqs}),
        "params, grads, exp_avgs, and exp_avg_sqs must have same dtype, device, and layout");
    _fused_adamw_cuda_impl_(
        params,
        grads,
        exp_avgs,
        exp_avg_sqs,
        state_steps,
        lr,
        beta1,
        beta2,
        weight_decay,
        eps,
        maximize,
        grad_scale,
        found_inf);
  }
}

} // namespace at::native
