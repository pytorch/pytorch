#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/FusedAdam.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_fused_adam.h>
#include <ATen/ops/_fused_adam_native.h>
#include <ATen/ops/_fused_adamw.h>
#include <ATen/ops/_fused_adamw_native.h>
#endif
namespace at {

namespace native {

void _fused_adam_kernel_cpu_(
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
  const float* grad_scale_ptr =
      grad_scale.has_value() ? grad_scale->data_ptr<float>() : nullptr;
  const float* found_inf_ptr =
      found_inf.has_value() ? found_inf->data_ptr<float>() : nullptr;
  if (found_inf_ptr && *found_inf_ptr == 1.0) {
      return;
  }
  size_t n_tensors = params.size();
  TORCH_CHECK(grads.size() == n_tensors);
  TORCH_CHECK(exp_avgs.size() == n_tensors);
  TORCH_CHECK(exp_avg_sqs.size() == n_tensors);
  if (amsgrad) {
    TORCH_CHECK(max_exp_avg_sqs.size() == n_tensors);
  } else {
    TORCH_CHECK(max_exp_avg_sqs.size() == 0);
  }
  TORCH_CHECK(state_steps.size() == n_tensors);
  at::Tensor max_exp_avg_sq = at::Tensor();
  for (size_t i = 0; i < n_tensors; i++){
    if (amsgrad) max_exp_avg_sq = max_exp_avg_sqs[i];
    fused_adam_stub(
      kCPU,
      params[i],
      grads[i],
      exp_avgs[i],
      exp_avg_sqs[i],
      max_exp_avg_sq,
      state_steps[i],
      lr,
      beta1,
      beta2,
      weight_decay,
      eps,
      amsgrad,
      maximize,
      grad_scale_ptr,
      ADAM_MODE::ORIGINAL);
  }
}

// The following overload simply has a Tensor lr
void _fused_adam_kernel_cpu_(
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
  _fused_adam_kernel_cpu_(params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps, lr.item<double>(), beta1, beta2, weight_decay, eps, amsgrad, maximize, grad_scale, found_inf);
}

void _fused_adamw_kernel_cpu_(
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
  const float* grad_scale_ptr =
      grad_scale.has_value() ? grad_scale->data_ptr<float>() : nullptr;
  const float* found_inf_ptr =
      found_inf.has_value() ? found_inf->data_ptr<float>() : nullptr;
  if (found_inf_ptr && *found_inf_ptr == 1.0) {
      return;
  }
  size_t n_tensors = params.size();
  TORCH_CHECK(grads.size() == n_tensors);
  TORCH_CHECK(exp_avgs.size() == n_tensors);
  TORCH_CHECK(exp_avg_sqs.size() == n_tensors);
  if (amsgrad) {
    TORCH_CHECK(max_exp_avg_sqs.size() == n_tensors);
  } else {
    TORCH_CHECK(max_exp_avg_sqs.size() == 0);
  }
  TORCH_CHECK(state_steps.size() == n_tensors);
  at::Tensor max_exp_avg_sq = at::Tensor();
  for (size_t i = 0; i < n_tensors; i++){
    if (amsgrad) max_exp_avg_sq = max_exp_avg_sqs[i];
    fused_adam_stub(
      kCPU,
      params[i],
      grads[i],
      exp_avgs[i],
      exp_avg_sqs[i],
      max_exp_avg_sq,
      state_steps[i],
      lr,
      beta1,
      beta2,
      weight_decay,
      eps,
      amsgrad,
      maximize,
      grad_scale_ptr,
      ADAM_MODE::ADAMW);
  }
}

// The following overload simply has a Tensor lr
void _fused_adamw_kernel_cpu_(
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
  _fused_adamw_kernel_cpu_(params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps, lr.item<double>(), beta1, beta2, weight_decay, eps, amsgrad, maximize, grad_scale, found_inf);
}


DEFINE_DISPATCH(fused_adam_stub);

}
}
