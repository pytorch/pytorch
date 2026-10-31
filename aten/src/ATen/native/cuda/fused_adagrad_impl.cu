#include <ATen/native/cuda/fused_adagrad_impl.cuh>

#include <ATen/Dispatch.h>
#include <ATen/native/ForeachUtils.h>
#include <ATen/native/cuda/FusedOptimizerUtils.cuh>
#include <ATen/native/cuda/MultiTensorApply.cuh>
#include <ATen/native/cuda/fused_adagrad_utils.cuh>
#include <algorithm>

namespace at::native {

void _fused_adagrad_cuda_impl_(
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
  const auto step_dtype = state_steps[0].scalar_type();
  TORCH_CHECK(
      std::all_of(
          state_steps.begin(),
          state_steps.end(),
          [step_dtype](const at::Tensor& step) {
            return step.scalar_type() == step_dtype;
          }),
      "Fused optimizer step tensors must have the same dtype");
  auto tensor_lists =
      c10::make_nested<Tensor>(params.vec(), grads.vec(), state_sums.vec());

  const float* grad_scale_ptr =
      grad_scale.has_value() ? grad_scale->data_ptr<float>() : nullptr;
  const float* found_inf_ptr =
      found_inf.has_value() ? found_inf->data_ptr<float>() : nullptr;
  const float* lr_ptr = nullptr;

  AT_DISPATCH_FLOATING_TYPES_AND2(
      kHalf,
      kBFloat16,
      params[0].scalar_type(),
      "fused_adagrad_kernel_cuda",
      [&]() {
        AT_DISPATCH_FUSED_OPTIMIZER_STEP_TYPES(
            state_steps[0].scalar_type(), "fused_adagrad_step_cuda", [&]() {
              multi_tensor_apply_for_fused_optimizer<3>(
                  tensor_lists,
                  state_steps,
                  FusedAdagradMathFunctor<scalar_t, step_t>(),
                  lr_ptr, // unused
                  lr,
                  lr_decay,
                  weight_decay,
                  eps,
                  maximize,
                  grad_scale_ptr,
                  found_inf_ptr);
            });
      });
}

void _fused_adagrad_cuda_impl_(
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
  auto tensor_lists =
      c10::make_nested<Tensor>(params.vec(), grads.vec(), state_sums.vec());

  const float* grad_scale_ptr =
      grad_scale.has_value() ? grad_scale->data_ptr<float>() : nullptr;
  const float* found_inf_ptr =
      found_inf.has_value() ? found_inf->data_ptr<float>() : nullptr;
  const float* lr_ptr = lr.const_data_ptr<float>();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      kHalf,
      kBFloat16,
      params[0].scalar_type(),
      "fused_adagrad_kernel_cuda",
      [&]() {
        AT_DISPATCH_FUSED_OPTIMIZER_STEP_TYPES(
            state_steps[0].scalar_type(), "fused_adagrad_step_cuda", [&]() {
              multi_tensor_apply_for_fused_optimizer<3>(
                  tensor_lists,
                  state_steps,
                  FusedAdagradMathFunctor<scalar_t, step_t>(),
                  lr_ptr,
                  1.0, // unused
                  lr_decay,
                  weight_decay,
                  eps,
                  maximize,
                  grad_scale_ptr,
                  found_inf_ptr);
            });
      });
}

} // namespace at::native
