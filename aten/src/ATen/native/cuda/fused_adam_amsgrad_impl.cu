#include <ATen/native/cuda/fused_adam_amsgrad_impl.cuh>

#include <ATen/Dispatch.h>
#include <ATen/native/ForeachUtils.h>
#include <ATen/native/cuda/fused_adam_utils.cuh>
#include <ATen/native/cuda/MultiTensorApply.cuh>
#include <vector>

namespace at::native {

void _fused_adam_amsgrad_cuda_impl_(
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
    const bool maximize,
    const c10::optional<at::Tensor>& grad_scale,
    const c10::optional<at::Tensor>& found_inf
) {
  std::vector<std::vector<at::Tensor>> tensor_lists{
    params.vec(), grads.vec(), exp_avgs.vec(), exp_avg_sqs.vec(), max_exp_avg_sqs.vec() };

  float* grad_scale_ptr = grad_scale.has_value() ? grad_scale->data_ptr<float>() : nullptr;
  float* found_inf_ptr = found_inf.has_value() ? found_inf->data_ptr<float>() : nullptr;
  float* lr_ptr = nullptr;

  AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, params[0].scalar_type(),
      "fused_adam_kernel_cuda", [&]() {
        multi_tensor_apply_for_fused_optimizer<5>(
            tensor_lists,
            state_steps,
            FusedAdamMathFunctor<scalar_t, 5>(),
            lr_ptr,  // unused
            lr,
            beta1,
            beta2,
            weight_decay,
            eps,
            maximize,
            /* amsgrad */true,
            grad_scale_ptr,
            found_inf_ptr,
            ADAM_MODE::ORIGINAL);
        });
}

// The following overload simply has a Tensor lr
void _fused_adam_amsgrad_cuda_impl_(
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
    const bool maximize,
    const c10::optional<at::Tensor>& grad_scale,
    const c10::optional<at::Tensor>& found_inf
) {
  std::vector<std::vector<at::Tensor>> tensor_lists{
    params.vec(), grads.vec(), exp_avgs.vec(), exp_avg_sqs.vec(), max_exp_avg_sqs.vec() };

  float* grad_scale_ptr = grad_scale.has_value() ? grad_scale->data_ptr<float>() : nullptr;
  float* found_inf_ptr = found_inf.has_value() ? found_inf->data_ptr<float>() : nullptr;
  float* lr_ptr = lr.data_ptr<float>();

  AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, params[0].scalar_type(),
      "fused_adam_kernel_cuda", [&]() {
        multi_tensor_apply_for_fused_optimizer<5>(
            tensor_lists,
            state_steps,
            FusedAdamMathFunctor<scalar_t, 5>(),
            lr_ptr,
            1.0,  // unused
            beta1,
            beta2,
            weight_decay,
            eps,
            maximize,
            /* amsgrad */true,
            grad_scale_ptr,
            found_inf_ptr,
            ADAM_MODE::ORIGINAL);
        });
}

} // namespace at::native
