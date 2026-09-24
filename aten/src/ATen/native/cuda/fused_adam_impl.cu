#include <ATen/native/cuda/fused_adam_impl.cuh>

#include <ATen/native/cuda/fused_adam_utils.cuh>

namespace at::native {

void _fused_adam_cuda_impl_(
    at::TensorList params,
    at::TensorList grads,
    at::TensorList exp_avgs,
    at::TensorList exp_avg_sqs,
    at::TensorList state_steps,
    const double lr,
    const double beta1,
    const double beta2,
    const double weight_decay,
    const double eps,
    const bool maximize,
    const std::optional<at::Tensor>& grad_scale,
    const std::optional<at::Tensor>& found_inf) {
  _fused_adam_cuda_impl_common<ADAM_MODE::ORIGINAL, /*amsgrad=*/false>(
      params,
      grads,
      exp_avgs,
      exp_avg_sqs,
      /*max_exp_avg_sqs=*/{},
      state_steps,
      /*lr_ptr=*/nullptr,
      lr,
      beta1,
      beta2,
      weight_decay,
      eps,
      maximize,
      grad_scale,
      found_inf);
}

// The following overload simply has a Tensor lr
void _fused_adam_cuda_impl_(
    at::TensorList params,
    at::TensorList grads,
    at::TensorList exp_avgs,
    at::TensorList exp_avg_sqs,
    at::TensorList state_steps,
    const at::Tensor& lr,
    const double beta1,
    const double beta2,
    const double weight_decay,
    const double eps,
    const bool maximize,
    const std::optional<at::Tensor>& grad_scale,
    const std::optional<at::Tensor>& found_inf) {
  _fused_adam_cuda_impl_common<ADAM_MODE::ORIGINAL, /*amsgrad=*/false>(
      params,
      grads,
      exp_avgs,
      exp_avg_sqs,
      /*max_exp_avg_sqs=*/{},
      state_steps,
      /*lr_ptr=*/lr.const_data_ptr<float>(),
      /*lr=*/1.0,
      beta1,
      beta2,
      weight_decay,
      eps,
      maximize,
      grad_scale,
      found_inf);
}

} // namespace at::native
