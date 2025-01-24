#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/mps/operations/FusedAdamAmsgradKernelImpl.h>

#include <ATen/Dispatch.h>
#include <ATen/native/ForeachUtils.h>
#include <ATen/native/mps/operations/MultiTensorApply.h>
#include <vector>

namespace at::native::mps {

void _fused_adam_amsgrad_mps_impl_(TensorList params,
                                   TensorList grads,
                                   TensorList exp_avgs,
                                   TensorList exp_avg_sqs,
                                   TensorList max_exp_avg_sqs,
                                   TensorList state_steps,
                                   const double lr,
                                   const double beta1,
                                   const double beta2,
                                   const double weight_decay,
                                   const double eps,
                                   const bool maximize,
                                   const std::optional<Tensor>& grad_scale,
                                   const std::optional<Tensor>& found_inf) {
  std::vector<std::vector<Tensor>> tensor_lists{
      params.vec(), grads.vec(), exp_avgs.vec(), exp_avg_sqs.vec(), max_exp_avg_sqs.vec()};

  const auto kernel_name =
      "fused_adam_amsgrad_" + scalarToMetalTypeString(params[0]) + "_" + scalarToMetalTypeString(state_steps[0]);

  multi_tensor_apply_for_fused_optimizer<5, 512>(kernel_name,
                                                 tensor_lists,
                                                 state_steps,
                                                 FusedAdamEncodingFunctor(),
                                                 lr,
                                                 beta1,
                                                 beta2,
                                                 weight_decay,
                                                 eps,
                                                 maximize);
}

void _fused_adam_amsgrad_mps_impl_(TensorList params,
                                   TensorList grads,
                                   TensorList exp_avgs,
                                   TensorList exp_avg_sqs,
                                   TensorList max_exp_avg_sqs,
                                   TensorList state_steps,
                                   const Tensor& lr,
                                   const double beta1,
                                   const double beta2,
                                   const double weight_decay,
                                   const double eps,
                                   const bool maximize,
                                   const std::optional<Tensor>& grad_scale,
                                   const std::optional<Tensor>& found_inf) {
  std::vector<std::vector<Tensor>> tensor_lists{
      params.vec(), grads.vec(), exp_avgs.vec(), exp_avg_sqs.vec(), max_exp_avg_sqs.vec()};

  const std::string kernel_name =
      "fused_adam_amsgrad_" + scalarToMetalTypeString(params[0]) + "_" + scalarToMetalTypeString(state_steps[0]);

  multi_tensor_apply_for_fused_optimizer<5, 512>(kernel_name,
                                                 tensor_lists,
                                                 state_steps,
                                                 FusedAdamEncodingFunctor(),
                                                 lr,
                                                 beta1,
                                                 beta2,
                                                 weight_decay,
                                                 eps,
                                                 maximize);
}

} // namespace at::native::mps
