#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/mps/operations/FusedAdamAmsgradKernelImpl.h>

#include <ATen/Dispatch.h>
#include <ATen/native/ForeachUtils.h>
#include <ATen/native/mps/operations/FusedOptimizerOps.h>
#include <ATen/native/mps/operations/MultiTensorApply.h>
#include <vector>

namespace at::native {
namespace mps {

void _fused_adam_amsgrad_mps_impl_(at::TensorList params,
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
                                   const c10::optional<at::Tensor>& found_inf) {
  std::vector<std::vector<at::Tensor>> tensor_lists{
      params.vec(), grads.vec(), exp_avgs.vec(), exp_avg_sqs.vec(), max_exp_avg_sqs.vec()};

  const std::string kernel_name = "fused_adam_amsgrad_" + scalarToMetalTypeString(params[0].scalar_type()) + "_" +
      scalarToMetalTypeString(state_steps[0].scalar_type());

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
} // namespace mps
} // namespace at::native