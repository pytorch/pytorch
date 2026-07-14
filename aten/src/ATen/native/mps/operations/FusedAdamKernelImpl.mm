#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/mps/operations/FusedAdamKernelImpl.h>

#include <ATen/Dispatch.h>
#include <ATen/native/ForeachUtils.h>
#include <ATen/native/mps/operations/MultiTensorApply.h>
#include <vector>

namespace at::native::mps {

void _fused_adam_mps_impl_(TensorList params,
                           TensorList grads,
                           TensorList exp_avgs,
                           TensorList exp_avg_sqs,
                           TensorList state_steps,
                           const double lr,
                           const double beta1,
                           const double beta2,
                           const double weight_decay,
                           const double eps,
                           const bool maximize,
                           const std::optional<Tensor>& grad_scale,
                           const std::optional<Tensor>& found_inf) {
  std::vector<std::vector<Tensor>> tensor_lists{params.vec(), grads.vec(), exp_avgs.vec(), exp_avg_sqs.vec()};

  const auto kernel_name =
      "fused_adam_" + scalarToMetalTypeString(params[0]) + "_" + scalarToMetalTypeString(state_steps[0]);

  multi_tensor_apply_for_fused_optimizer<4, 512>(kernel_name,
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

void _fused_adam_mps_impl_(TensorList params,
                           TensorList grads,
                           TensorList exp_avgs,
                           TensorList exp_avg_sqs,
                           TensorList state_steps,
                           const Tensor& lr,
                           const double beta1,
                           const double beta2,
                           const double weight_decay,
                           const double eps,
                           const bool maximize,
                           const std::optional<Tensor>& grad_scale,
                           const std::optional<Tensor>& found_inf) {
  std::vector<std::vector<Tensor>> tensor_lists{params.vec(), grads.vec(), exp_avgs.vec(), exp_avg_sqs.vec()};

  const auto kernel_name =
      "fused_adam_" + scalarToMetalTypeString(params[0]) + "_" + scalarToMetalTypeString(state_steps[0]);

  multi_tensor_apply_for_fused_optimizer<4, 512>(kernel_name,
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

#ifndef PYTORCH_JIT_COMPILE_SHADERS
static auto& lib = MetalShaderLibrary::getBundledLibrary();
#else
#include <ATen/native/mps/FusedOptimizerOps_metallib.h>
#endif

std::pair<id<MTLComputePipelineState>, id<MTLFunction>> getFusedAdamCPLState(const std::string& fname) {
  return {lib.getPipelineStateForFunc(fname), lib.getMTLFunction(fname)};
}

} // namespace at::native::mps
