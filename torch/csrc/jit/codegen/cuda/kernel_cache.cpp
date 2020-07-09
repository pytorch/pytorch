#include <torch/csrc/jit/codegen/cuda/kernel_cache.h>
#include <torch/csrc/jit/codegen/cuda/expr_evaluator.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>

/*
 */

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

at::optional<CudaKernel*> CudaKernelCache::getKernelPtr(
    const at::ArrayRef<c10::IValue> inputs,
    const std::vector<int64_t>& broadcasted_shape) {
  for (auto& cuda_kernel : kernels_) {
    // bound input sizes
    Fusion* fusion = cuda_kernel.fusion();
    FusionGuard fg(fusion);
    EvaluationContext eval_context(fusion);
    for (int i = 0; i < (int)inputs.size(); i++) {
      if (inputs[i].isTensor()) {
        ExtractSizeStride ess(inputs[i].toTensor(), broadcasted_shape);
        const int nDims = ess.sizes.size();
        TensorView* tv = fusion->inputs()[i]->as<TensorView>();
        for (int j = 0; j < nDims; j++) {
          eval_context.bind(tv->getRootDomain()[j]->extent(), ess.sizes[j]);
        }
      }
    }

    const auto val = ExpressionEvaluator::evaluate(
        fusion->getLaunchConfig(LaunchConfigType::Compatible), &eval_context);
    TORCH_INTERNAL_ASSERT(
        val.has_value(), "scheduler didn't bind launch configs properly");
    if (val.value()) {
      return &cuda_kernel;
    }
  }
  return at::nullopt;
}

CudaKernel* CudaKernelCache::allocateKernelInCache(
    const at::ArrayRef<c10::IValue> inputs) {
  kernels_.emplace_back();
  return &kernels_.back();
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
