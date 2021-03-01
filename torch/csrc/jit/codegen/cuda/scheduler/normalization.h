#include <ATen/core/ivalue.h>

#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/scheduler/reduction_heuristic.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {
class ExpressionEvaluator;

TORCH_CUDA_CU_API c10::optional<ReductionParams> getNormalizationHeuristics(
    Fusion* fusion,
    const at::ArrayRef<c10::IValue>& fusion_inputs,
    const std::vector<TensorView*>& reduction_tv);

TORCH_CUDA_CU_API c10::optional<ReductionParams> getNormalizationHeuristics(
    Fusion* fusion,
    ExpressionEvaluator& evaluator,
    const std::vector<TensorView*>& reduction_tv);

TORCH_CUDA_CU_API void scheduleNormalization(
    Fusion* fusion,
    const ReductionParams& rparams,
    const std::vector<TensorView*>& reduction_tv,
    std::vector<TensorView*>& other_tv);

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
