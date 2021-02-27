#include <ATen/core/ivalue.h>

#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/scheduler/reduction_heuristic.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

class ExpressionEvaluator;

TORCH_CUDA_CU_API c10::optional<ReductionParams> getReductionHeuristics(
    Fusion* fusion,
    const at::ArrayRef<c10::IValue>& fusion_inputs,
    TensorView* red_tv);

TORCH_CUDA_CU_API c10::optional<ReductionParams> getReductionHeuristics(
    Fusion* fusion,
    ExpressionEvaluator& evaluator,
    TensorView* red_tv);

TORCH_CUDA_CU_API void scheduleReduction(
    Fusion* fusion,
    const ReductionParams& rparams,
    TensorView* red_tv,
    const std::vector<TensorView*>& outs_of_red);
} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
