#pragma once

#include <ATen/core/ivalue.h>

#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/scheduler/pointwise_heuristic.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

class SchedulerRuntimeInfo;
class HeuristicSummary;

TORCH_CUDA_CU_API std::shared_ptr<PointwiseParams> getPointwiseHeuristics(
    Fusion* fusion,
    const at::ArrayRef<c10::IValue>& runtime_inputs,
    HeuristicSummary* data_cache = nullptr);

TORCH_CUDA_CU_API std::shared_ptr<PointwiseParams> getPointwiseHeuristics(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicSummary* data_cache = nullptr);

TORCH_CUDA_CU_API void schedulePointwise(
    Fusion* fusion,
    const PointwiseParams& params);

TORCH_CUDA_CU_API LaunchParams schedulePointwise(
    Fusion* fusion,
    const at::ArrayRef<c10::IValue>& runtime_inputs);

//! Utility for canSchedule interface to check if this fusion has
//!  a fully broadcasted reference tensor, which is necessary for
//!  the pointwise scheduler.
bool hasReferenceTensorView(Fusion* fusion);

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
