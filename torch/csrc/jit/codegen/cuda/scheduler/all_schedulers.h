#pragma once
#include <torch/csrc/jit/codegen/cuda/scheduler/normalization.h>
#include <torch/csrc/jit/codegen/cuda/scheduler/pointwise.h>
#include <torch/csrc/jit/codegen/cuda/scheduler/reduction.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

enum class TORCH_CUDA_CU_API ScheduleHeuristic {
  PointWise,
  Reduction,
  Normalization
};

}
} // namespace fuser
} // namespace jit
} // namespace torch
