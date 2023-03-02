#pragma once
#include <scheduler/normalization.h>
#include <scheduler/pointwise.h>
#include <scheduler/reduction.h>
#include <scheduler/transpose.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

enum class TORCH_CUDA_CU_API ScheduleHeuristic {
  None,
  NoOp,
  PointWise,
  Reduction,
  Persistent,
  Transpose
};
}
} // namespace fuser
} // namespace jit
} // namespace torch
