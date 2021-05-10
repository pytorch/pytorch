#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>

#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/transform_iter.h>

#include <algorithm>
#include <vector>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

// TODO: Only replay dispatch is really borrowed from TransformIter, we should
// reevaluate the reuse of dispatch for classes that inherit TransformIter.
class TORCH_CUDA_CU_API TransformRFactor {
 public:
  // Create a copy of td, change its history by presrving axes so they appear in
  // the root domain
  static TensorDomain* runReplay(TensorDomain*, std::vector<int> axes);

  static TensorDomain* runReplay2(TensorDomain*, std::vector<int> axes);
};

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
