#pragma once

#include <c10/macros/Export.h>

#include <torch/csrc/jit/codegen/cuda/compute_at_map.h>
#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

// Looks through all transformations assocaited with view, or enforced divisible
// vectorization splits and gathers all splits that provably don't have a
// remainder, therefore the extents of the associated IterDomains do not require
// a ceilDiv expressions.
TORCH_CUDA_CU_API std::unordered_set<Split*> getAllDivisibleSplits(
    Fusion* fusion);

// Same as above but will use provided ComputeAtMap instead of building its own.
TORCH_CUDA_CU_API std::unordered_set<Split*> getAllDivisibleSplits(
    Fusion* fusion,
    const ComputeAtMap* ca_map);

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
