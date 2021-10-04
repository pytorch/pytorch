#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>

#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

void validateIr(Fusion* fusion);

void validateVectorize(Fusion* fusion);

//! Validates all tensors are consistently parallelized. Basically,
//! when a producer axis is threaded, either with threadIdx or
//! blockIdx, there must be a mapped consumer axis with the
//! same ParallelType with some exceptions.
//!
//! This function assumes Loop and Parallel ComputeAtMaps are already
//! built as they are used to validate consistency.
void validateParallelize(Fusion* fusion);

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
