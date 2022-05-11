#pragma once

#include <c10/macros/Export.h>

#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

void validateIr(Fusion* fusion);

//! Validate vectorization and collect information on vectorization
//! used in code generation as well as runtime validation.
void validateAndCollectVectorizeInfo(Fusion* fusion);

//! Validates partial split expressions. Partial split only uses an
//! inner subdomain specified by start and stop offsets, ignoring the
//! values outside the range. It's designed to be used with non-padded
//! shift, which introduces non-zero start and stop smaller than the
//! extent. This function makes sure all tensors have all values
//! calculated that are necessary for output values.
void validatePartialSplit(Fusion* fusion);

//! Validate data format and GPU arch compatibility of scheduled
//!  mma operators on the fusion.
void validateMma(Fusion* fusion);

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
