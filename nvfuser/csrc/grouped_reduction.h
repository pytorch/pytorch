#pragma once

#include <ir_all_nodes.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

//! Horizontally fuse multiple reductions.
//!
//! Given a list of tensors produced by ReductionOp, create a new
//! GroupedReductionOp expression that takes the input tensors of the
//! original reductions and produces the given tensors, replacing
//! their defining expressions.
//!
//! GroupedReductionOp works just like ReductionOp with a potential
//! benefit of aggregating synchronizations across individual
//! reductions. See the reduction::gridReduce2 runtime function for a
//! two-input version of grid reduction.
//!
//! The grouped reductions must follow several constraints, which
//! include:
//! - There must not exist any data dependency between individual
//!   reductions.
//! - All reduction output tensors must have the same number of
//!   dimensions, the same transformations and the same axes to
//!   reduce.
//!
//! Note that Welford is not allowed yet, though it should be
//! technically straightforward to support horizontal fusions of
//! welford ops. Unclear how common it would be in practice, though.
//!
//! \param reduction_outputs Tensors produced by ReductionOp
TORCH_CUDA_CU_API void groupReductions(
    const std::vector<TensorView*>& reduction_outputs);

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
