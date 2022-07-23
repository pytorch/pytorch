#pragma once

#include <ATen/core/ivalue.h>

#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/mma_type.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

//! Prototype auto scheduling function.
//!  Currently only support a pure matmul with no
//!   fused prolog or epilog.
//!
//! TODO:
//!   - will support a range of fusions in a follow up
//!   - will formalize scheduling decisions into
//! matmul params data structure.
TORCH_CUDA_CU_API void scheduleMatmul(
    TensorView* c_tv,
    TensorView* a_tv,
    TensorView* b_tv,
    MmaBuilder& mma_builder,
    MatMulTileOptions& gemm_tile);

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
