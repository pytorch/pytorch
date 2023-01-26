#pragma once

#include <c10/macros/Export.h>

#include <ir_interface_nodes.h>
#include <type.h>

//
// The operations defined in this header is intended as user facing functions.
// The user will provide the necessary input TensorViews and the function will
// create the correct intermediate nodes and return the output TensorViews.
//

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

TORCH_CUDA_CU_API TensorView* view(TensorView* x, DataType dtype);

TORCH_CUDA_CU_API TensorView* view(
    TensorView* x,
    const std::vector<int64_t>& original_sizes,
    const std::vector<int64_t>& new_sizes);

TORCH_CUDA_CU_API TensorView* flatten(
    TensorView* x,
    int64_t start_dim = 0,
    int64_t end_dim = -1);

TORCH_CUDA_CU_API TensorView* squeeze(
    TensorView* x,
    const std::vector<int64_t>& sizes);

TORCH_CUDA_CU_API TensorView* squeeze(
    TensorView* x,
    const std::vector<int64_t>& sizes,
    int dim);

TORCH_CUDA_CU_API TensorView* unsqueeze(TensorView* x, int dim);

//! Permute a tensor as specified by axis mappings.
//!
//! The transposition mapping is specified with a list of pairs from
//! new to old positions. Positions are relative to the noReduction
//! domain.
//!
//! \param inp Tensor to transpose
//! \param new2old vector mapping from new to old positions.
TORCH_CUDA_CU_API TensorView* permute(
    TensorView* x,
    const std::vector<int64_t>& new2old);

//! Transpose a tensor by swapping the two dimensions.
TORCH_CUDA_CU_API TensorView* transpose(
    TensorView* x,
    int64_t dim0,
    int64_t dim1);

//! Transpose a 2D tensor.
TORCH_CUDA_CU_API TensorView* transpose(TensorView* x);

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
