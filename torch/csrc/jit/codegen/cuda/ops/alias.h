#pragma once

#include <c10/macros/Export.h>

#include <torch/csrc/jit/codegen/cuda/ir_interface_nodes.h>
#include <torch/csrc/jit/codegen/cuda/type.h>

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

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
