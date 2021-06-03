#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>

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

struct ForwardDropoutResult {
  TensorView* output = nullptr;
  TensorView* mask = nullptr;
};

TORCH_CUDA_CU_API ForwardDropoutResult dropout(TensorView* x, Val* prob);

TORCH_CUDA_CU_API ForwardDropoutResult
dropout(TensorView* x, Val* prob, Val* scale);

TORCH_CUDA_CU_API TensorView* dropout_backward(
    TensorView* dy,
    TensorView* mask,
    Val* scale);

TORCH_CUDA_CU_API Val* softplus(Val* x, Val* beta, Val* threshold);

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
