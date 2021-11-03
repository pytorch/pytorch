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

struct LstmResult {
  TensorView* cell = nullptr;
  TensorView* hidden = nullptr;
};

TORCH_CUDA_CU_API LstmResult lstm(
    TensorView* prev_cell,
    TensorView* in_x,
    TensorView* forget_x,
    TensorView* cell_x,
    TensorView* out_x);

TORCH_CUDA_CU_API Val* fast_gelu(Val* x);
TORCH_CUDA_CU_API Val* fast_gelu_backward(Val* dy, Val* x);
TORCH_CUDA_CU_API Val* gelu_backward(Val* dy, Val* x);

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
