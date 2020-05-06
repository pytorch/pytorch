#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>

#include <torch/csrc/jit/codegen/cuda/ir_interface_nodes.h>
#include <torch/csrc/jit/codegen/cuda/type.h>

struct Val;

/*
 * The operations defined in this header is intended as user facing functions.
 * Generally users should not directly instantiate temporary TensorViews they
 * should instead use the functions below which will automatically create IR
 * nodes, and return a resulting TensorView of correctly tracked shapes.
 */

namespace torch {
namespace jit {
namespace fuser {

// Promotion logic between two values, returns a new val from resulting type
// promotion.
TORCH_CUDA_API Val* promoteNew(Val* v1, Val* v2);

// Insertion of casting op to dtype, returns new resulting val
TORCH_CUDA_API Val* castOp(DataType dtype, Val* v1);

// Perform unary op type and return the output
TORCH_CUDA_API Val* unaryOp(UnaryOpType type, Val* v1);

// Perform binary op type on v1 and v2 and return a type promoted output.
// Mod, CeilDiv, and LT are considered Int only output operations for now.
TORCH_CUDA_API Val* binaryOp(BinaryOpType type, Val* v1, Val* v2);

TORCH_CUDA_API Val* add(Val* v1, Val* v2);
TORCH_CUDA_API Val* sub(Val* v1, Val* v2);
TORCH_CUDA_API Val* mul(Val* v1, Val* v2);
TORCH_CUDA_API Val* div(Val* v1, Val* v2);
TORCH_CUDA_API Val* mod(Val* v1, Val* v2);
TORCH_CUDA_API Val* lt(Val* v1, Val* v2);
TORCH_CUDA_API Val* ceilDiv(Val* v1, Val* v2);
TORCH_CUDA_API Val* andOp(Val* v1, Val* v2);

} // namespace fuser
} // namespace jit
} // namespace torch
