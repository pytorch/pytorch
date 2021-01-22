#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>

#include <torch/csrc/jit/codegen/cuda/ir_interface_nodes.h>
#include <torch/csrc/jit/codegen/cuda/type.h>

class Val;

/*
 * The operations defined in this header is intended as user facing functions.
 * Generally users should not directly instantiate temporary TensorViews they
 * should instead use the functions below which will automatically create IR
 * nodes, and return a resulting TensorView of correctly tracked shapes.
 */

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

// Insertion of casting op to dtype, returns new resulting val
TORCH_CUDA_CU_API Val* castOp(DataType dtype, Val* v1);
TORCH_CUDA_CU_API TensorView* castOp(DataType dtype, TensorView* v1);

// Perform unary op type and return the output
TORCH_CUDA_CU_API Val* unaryOp(UnaryOpType type, Val* v1);
TORCH_CUDA_CU_API TensorView* unaryOp(UnaryOpType type, TensorView* v1);

// Perform binary op type on v1 and v2 and return a type promoted output.
// Mod, CeilDiv, and LT are considered Int only output operations for now.
TORCH_CUDA_CU_API Val* binaryOp(BinaryOpType type, Val* v1, Val* v2);
TORCH_CUDA_CU_API TensorView* binaryOp(
    BinaryOpType type,
    TensorView* v1,
    Val* v2);
TORCH_CUDA_CU_API TensorView* binaryOp(
    BinaryOpType type,
    Val* v1,
    TensorView* v2);
TORCH_CUDA_CU_API TensorView* binaryOp(
    BinaryOpType type,
    TensorView* v1,
    TensorView* v2);

// Perform a reduction operation on v1, initial value for reduction is init,
// reduces across axes, and reduction operation defined by BinaryOp.
TORCH_CUDA_CU_API TensorView* reductionOp(
    BinaryOpType reduction_op_type,
    const std::vector<int>& axes,
    Val* init,
    TensorView* v1);

// UNARY OPERATIONS
TORCH_CUDA_CU_API Val* neg(Val* v);
TORCH_CUDA_CU_API TensorView* neg(TensorView* v);

// Broadcasts v1 based on bool vector. Size of broadcast bool vector should be
// the number of dims desired in the broadcasted tensor. This vector should be
// true if output dim should be a broadcasted dim, and false if it is not a
// broadcasted dim. Number of false entires must match the number of input dims.
TORCH_CUDA_CU_API TensorView* broadcast(
    TensorView* inp,
    const std::vector<bool>& is_broadcast_dim);

// BINARY OPERATIONS
// add
TORCH_CUDA_CU_API Val* add(Val* v1, Val* v2);
TORCH_CUDA_CU_API TensorView* add(TensorView* v1, Val* v2);
TORCH_CUDA_CU_API TensorView* add(Val* v1, TensorView* v2);
TORCH_CUDA_CU_API TensorView* add(TensorView* v1, TensorView* v2);
// sub
TORCH_CUDA_CU_API Val* sub(Val* v1, Val* v2);
TORCH_CUDA_CU_API TensorView* sub(TensorView* v1, Val* v2);
TORCH_CUDA_CU_API TensorView* sub(Val* v1, TensorView* v2);
TORCH_CUDA_CU_API TensorView* sub(TensorView* v1, TensorView* v2);
// mul
TORCH_CUDA_CU_API Val* mul(Val* v1, Val* v2);
TORCH_CUDA_CU_API TensorView* mul(TensorView* v1, Val* v2);
TORCH_CUDA_CU_API TensorView* mul(Val* v1, TensorView* v2);
TORCH_CUDA_CU_API TensorView* mul(TensorView* v1, TensorView* v2);
// div
TORCH_CUDA_CU_API Val* div(Val* v1, Val* v2);
TORCH_CUDA_CU_API TensorView* div(TensorView* v1, Val* v2);
TORCH_CUDA_CU_API TensorView* div(Val* v1, TensorView* v2);
TORCH_CUDA_CU_API TensorView* div(TensorView* v1, TensorView* v2);
// mod
TORCH_CUDA_CU_API Val* mod(Val* v1, Val* v2);
TORCH_CUDA_CU_API TensorView* mod(TensorView* v1, Val* v2);
TORCH_CUDA_CU_API TensorView* mod(Val* v1, TensorView* v2);
TORCH_CUDA_CU_API TensorView* mod(TensorView* v1, TensorView* v2);
// lt
TORCH_CUDA_CU_API Val* lt(Val* v1, Val* v2);
TORCH_CUDA_CU_API TensorView* lt(TensorView* v1, Val* v2);
TORCH_CUDA_CU_API TensorView* lt(Val* v1, TensorView* v2);
TORCH_CUDA_CU_API TensorView* lt(TensorView* v1, TensorView* v2);
// eq
TORCH_CUDA_CU_API Val* eq(Val* v1, Val* v2);
TORCH_CUDA_CU_API TensorView* eq(TensorView* v1, Val* v2);
TORCH_CUDA_CU_API TensorView* eq(Val* v1, TensorView* v2);
TORCH_CUDA_CU_API TensorView* eq(TensorView* v1, TensorView* v2);
// ceilDiv
TORCH_CUDA_CU_API Val* ceilDiv(Val* v1, Val* v2);
TORCH_CUDA_CU_API TensorView* ceilDiv(TensorView* v1, Val* v2);
TORCH_CUDA_CU_API TensorView* ceilDiv(Val* v1, TensorView* v2);
TORCH_CUDA_CU_API TensorView* ceilDiv(TensorView* v1, TensorView* v2);
// andOp
TORCH_CUDA_CU_API Val* andOp(Val* v1, Val* v2);
TORCH_CUDA_CU_API TensorView* andOp(TensorView* v1, Val* v2);
TORCH_CUDA_CU_API TensorView* andOp(Val* v1, TensorView* v2);
TORCH_CUDA_CU_API TensorView* andOp(TensorView* v1, TensorView* v2);

// REDUCTION OPERATIONS
TORCH_CUDA_CU_API TensorView* sum(
    TensorView* v1,
    const std::vector<int>& reduction_axes);

// COMPOUND OPERATIONS
// add_alpha
TORCH_CUDA_CU_API Val* add_alpha(Val* v1, Val* v2, Val* s);
TORCH_CUDA_CU_API TensorView* add_alpha(TensorView* v1, Val* v2, Val* s);
TORCH_CUDA_CU_API TensorView* add_alpha(Val* v1, TensorView* v2, Val* s);
TORCH_CUDA_CU_API TensorView* add_alpha(TensorView* v1, TensorView* v2, Val* s);
// sub_alpha
TORCH_CUDA_CU_API Val* sub_alpha(Val* v1, Val* v2, Val* s);
TORCH_CUDA_CU_API TensorView* sub_alpha(TensorView* v1, Val* v2, Val* s);
TORCH_CUDA_CU_API TensorView* sub_alpha(Val* v1, TensorView* v2, Val* s);
TORCH_CUDA_CU_API TensorView* sub_alpha(TensorView* v1, TensorView* v2, Val* s);
// lerp
TORCH_CUDA_CU_API Val* lerp(Val* start, Val* end, Val* weight);
TORCH_CUDA_CU_API TensorView* lerp(TensorView* start, Val* end, Val* weight);
TORCH_CUDA_CU_API TensorView* lerp(Val* start, TensorView* end, Val* weight);
TORCH_CUDA_CU_API TensorView* lerp(Val* start, Val* end, TensorView* weight);
TORCH_CUDA_CU_API TensorView* lerp(
    TensorView* start,
    TensorView* end,
    Val* weight);
TORCH_CUDA_CU_API TensorView* lerp(
    TensorView* start,
    Val* end,
    TensorView* weight);
TORCH_CUDA_CU_API TensorView* lerp(
    Val* start,
    TensorView* end,
    TensorView* weight);
TORCH_CUDA_CU_API TensorView* lerp(
    TensorView* start,
    TensorView* end,
    TensorView* weight);
// addcmul
TORCH_CUDA_CU_API Val* addcmul(Val* v1, Val* v2, Val* v3, Val* s);
TORCH_CUDA_CU_API TensorView* addcmul(TensorView* v1, Val* v2, Val* v3, Val* s);
TORCH_CUDA_CU_API TensorView* addcmul(Val* v1, TensorView* v2, Val* v3, Val* s);
TORCH_CUDA_CU_API TensorView* addcmul(Val* v1, Val* v2, TensorView* v3, Val* s);
TORCH_CUDA_CU_API TensorView* addcmul(
    TensorView* v1,
    TensorView* v2,
    Val* v3,
    Val* s);
TORCH_CUDA_CU_API TensorView* addcmul(
    TensorView* v1,
    Val* v2,
    TensorView* v3,
    Val* s);
TORCH_CUDA_CU_API TensorView* addcmul(
    Val* v1,
    TensorView* v2,
    TensorView* v3,
    Val* s);
TORCH_CUDA_CU_API TensorView* addcmul(
    TensorView* v1,
    TensorView* v2,
    TensorView* v3,
    Val* s);

// TERNARY OPERATIONS
// where
TORCH_CUDA_CU_API Val* where(Val* c, Val* v1, Val* v2);
TORCH_CUDA_CU_API TensorView* where(TensorView* c, Val* v1, Val* v2);
TORCH_CUDA_CU_API TensorView* where(Val* c, TensorView* v1, Val* v2);
TORCH_CUDA_CU_API TensorView* where(Val* c, Val* v1, TensorView* v2);
TORCH_CUDA_CU_API TensorView* where(TensorView* c, TensorView* v1, Val* v2);
TORCH_CUDA_CU_API TensorView* where(TensorView* c, Val* v1, TensorView* v2);
TORCH_CUDA_CU_API TensorView* where(Val* c, TensorView* v1, TensorView* v2);
TORCH_CUDA_CU_API TensorView* where(
    TensorView* c,
    TensorView* v1,
    TensorView* v2);
// threshold
TORCH_CUDA_CU_API Val* threshold(Val* in, Val* thresh, Val* value);
TORCH_CUDA_CU_API TensorView* threshold(
    TensorView* in,
    Val* thresh,
    Val* value);
// clamp
TORCH_CUDA_CU_API Val* clamp(Val* in, Val* min_val, Val* max_val);
TORCH_CUDA_CU_API TensorView* clamp(TensorView* in, Val* min_val, Val* max_val);

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
