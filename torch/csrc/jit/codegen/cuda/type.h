#pragma once

#include <c10/core/ScalarType.h>
#include <c10/util/Exception.h>

#include <torch/csrc/WindowsTorchApiMacro.h>

#include <cstdint>
#include <iostream>
#include <string>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

// https://stackoverflow.com/questions/18837857/cant-use-enum-class-as-unordered-map-key
struct TypeHash {
  template <typename T>
  std::size_t operator()(T t) const {
    return static_cast<std::size_t>(t);
  }
};

// Order of strength
enum class ValType {
  TensorDomain,
  IterDomain,
  TensorView,
  Scalar,
  NamedScalar,

  // Temporary: Kernel IR nodes
  TensorIndex,
  KirNamedScalar,
  KirScalar,
  KirTensorDomain,
  KirIterDomain,
  KirTensorView,
};

enum class DataType { Bool, Float, Half, Int, Null };

enum class ExprType {
  Invalid,
  UnaryOp,
  BinaryOp,
  TernaryOp,
  ReductionOp,
  BroadcastOp,
  Split,
  Merge,

  // Temporary: Kernel IR nodes
  GridReduction,
  ForLoop,
  IfThenElse,
  Allocate,
  Sync,
  KirUnaryOp,
  KirBinaryOp,
  KirTernaryOp,
  KirReductionOp,
  KirBroadcastOp,
};

enum class UnaryOpType {
  Abs,
  Acos,
  Asin,
  Atan,
  Atanh,
  Cast,
  Ceil,
  Cos,
  Cosh,
  Exp,
  Expm1,
  Erf,
  Erfc,
  Floor,
  Frac,
  Gelu,
  Lgamma,
  Log,
  Log10,
  Log1p,
  Log2,
  Neg,
  RandLike,
  Reciprocal,
  Relu,
  Rsqrt,
  Round,
  Set,
  Sigmoid,
  Sin,
  Sinh,
  Sqrt,
  Tan,
  Tanh,
  Trunc
};

// TODO: Order of this list is important as it affects type promotion. it's not
// in the right order now.
enum class BinaryOpType {
  // Math Ops
  Add,
  Atan2,
  Div,
  Fmod,
  Max,
  Min,
  Mul,
  Pow,
  Remainder,
  Sub,
  // TypeAs,

  // Logical Ops
  // Int operations, leave position of Mod we depend on its location of first
  Mod,
  CeilDiv,
  And,
  Eq,
  GE,
  GT,
  LE,
  LT,
  NE
};

enum class TernaryOpType { Clamp, Threshold, Where };

enum class ParallelType {
  BIDz,
  BIDy,
  BIDx,
  TIDz,
  TIDy,
  TIDx,
  Vectorize,
  Unroll,
  Serial
};

enum class MemoryType { Local, Shared, Global };

// sometimes broadcasted tensors may be inputed in the kernel with an explicit 1
// size. If that size is there, we need to account that there's also a stride
// there, even if the stride = 0. If we don't account for that stride when
// accessing a tensor like: [b2{1}, i0, i1] we would linearize the access like:
// [i0*stride[0] + i1*stride[1]] when it should be: [i0*stride[1] +
// i1*stride[2]]. Broadcasts that translate to a physical memory dim we consider
// "with stride", Broadcasts only through our broadcast op we consider "without
// stride"
enum class IterType {
  Iteration,
  Reduction,
  BroadcastWithStride,
  BroadcastWithoutStride
};

ValType promote_type(const ValType& t1, const ValType& t2);
DataType promote_type(const DataType& t1, const DataType& t2);
bool is_logical_op(const BinaryOpType& bot);

DataType aten_to_data_type(const at::ScalarType& scalar_type);
at::ScalarType data_type_to_aten(const DataType& data_type);

TORCH_CUDA_CU_API std::ostream& operator<<(std::ostream&, const ValType);
TORCH_CUDA_CU_API std::ostream& operator<<(std::ostream&, const DataType);
TORCH_CUDA_CU_API std::ostream& operator<<(std::ostream&, const ExprType);
TORCH_CUDA_CU_API std::ostream& operator<<(std::ostream&, const UnaryOpType);
TORCH_CUDA_CU_API std::ostream& operator<<(std::ostream&, const BinaryOpType);
TORCH_CUDA_CU_API std::ostream& operator<<(std::ostream&, const TernaryOpType);
TORCH_CUDA_CU_API std::ostream& operator<<(std::ostream&, const ParallelType);
TORCH_CUDA_CU_API std::ostream& operator<<(std::ostream&, const MemoryType);
TORCH_CUDA_CU_API std::ostream& operator<<(std::ostream&, const IterType);

std::string stringifyThreadSize(const ParallelType);
std::string stringifyThread(const ParallelType);

TORCH_CUDA_CU_API c10::optional<std::string> inline_op_str(const UnaryOpType);
TORCH_CUDA_CU_API c10::optional<std::string> inline_op_str(const BinaryOpType);

TORCH_CUDA_CU_API c10::optional<std::string> cast_func_str(
    const std::pair<DataType, DataType>&);

size_t dataTypeSize(DataType type);

enum class LaunchConfigType {
  Compatible,
  SharedMemory,
  BIDz,
  BIDy,
  BIDx,
  TIDz,
  TIDy,
  TIDx
};

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
