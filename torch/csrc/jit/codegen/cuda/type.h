#pragma once

#include <c10/core/ScalarType.h>
#include <c10/util/Exception.h>

#include <torch/csrc/WindowsTorchApiMacro.h>

#include <array>
#include <cstdint>
#include <iostream>
#include <string>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

enum class KernelIndexMode { INT32, INT64 };

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
};

// Manual - The user provides the Bool value. Predicate generation is bypassed.
// Inline corresponds with PredicateCompute::getInlinePredicate
// Unswitch corresponds with UnswitchPredicate::get
// Misaligned - PredicateCompute::getInlinePredicate + Misaligned flag
// Shift - ShiftPredicateInserter::getShiftPredicate
// Padding - ShiftPredicateInserter::getPaddingPredicate
// ReductionWrite - Same as Inline but without reduction axes
enum class PredicateType {
  Manual,
  Inline,
  Unswitch,
  Vectorize,
  Misaligned,
  Shift,
  Padding,
  ReductionWrite
};

enum class DataType { Double, Float, Half, Int, Int32, Bool, BFloat16, Null };

// Returns if the datatype is a floating point type
bool isFloatingPointType(DataType dtype);
// Returns if the datatype is an integer type
bool isIntegralType(DataType dtype);

enum class ExprType {
  Invalid,
  UnaryOp,
  BinaryOp,
  TernaryOp,
  ReductionOp,
  BroadcastOp,
  WelfordOp,
  TransposeOp,
  ShiftOp,
  GatherOp,
  Split,
  Merge,
};

enum class UnaryOpType {
  Abs,
  Acos,
  Address,
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
  Silu,
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
  Trunc,

  // Might be a bitwise operator or boolean operator.
  Not
};

// Primarily for Not, which could be Not a boolean, or a bitwise not.
bool alsoBooleanOperator(const UnaryOpType uopt);

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

  // Integer output ops. If changing modify isIntegerOp
  Mod,
  CeilDiv,
  Lshift,
  Rshift,

  // Logical Ops
  // Int operations, leave position of Mod as first logical op see
  // isLogicalOp(BinaryOpType bopt)
  Eq,
  GE,
  GT,
  LE,
  LT,
  NE,

  // Maybe bitwise or boolean op, leave position of and as first bool/int
  // op. These are ops that have different operators based on output type. See
  // is boolean op. These ops also don't work on floating point inputs.
  And,
  Or,
  Xor
};

// Return if output of operator should be a boolean
bool isIntegerOp(const BinaryOpType bopt);

// Return if output of operator should be a boolean
bool isLogicalOp(const BinaryOpType bopt);

// Operations that could be a bitwise operation or a boolean operation depending
// on input, for example bitwise_and is also used for boolean and in the jit
bool alsoBooleanOperator(const BinaryOpType bopt);

//! Operations that have tricky behaviors with all integer inputs
bool noFullIntegerSupport(const BinaryOpType bopt);

enum class TernaryOpType { Clamp, Threshold, Where };

enum class ParallelType {
  BIDz,
  BIDy,
  BIDx,
  TIDz,
  TIDy,
  TIDx,
  Vectorize,
  MisalignedVectorize,
  Unroll,
  Unswitch,
  Serial
};

static constexpr std::array<ParallelType, 6> kParallelTypeThreads = {
    ParallelType::BIDx,
    ParallelType::BIDy,
    ParallelType::BIDz,
    ParallelType::TIDx,
    ParallelType::TIDy,
    ParallelType::TIDz};

static constexpr std::array<ParallelType, 6> kParallelTypeBIDs = {
    ParallelType::BIDx,
    ParallelType::BIDy,
    ParallelType::BIDz};

static constexpr std::array<ParallelType, 6> kParallelTypeTIDs = {
    ParallelType::BIDx,
    ParallelType::BIDy,
    ParallelType::BIDz};

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
  BroadcastWithoutStride,
  Gather
};

enum class SwizzleType { NoSwizzle, Transpose };

// Returns if function needs an f suffix on the operator when operating on a
// float value i.e. sin->sinf
bool needFloatSuffix(UnaryOpType t);
bool needFloatSuffix(BinaryOpType t);

ValType promote_type(const ValType& t1, const ValType& t2);
DataType promote_type(const DataType& t1, const DataType& t2);

// If type cannot be found (i.e. codegen does not support provided type) returns
// DataType::Null
TORCH_CUDA_CU_API DataType aten_to_data_type(const at::ScalarType& scalar_type);
TORCH_CUDA_CU_API at::ScalarType data_type_to_aten(const DataType& data_type);

TORCH_CUDA_CU_API std::ostream& operator<<(std::ostream&, const ValType);
TORCH_CUDA_CU_API std::ostream& operator<<(std::ostream&, const DataType);
TORCH_CUDA_CU_API std::ostream& operator<<(std::ostream&, const ExprType);
TORCH_CUDA_CU_API std::ostream& operator<<(std::ostream&, const UnaryOpType);
TORCH_CUDA_CU_API std::ostream& operator<<(std::ostream&, const BinaryOpType);
TORCH_CUDA_CU_API std::ostream& operator<<(std::ostream&, const TernaryOpType);
TORCH_CUDA_CU_API std::ostream& operator<<(std::ostream&, const ParallelType);
TORCH_CUDA_CU_API std::ostream& operator<<(std::ostream&, const MemoryType);
TORCH_CUDA_CU_API std::ostream& operator<<(std::ostream&, const IterType);

std::string stringifyBooleanOp(const UnaryOpType);
std::string stringifyBooleanOp(const BinaryOpType);

std::string stringifyThreadSize(const ParallelType);
std::string stringifyThread(const ParallelType);
std::string typePrefix(const DataType);

// TODO: ThreadDim should be BlockDim and BlockDim should be GridDim
TORCH_CUDA_CU_API bool isParallelTypeThreadDim(ParallelType);
TORCH_CUDA_CU_API bool isParallelTypeBlockDim(ParallelType);
TORCH_CUDA_CU_API bool isParallelTypeThread(ParallelType);

TORCH_CUDA_CU_API bool isParallelTypeVectorize(ParallelType);

TORCH_CUDA_CU_API c10::optional<std::string> inline_op_str(const UnaryOpType);
TORCH_CUDA_CU_API c10::optional<std::string> inline_op_str(const BinaryOpType);
TORCH_CUDA_CU_API c10::optional<std::string> integer_op_str(const BinaryOpType);

TORCH_CUDA_CU_API c10::optional<std::string> cast_func_str(
    const std::pair<DataType, DataType>&);

TORCH_CUDA_CU_API size_t dataTypeSize(DataType type);

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
