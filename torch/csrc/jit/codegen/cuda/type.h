#pragma once

#include <c10/util/Exception.h>
#include <c10/core/ScalarType.h>
#include <torch/csrc/WindowsTorchApiMacro.h>

#include <cstdint>
#include <iostream>
#include <string>

namespace torch {
namespace jit {
namespace fuser {

//Order of strength
enum class TORCH_API ValType {
  TensorIndex,
  TensorDomain,
  IterDomain,
  TensorView,
  Scalar
};

enum class TORCH_API DataType {
  Float,
  Int,
  Null
};

enum class TORCH_API ExprType {
    UnaryOp
  , BinaryOp
  , ForLoop
  , IfThenElse
  , Split
  , Merge
  , Reorder
// , Swap
// , Index
};

enum class TORCH_API UnaryOpType {
    Neg
  , Cast
};

enum class TORCH_API BinaryOpType {
    Add
  , Sub
  , Mul
  , Div
  //Int operations, leave position oif Mod we depend on its location of first Int op
  , Mod
  , LT
  , CeilDiv
};

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


ValType promote_type(const ValType& t1, const ValType& t2);
DataType promote_type(const DataType& t1, const DataType& t2);
bool is_cast_legal(const DataType& t1, const DataType& t2);

DataType aten_to_data_type(const at::ScalarType& scalar_type);

TORCH_API std::ostream& operator<<(std::ostream&, const ValType);
TORCH_API std::ostream& operator<<(std::ostream&, const DataType);
TORCH_API std::ostream& operator<<(std::ostream&, const ExprType);
TORCH_API std::ostream& operator<<(std::ostream&, const UnaryOpType);
TORCH_API std::ostream& operator<<(std::ostream&, const BinaryOpType);
TORCH_API std::ostream& operator<<(std::ostream&, const ParallelType);

TORCH_API c10::optional<std::string> inline_op_str(const UnaryOpType);
TORCH_API c10::optional<std::string> inline_op_str(const BinaryOpType);

} // namespace fuser
} // namespace jit
} // namespace torch
