#pragma once

#include <c10/util/Exception.h>
#include <torch/csrc/WindowsTorchApiMacro.h>

#include <cstdint>
#include <iostream>
#include <string>

namespace torch {
namespace jit {
namespace fuser {

//Order of strength
enum class TORCH_API ValType {
  TensorDomain,
  TensorView,
  IterDomain,
  Tensor,
  Scalar
};

enum class TORCH_API DataType {
  Float,
  Int,
  Null
};

enum class TORCH_API ExprType {
  Add
  // Sub,
  // Mul,
  // Div,
  // Mod,
  // Loop,
  // Swap,
  // Merge,
  // Split,
  // Index,
};

ValType promote_type(const ValType& t1, const ValType& t2);
DataType promote_type(const DataType& t1, const DataType& t2);

TORCH_API std::string stringify(const ValType);
TORCH_API std::string stringify(const DataType);
TORCH_API std::string stringify(const ExprType);

TORCH_API std::ostream& operator<<(std::ostream& out, const ValType valtype);
TORCH_API std::ostream& operator<<(std::ostream& out, const DataType datatype);
TORCH_API std::ostream& operator<<(std::ostream& out, const ExprType exprtype);

} // namespace fuser
} // namespace jit
} // namespace torch
