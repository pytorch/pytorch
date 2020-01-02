#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>

#include <cstdint>
#include <iostream>
#include <string>

namespace torch {
namespace jit {
namespace fuser {

//TODO: Type promotion

enum class TORCH_API ValType {
  Float
//   Expr
// , TensorLike
// , Addr
// , Float
// , Range
};

enum class TORCH_API ExprType {
  Add
//   Loop  // swap, merge, split
// , Index
// , Add
};

TORCH_API std::string stringify(const ValType);
TORCH_API std::string stringify(const ExprType);

TORCH_API std::ostream& operator<<(std::ostream& out, const ValType valtype);
TORCH_API std::ostream& operator<<(std::ostream& out, const ExprType exprtype);

}}} // torch::jit::fuser
