#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>
#include <c10/util/Exception.h>

#include <cstdint>
#include <iostream>
#include <string>

namespace torch {
namespace jit {
namespace fuser {

//Order on strength
enum class TORCH_API ValType {
    Tensor
  , Float
  , Int
  
  
// , Addr
// , Range
};

enum class TORCH_API ExprType {
  Add
// , Sub
// , Mul
// , Div
// , Mod
// , Loop
// , Swap
// , Merge
// , Split
// , Index
// , Add
};

ValType promote_scalar(const ValType& t1, const ValType& t2);

TORCH_API std::string stringify(const ValType);
TORCH_API std::string stringify(const ExprType);

TORCH_API std::ostream& operator<<(std::ostream& out, const ValType valtype);
TORCH_API std::ostream& operator<<(std::ostream& out, const ExprType exprtype);

}}} // torch::jit::fuser
