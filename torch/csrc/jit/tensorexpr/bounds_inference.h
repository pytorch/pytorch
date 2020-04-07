#pragma once

#include <unordered_map>
#include <vector>

#include <torch/csrc/WindowsTorchApiMacro.h>

namespace torch {
namespace jit {
namespace tensorexpr {

class Expr;
class Buf;
class Stmt;

enum TensorAccessKind { kLoad, kStore };

struct TensorAccessBoundsInfo {
  const Buf* buf;
  TensorAccessKind kind;
  std::vector<const Expr*> start;
  std::vector<const Expr*> stop;
};

using BoundsInfo = std::vector<TensorAccessBoundsInfo>;

TORCH_API BoundsInfo inferBounds(Stmt* s);

TORCH_API void printBoundsInfo(const BoundsInfo& v);

} // namespace tensorexpr
} // namespace jit
} // namespace torch
