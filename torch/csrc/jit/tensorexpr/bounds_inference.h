#pragma once

#include <map>
#include <unordered_map>
#include <vector>

#include <torch/csrc/WindowsTorchApiMacro.h>

namespace torch {
namespace jit {
namespace tensorexpr {

class Expr;
class Buf;
class Stmt;

enum TORCH_API TensorAccessKind { kLoad, kStore };

struct TORCH_API TensorAccessBoundsInfo {
  TensorAccessKind kind;
  std::vector<const Expr*> start;
  std::vector<const Expr*> stop;
};

using BoundsInfo =
    std::unordered_map<const Buf*, std::vector<TensorAccessBoundsInfo>>;

TORCH_API BoundsInfo inferBounds(Stmt* s);

TORCH_API void printBoundsInfo(const BoundsInfo& v);

TORCH_API BoundsInfo mergeTensorAccesses(const BoundsInfo& unmerged);

} // namespace tensorexpr
} // namespace jit
} // namespace torch
