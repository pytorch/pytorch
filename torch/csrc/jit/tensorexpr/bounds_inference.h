#pragma once

#include <map>
#include <unordered_map>
#include <vector>

#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/csrc/jit/tensorexpr/mem_dependency_checker.h>

namespace torch {
namespace jit {
namespace tensorexpr {

class Expr;
class Buf;
class Stmt;

enum C10_API_ENUM TensorAccessKind { kLoad, kStore, kMutate };

struct TORCH_API TensorAccessBoundsInfo {
  TensorAccessKind kind;
  std::vector<const Expr*> start;
  std::vector<const Expr*> stop;
};

using BoundsInfo =
    std::unordered_map<const Buf*, std::vector<TensorAccessBoundsInfo>>;

TORCH_API BoundsInfo inferBounds(Stmt* s, bool distinctAccessKinds = true);

TORCH_API void printBoundsInfo(const BoundsInfo& v);

TORCH_API std::vector<const Expr*> getBoundExtents(
    const std::vector<TensorAccessBoundsInfo>& infos);

} // namespace tensorexpr
} // namespace jit
} // namespace torch
