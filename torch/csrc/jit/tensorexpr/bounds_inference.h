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

// Bounds inference caching the analysis. The MemDependencyChecker must already
// have been run.
TORCH_API BoundsInfo getInferredBounds(
    analysis::MemDependencyChecker& analyzer,
    Stmt* s,
    bool distinctAccessKinds = true);

TORCH_API void printBoundsInfo(const BoundsInfo& v);

TORCH_API std::vector<const Expr*> getBoundExtents(
    const std::vector<TensorAccessBoundsInfo>& infos);

// The kind of dependency found, in increasing order of exclusivity.
enum class HazardKind {
  ReadAfterWrite,
  WriteAfterRead,
  WriteAfterWrite,
  NoDependency,
};
TORCH_API HazardKind
getPotentialHazards(analysis::MemDependencyChecker& analyzer, Stmt* A, Stmt* B);

// Returns true if there is a partial overlap between accesses in A and B.
TORCH_API bool hasPartialOverlap(
    analysis::MemDependencyChecker& analyzer,
    Stmt* A,
    Stmt* B);

} // namespace tensorexpr
} // namespace jit
} // namespace torch
