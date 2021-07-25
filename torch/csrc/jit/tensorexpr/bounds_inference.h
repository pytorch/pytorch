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

// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
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
TORCH_API BoundsInfo getInferredBounds(
    analysis::MemDependencyChecker& analyzer,
    Expr* e,
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

// Returns true if there is a conflicting overlap between accesses in
// statements A and B. A conflicting overlap is an overlap in buffer accesses
// where at least one of the accesses is a Store.
TORCH_API bool hasConflictingOverlap(
    analysis::MemDependencyChecker& analyzer,
    Stmt* A,
    Stmt* B);
// Same as above, between accesses in stores S1 and S2.
TORCH_API bool isOverlapping(
    analysis::MemDependencyChecker& analyzer,
    Store* S1,
    Store* S2);
// Same as above, between accesses in store S and load L.
TORCH_API bool isOverlapping(
    analysis::MemDependencyChecker& analyzer,
    Store* S,
    Load* L);

} // namespace tensorexpr
} // namespace jit
} // namespace torch
