#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>

#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>

#include <bitset>

// Provides utilities for dealing with nested ForLoop and IfThenElse scopes

namespace torch {
namespace jit {
namespace fuser {

class ThreadPredicateMap;

namespace scope_utils {

// Grab the ForLoop starting from scope working out
std::vector<kir::ForLoop*> getLoops(Expr* scope);

// Track how far our for loop scope is
unsigned int computeForDepth(Expr* scope);

// Push back an expr to scope
void pushBack(Expr* scope, Expr* expr);

// Insert expr in scope before ref
void insertBefore(Expr* scope, Expr* ref, Expr* expr);

// Return the parent of the active scope
Expr* getParent(Expr* scope);

// Open a new inner most for loop
kir::ForLoop* openFor(Expr* scope, IterDomain*);

// Close the inner most for loop
Expr* closeScope(Expr* scope);

// Clear all expressions from the scope
Expr* clearScope(Expr* scope);

// Provide a new for loop matching the one provided, sets parent_scope as
// parent_scope, but does not insert into parent scope.
kir::ForLoop* cloneLoopNest(kir::ForLoop* to_clone, Expr* parent_scope);

// Run through a scope and replace expressions inside with replacement_map
void replaceExprsInScope(
    Expr* scope,
    std::unordered_map<Expr*, Expr*> replacement_map);

Expr* firstInnerMostScope(Expr* scope);

} // namespace scope_utils

namespace ir_utils {

std::vector<Val*> indices(std::vector<kir::ForLoop*>);

std::vector<IterDomain*> iterDomains(std::vector<kir::ForLoop*>);

bool isTV(const Val* const);

bool isTVOp(const Expr*);

bool isScalarOp(const Expr*);

void ASSERT_EXPR(Statement*);

bool isScope(const Expr*);

Expr* asExpr(Statement*);

TensorView* asTV(Val*);

kir::ForLoop* asForLoop(Statement*);

const TensorView* asConstTV(const Val* const);

bool isUnrolledFor(const Expr*);

// Represents mapping to bool from BIDx, BIDy, BIDz, TIDx, TIDy and TIDz.
class ParallelTypeBitmap {
 public:
  static constexpr int num_p_type = 6;
  ParallelTypeBitmap() = default;
  bool get(ParallelType pt) const;
  bool set(ParallelType pt, bool);
  ParallelTypeBitmap operator&=(const ParallelTypeBitmap& other);
  ParallelTypeBitmap operator|=(const ParallelTypeBitmap& other);
  ParallelTypeBitmap operator^=(const ParallelTypeBitmap& other);
  ParallelTypeBitmap operator~() const;
  bool none() const;
  bool any() const;
  bool all() const;
  bool operator[](size_t pos) const;
  std::map<ParallelType, bool> getMap() const;

 private:
  ParallelTypeBitmap(const std::bitset<num_p_type>& bs) : bitset_(bs) {}
  std::bitset<num_p_type> bitset_;
  const static std::unordered_map<ParallelType, int> pt_to_offset_;
  const static std::unordered_map<int, ParallelType> offset_to_pt_;
};

ParallelTypeBitmap operator&(
    const ParallelTypeBitmap& lhs,
    const ParallelTypeBitmap& rhs);

ParallelTypeBitmap operator|(
    const ParallelTypeBitmap& lhs,
    const ParallelTypeBitmap& rhs);

ParallelTypeBitmap operator^(
    const ParallelTypeBitmap& lhs,
    const ParallelTypeBitmap& rhs);

// Returns a ParallelTypeBitmap representing which domain needs
// blockBroadcast.
// Even when a domain is broadcast and parallelized, it does not need
// blockBroadcast unless it is predicated.
ParallelTypeBitmap getParallelBroadcastDomains(
    const BroadcastOp* const,
    const ThreadPredicateMap& preds);

} // namespace ir_utils
} // namespace fuser
} // namespace jit
} // namespace torch
