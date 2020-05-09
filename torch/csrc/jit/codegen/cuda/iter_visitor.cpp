#include <torch/csrc/jit/codegen/cuda/iter_visitor.h>
#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/type.h>

namespace torch {
namespace jit {
namespace fuser {

std::vector<Statement*> IterVisitor::next(Statement* statement) {
  if (statement->isVal())
    return next(static_cast<Val*>(statement));
  else if (statement->isExpr())
    return next(static_cast<Expr*>(statement));
  else
    TORCH_INTERNAL_ASSERT(
        false, "IterVisitor could not detect type in next_dispatch.");
}

std::vector<Statement*> IterVisitor::next(Val* v) {
  if (FusionGuard::getCurFusion()->origin(v) != nullptr)
    return {FusionGuard::getCurFusion()->origin(v)};
  return {};
}

std::vector<Statement*> IterVisitor::next(Expr* expr) {
  FusionGuard::getCurFusion()->assertInFusion(expr, "Cannot traverse expr, ");
  return {expr->inputs().begin(), expr->inputs().end()};
}

// Remove any stmt in stmts that is in visited
namespace {
void remove_visited(
    std::vector<Statement*>& stmts,
    const std::unordered_set<Statement*>& visited) {
  std::deque<std::vector<Statement*>::iterator> to_erase;
  for (auto it = stmts.begin(); it != stmts.end(); it++) {
    if (visited.find(*it) != visited.end())
      to_erase.push_back(it);
  }

  while (!to_erase.empty()) {
    stmts.erase(to_erase.back());
    to_erase.pop_back();
  }
}
} // namespace

void IterVisitor::traverseFrom(
    Fusion* const fusion,
    const std::vector<Val*>& from,
    bool traverseAllPaths) {
  FusionGuard fg(fusion);
  std::unordered_set<Statement*> visited;
  stmt_stack.clear();
  stmt_stack.push_back(std::vector<Statement*>(from.rbegin(), from.rend()));

  while (!stmt_stack.empty()) {
    auto next_stmts = next(stmt_stack.back().back());

    // Remove statements we already visited if we're not traversing all paths
    if (!traverseAllPaths)
      remove_visited(next_stmts, visited);

    // Traverse down until we get to a leaf
    while (!next_stmts.empty()) {
      stmt_stack.push_back(
          std::vector<Statement*>(next_stmts.rbegin(), next_stmts.rend()));
      next_stmts = next(stmt_stack.back().back());
      // Remove statements we already visited if we're not traversing all paths
      if (!traverseAllPaths)
        remove_visited(next_stmts, visited);
    }

    // Traverse back up
    // Mark visited
    visited.emplace(stmt_stack.back().back());
    // Handle
    handle(stmt_stack.back().back());
    // Remove
    stmt_stack.back().pop_back();

    while (!stmt_stack.empty() && stmt_stack.back().empty()) {
      stmt_stack.pop_back();
      if (!stmt_stack.empty()) {
        // Mark visited
        visited.emplace(stmt_stack.back().back());
        // Handle
        handle(stmt_stack.back().back());
        // Remove
        stmt_stack.back().pop_back();
      }
    }
  }
}

void IterVisitor::traverse(
    Fusion* const fusion,
    bool from_outputs_only,
    bool breadth_first) {
  FusionGuard fg(fusion);
  if (breadth_first)
    TORCH_INTERNAL_ASSERT(false, "Not implemented yet.");

  std::vector<Val*> outputs;
  for (Val* out : fusion->outputs()) {
    outputs.push_back(out);
  }
  // Search for Vals with no uses (output edges)
  if (!from_outputs_only)
    for (Val* val : fusion->vals()) {
      if (!fusion->used(val))
        if (!fusion->hasOutput(val))
          outputs.push_back(val);
    }

  if (outputs.empty())
    return;

  traverseFrom(fusion, outputs, false);
}

void IterVisitor::traverseAllPaths(
    Fusion* const fusion,
    bool from_outputs_only,
    bool breadth_first) {
  FusionGuard fg(fusion);
  if (breadth_first)
    TORCH_INTERNAL_ASSERT(false, "Not implemented yet.");

  std::vector<Val*> outputs;
  for (Val* out : fusion->outputs()) {
    outputs.push_back(out);
  }
  // Search for Vals with no uses (output edges)
  if (!from_outputs_only)
    for (Val* val : fusion->vals()) {
      if (!fusion->used(val))
        if (!fusion->hasOutput(val))
          outputs.push_back(val);
    }

  traverseFrom(fusion, outputs, true);
}

/* DEPENDENCY CHECKING */

namespace {

// Looks for and returns
struct DependencyChains : public IterVisitor {
  std::deque<std::deque<Val*>> dep_chains;
  bool is_dependency = false;
  Val* dependency_;

  void handle(Val* val) {
    if (val->sameAs(dependency_)) {
      is_dependency = true;
      std::deque<Val*> deps;
      for (auto stack : stmt_stack) {
        if (stack.back()->isVal())
          deps.push_back(static_cast<Val*>(stack.back()));
      }
      // Order as dependency -> of
      dep_chains.push_back(std::deque<Val*>(deps.rbegin(), deps.rend()));
    }
  }

  DependencyChains(Val* _dependency, Val* _of, bool all_chains_ = false)
      : dependency_(_dependency) {
    traverseFrom(_of->fusion(), {_of}, all_chains_);
  }

  DependencyChains(Val* _dependency, bool all_chains_ = false)
      : dependency_(_dependency) {
    if (all_chains_)
      traverseAllPaths(_dependency->fusion(), false);
    else
      traverse(_dependency->fusion(), false);
  }

  static std::deque<Val*> getDependencyChain(Val* dependency, Val* of) {
    DependencyChains dp(dependency, of, false);
    if (dp.dep_chains.empty())
      return std::deque<Val*>();
    return dp.dep_chains[0];
  }

  static std::deque<std::deque<Val*>> getDependencyChains(
      Val* dependency,
      Val* of) {
    DependencyChains dp(dependency, of, true);
    if (dp.dep_chains.empty())
      return std::deque<std::deque<Val*>>();
    return dp.dep_chains;
  }

  static std::deque<std::deque<Val*>> getDependencyChainsTo(Val* dependency) {
    DependencyChains dp(dependency, true);
    if (dp.dep_chains.empty())
      return std::deque<std::deque<Val*>>();
    return dp.dep_chains;
  }
};

} // namespace

bool DependencyCheck::isDependencyOf(Val* dependency, Val* of) {
  return !DependencyChains::getDependencyChain(dependency, of).empty();
}

std::deque<Val*> DependencyCheck::getSingleDependencyChain(
    Val* dependency,
    Val* of) {
  return DependencyChains::getDependencyChain(dependency, of);
}

std::deque<std::deque<Val*>> DependencyCheck::getAllDependencyChains(
    Val* dependency,
    Val* of) {
  return DependencyChains::getDependencyChains(dependency, of);
}

std::deque<std::deque<Val*>> DependencyCheck::getAllDependencyChainsTo(
    Val* dependency) {
  return DependencyChains::getDependencyChainsTo(dependency);
}

} // namespace fuser
} // namespace jit
} // namespace torch
