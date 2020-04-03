#include <torch/csrc/jit/codegen/cuda/iter_visitor.h>
#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/type.h>
#include <deque>
#include <iostream>
#include <queue>

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

void IterVisitor::traverseFrom(
    Fusion* const fusion,
    const std::vector<Val*>& from) {
  FusionGuard fg(fusion);

  std::set<Statement*> visited;
  std::deque<Statement*> to_visit;

  std::queue<Val*> outputs_to_visit;
  for (Val* entry : from)
    outputs_to_visit.emplace(entry);

  while (!outputs_to_visit.empty()) {
    if (stopCondition())
      break;

    to_visit.push_front(outputs_to_visit.front());
    outputs_to_visit.pop();
    while (!to_visit.empty()) {
      if (stopCondition())
        break;

      Statement* stmt = to_visit.front();
      std::vector<Statement*> inps = next(stmt);
      for (auto it = inps.rbegin(); it != inps.rend(); it++) {
        Statement* inp = *it;
        if (visited.find(inp) == visited.end()) {
          toVisitCallback(inp);
          to_visit.emplace_front(inp);
        }
      }

      if (to_visit.front() != stmt) {
        continue;
      }

      to_visit.pop_front();
      if (visited.find(stmt) == visited.end()) {
        handle(stmt);
        visited.emplace(stmt);
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
  std::set<Statement*> visited;
  std::deque<Statement*> to_visit;

  std::vector<Val*> outputs_to_visit;
  if (from_outputs_only) {
    for (Val* out : fusion->outputs()) {
      outputs_to_visit.push_back(out);
    }
    // Search for Vals with no uses (output edges)
  } else
    for (Val* val : fusion->vals()) {
      if (!fusion->used(val))
        outputs_to_visit.push_back(val);
    }

  traverseFrom(fusion, outputs_to_visit);
}

void DependencyCheck::handle(Val* val) {
  // Debug dependency chain
  if (val->sameAs(dependency_))
    is_dependency = true;
}

void DependencyCheck::handle(Expr* expr) {
  // We want to update the dependency chain, but we want to make sure
  // that the top value on the chain is an output of this expr

  for (decltype(expr->nOutputs()) i = 0; i < expr->nOutputs(); i++) {
    TORCH_INTERNAL_ASSERT(
        expr->hasOutput(dep_chain.top()),
        "IterVisitor attempted to visit an expr, but this expr was visited in an incorrect order.");
    dep_chain.pop();
  }
}

void DependencyCheck::toVisitCallback(Statement* stmt) {
  // If an expression push outputs of expr to dependency chain.
  if (stmt->isExpr()) {
    Expr* expr = static_cast<Expr*>(stmt);
    for (auto out : expr->outputs()) {
      dep_chain.push(static_cast<Val*>(out));
    }
  }
}

bool DependencyCheck::check() {
  is_dependency = false;
  IterVisitor::traverseFrom(of_->fusion(), {of_});
  return is_dependency;
}

std::stack<Val*> DependencyCheck::getDependencyChain(Val* dependency, Val* of) {
  DependencyCheck dp(dependency, of);
  dp.check();

  // Return the reversed stack, we start from output and go to the input,
  // including of, but not dependency
  std::stack<Val*> dep_copy = dp.dep_chain;
  std::stack<Val*> reversed_clean;

  while (!dep_copy.empty()) {
    Val* next = dep_copy.top();
    dep_copy.pop();
    reversed_clean.push(next);
  }
  return reversed_clean;
}

} // namespace fuser
} // namespace jit
} // namespace torch
