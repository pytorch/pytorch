#include <torch/csrc/jit/codegen/cuda/iter_visitor.h>

#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>
#include <torch/csrc/jit/codegen/cuda/type.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

/* ITER VISITOR */

namespace {

// Remove any stmt in stmts that is in visited
void remove_visited(
    std::vector<Statement*>& stmts,
    const std::unordered_set<Statement*>& visited) {
  std::deque<std::vector<Statement*>::iterator> to_erase;
  for (auto it = stmts.begin(); it != stmts.end(); it++) {
    if (visited.find(*it) != visited.end()) {
      to_erase.push_back(it);
    }
  }

  while (!to_erase.empty()) {
    stmts.erase(to_erase.back());
    to_erase.pop_back();
  }
}

} // namespace

// Implementation details:
// We start with an entry in stmt_stack that is the outputs we want to
// process. We cannot process these outputs untill all Stmts in their history
// have been processed, as those Stmts contain all dependencies to produce
// these values. What we will do is traverse towards inputs until we hit a
// leaf node. Once we hit a leaf node that node will be visited, then we will
// take them off the stack. Once a stack entry is empty, know everything
// needed to be visited to visit stmt_stack.back().back(). We then visit that
// node, make it as visisted and remove it from the stack.
//
// To prevent traversing all paths through a DAG (unless we want to) we have a
// function to remove visited nodes from being re-added to the stack
// (remove_visited).
void IterVisitor::traverseFrom(
    Fusion* fusion,
    const std::vector<Val*>& from,
    bool traverseAllPaths) {
  FusionGuard fg(fusion);

  std::unordered_set<Statement*> visited;

  stmt_stack.clear();
  stmt_stack.emplace_back(from.rbegin(), from.rend());

  bool all_inputs_visited = false;

  while (!stmt_stack.empty()) {
    auto& current_inputs = stmt_stack.back();

    // If current_inputs is empty, pop a level of the stmt_stack, mark the level
    // we pop to as having all inputs processed, the layer we processed were all
    // added inputs required for that Stmt.
    if (current_inputs.empty()) {
      stmt_stack.pop_back();
      all_inputs_visited = true;
      continue;
    }

    // Get the very last entry in the stack to process
    const auto& stmt = current_inputs.back();

    // If we just poped a stmt_stack level, we can finally visit it!
    if (all_inputs_visited) {
      // Mark visited
      visited.insert(stmt);

      // Actually visit stmt
      handle(stmt);

      // Remove last value just visited
      current_inputs.pop_back();

      // Mark that we need to visit a new Stmt's.
      all_inputs_visited = false;
    } else {
      // We're not ready to process this node, so add all its inputs to be
      // checked Visit input nodes.
      auto next_stmts = next(stmt);
      // We may want to retraverse nodes, in that case revisit everything!
      if (!traverseAllPaths) {
        // If we don't want to retraverse, remove nodes we already visisted.
        remove_visited(next_stmts, visited);
      }
      if (next_stmts.empty()) {
        // If there's nothing to visit because it was all already visited, mark
        // to process
        all_inputs_visited = true;
      } else {
        // Add all these new stmts to visit to the stack.
        stmt_stack.emplace_back(next_stmts.rbegin(), next_stmts.rend());
        // We have new things to visit,
        all_inputs_visited = false;
      }
    }
  }
}

void IterVisitor::traverse_(
    Fusion* fusion,
    bool from_outputs_only,
    bool traverse_all_paths) {
  FusionGuard fg(fusion);

  if (from_outputs_only) {
    auto term_val_outs = fusion->getTerminatingOutputs();
    if (!term_val_outs.empty()) {
      traverseFrom(fusion, term_val_outs, traverse_all_paths);
    }
    return;
  }

  std::vector<Val*> leaves;
  // Search for Vals with no uses (output edges)
  for (Val* val : fusion->deterministic_vals())
    if (!fusion->used(val)) {
      leaves.push_back(val);
    }

  if (!leaves.empty()) {
    traverseFrom(fusion, leaves, traverse_all_paths);
  }
}

void IterVisitor::traverse(Fusion* fusion, bool from_outputs_only) {
  traverse_(fusion, from_outputs_only, false);
}

void IterVisitor::traverseAllPaths(Fusion* fusion, bool from_outputs_only) {
  traverse_(fusion, from_outputs_only, true);
}

namespace {

// Expr sort will take a fusion and return a topologically sorted list of
// expressions.
class Inputs : public IterVisitor {
 private:
  std::unordered_set<Val*> inputs;

  void handle(Val* val) override {
    if (val->getOrigin() == nullptr) {
      inputs.emplace(val);
    }
  }

 public:
  static std::unordered_set<Val*> getInputs(const std::vector<Val*>& of) {
    if (of.empty()) {
      return std::unordered_set<Val*>();
    }
    Inputs inps;
    inps.traverseFrom(of[0]->fusion(), of);
    return inps.inputs;
  }
};

} // namespace

std::unordered_set<Val*> IterVisitor::getInputsTo(
    const std::vector<Val*>& vals) {
  return Inputs::getInputs(vals);
}

namespace {

class AllVals : public IterVisitor {
 private:
  std::unordered_set<Val*> vals;

  void handle(Val* val) final {
    vals.emplace(val);
  }

 public:
  // Return all values in history of all values in from
  static std::unordered_set<Val*> get(
      Fusion* fusion,
      const std::vector<Val*>& from) {
    AllVals av;
    av.traverseFrom(fusion, from, false);
    return av.vals;
  }
};

} // namespace

/* BACKWARDS VISITOR */

std::vector<Statement*> BackwardVisitor::next(Statement* stmt) {
  if (stmt->isVal()) {
    return next(stmt->as<Val>());
  } else if (stmt->isExpr()) {
    return next(stmt->as<Expr>());
  } else {
    TORCH_INTERNAL_ASSERT(
        false, "BackwardVisitor could not detect type in next_dispatch.");
  }
}

std::vector<Statement*> BackwardVisitor::next(Expr* expr) {
  return std::vector<Statement*>(
      expr->outputs().begin(), expr->outputs().end());
}

std::vector<Statement*> BackwardVisitor::next(Val* val) {
  // Going to sort based on relative topological position
  std::map<size_t, Statement*> exprs;

  for (auto expr : FusionGuard::getCurFusion()->unordered_uses(val)) {
    // Make sure it's an expr we can traverse
    if (traversal_exprs_.find(expr) != traversal_exprs_.end()) {
      exprs[traversal_exprs_[expr]] = expr;
    }
  }

  std::vector<Statement*> next_stmts(exprs.size());
  std::transform(
      exprs.begin(),
      exprs.end(),
      next_stmts.begin(),
      [](std::pair<size_t, Statement*> pair) { return pair.second; });

  return next_stmts;
}

void BackwardVisitor::traverseFrom(
    Fusion* fusion,
    const std::vector<Val*>& from,
    bool traverseAllPaths) {
  FusionGuard fg(fusion);

  // Reset members
  stmt_stack_.clear();
  traversal_exprs_.clear();

  if (from.empty()) {
    return;
  }

  auto vals = AllVals::get(fusion, from);

  auto exprs = ExprSort::getExprs(fusion, from);

  {
    size_t pos = 0;
    for (auto expr : exprs)
      traversal_exprs_[expr] = pos++;
  }

  // All stmts we've called handle on
  std::unordered_set<Statement*> visited_stmts_;

  for (auto traversal_pair : traversal_exprs_) {
    for (auto out : traversal_pair.first->outputs()) {
      TORCH_INTERNAL_ASSERT(
          vals.find(out) != vals.end(),
          "Invalid backward traversal found. Some output paths were not provided.");
    }
  }

  auto inputs = InputsOf::getInputsTo(from);
  stmt_stack_.emplace_back(inputs.begin(), inputs.end());

  // The rest is basically copy-pasted from IterVitor:
  while (!stmt_stack_.empty()) {
    auto next_stmts = next(stmt_stack_.back().back());

    // Remove statements we already visited if we're not traversing all paths
    if (!traverseAllPaths) {
      remove_visited(next_stmts, visited_stmts_);
    }

    // Traverse down until we get to a leaf
    while (!next_stmts.empty()) {
      stmt_stack_.emplace_back(next_stmts.rbegin(), next_stmts.rend());
      next_stmts = next(stmt_stack_.back().back());
      // Remove statements we already visited if we're not traversing all paths
      if (!traverseAllPaths) {
        remove_visited(next_stmts, visited_stmts_);
      }
    }

    // Traverse back up
    // Mark visited
    visited_stmts_.emplace(stmt_stack_.back().back());
    // Handle
    handle(stmt_stack_.back().back());
    // Remove
    stmt_stack_.back().pop_back();

    while (!stmt_stack_.empty() && stmt_stack_.back().empty()) {
      stmt_stack_.pop_back();
      if (!stmt_stack_.empty()) {
        // Mark visited
        visited_stmts_.emplace(stmt_stack_.back().back());
        // Handle
        handle(stmt_stack_.back().back());
        // Remove
        stmt_stack_.back().pop_back();
      }
    }
  }
}

/* DEPENDENCY CHECKING */

namespace {

// Looks for and returns all values in between dependencies and vals, including
// them.
struct Dependencies : public IterVisitor {
  std::unordered_set<Val*> dependencies_;
  std::unordered_set<Val*> vals_;

  std::vector<Statement*> next(Val* v) override {
    if (dependencies_.find(v) != dependencies_.end())
      return std::vector<Statement*>();
    return IterVisitor::next(v);
  }

  void handle(Val* val) override {
    vals_.emplace(val);
  }

  Dependencies(
      std::unordered_set<Val*> _dependencies,
      const std::vector<Val*>& of)
      : dependencies_(std::move(_dependencies)) {
    traverseFrom(of[0]->fusion(), of, false);
  };

 public:
  static std::unordered_set<Val*> getAllVals(
      const std::unordered_set<Val*>& dependencies,
      const std::vector<Val*>& of) {
    if (of.empty()) {
      return std::unordered_set<Val*>();
    }

    Dependencies deps(dependencies, of);
    return deps.vals_;
  }
};

// Looks for and returns all output values with dependencies on `of`.
struct FindOutputs : public IterVisitor {
  const std::unordered_set<Val*>& of_;
  std::unordered_set<Val*> outs_;

  void handle(Val* val) override {
    if (of_.find(val) != of_.end()) {
      Statement* out_stmt = stmt_stack.front().back();
      if (out_stmt->isVal()) {
        auto out_val = out_stmt->as<Val>();
        if (of_.find(out_val) == of_.end()) {
          outs_.emplace(out_val);
        }
      }
    }
  }

  FindOutputs(const std::unordered_set<Val*>& _of) : of_(_of) {
    auto fusion = (*of_.begin())->fusion();
    traverseFrom(fusion, fusion->outputs(), false);
  };

  static std::unordered_set<Val*> getAllOutputsOf(
      const std::unordered_set<Val*>& of) {
    if (of.empty()) {
      return std::unordered_set<Val*>();
    }

    FindOutputs finder(of);
    return finder.outs_;
  }
};

class DependencyChains : public IterVisitor {
 public:
  std::deque<std::deque<Val*>> dep_chains;
  bool is_dependency = false;
  std::unordered_set<Val*> dependencies_;

  void handle(Val* val) override {
    if (dependencies_.find(val) != dependencies_.end()) {
      is_dependency = true;
      std::deque<Val*> deps;
      for (auto stack : stmt_stack) {
        if (stack.back()->isVal()) {
          deps.push_back(stack.back()->as<Val>());
        }
      }
      // Order as dependency -> of
      dep_chains.emplace_back(deps.rbegin(), deps.rend());
    }
  }

  DependencyChains(Val* _dependency, Val* _of, bool all_chains_ = false)
      : dependencies_({_dependency}) {
    traverseFrom(_of->fusion(), {_of}, all_chains_);
  }

  DependencyChains(Val* _dependency, bool all_chains_ = false)
      : dependencies_({_dependency}) {
    if (all_chains_) {
      traverseAllPaths(_dependency->fusion(), false);
    } else {
      traverse(_dependency->fusion(), false);
    }
  }

  DependencyChains(
      std::unordered_set<Val*> _dependencies,
      bool all_chains_ = false)
      : dependencies_(std::move(_dependencies)) {
    if (dependencies_.empty()) {
      return;
    }

    if (all_chains_) {
      traverseAllPaths((*dependencies_.begin())->fusion(), false);
    } else {
      traverse((*dependencies_.begin())->fusion(), false);
    }
  }

  static std::deque<Val*> getDependencyChain(Val* dependency, Val* of) {
    DependencyChains dp(dependency, of, false);
    if (dp.dep_chains.empty()) {
      return std::deque<Val*>();
    }
    return dp.dep_chains[0];
  }

  // I don't think this is actually hooked up, but leaving for now.
  static std::deque<std::deque<Val*>> getDependencyChains(
      Val* dependency,
      Val* of) {
    DependencyChains dp(dependency, of, true);
    if (dp.dep_chains.empty()) {
      return std::deque<std::deque<Val*>>();
    }
    return dp.dep_chains;
  }

  static std::deque<std::deque<Val*>> getAllUseChains(Val* dependency) {
    DependencyChains dp(dependency, true);
    if (dp.dep_chains.empty()) {
      return std::deque<std::deque<Val*>>();
    }
    return dp.dep_chains;
  }

  static std::deque<std::deque<Val*>> getAllUseChains(
      const std::unordered_set<Val*>& dependencies) {
    DependencyChains dp(dependencies, true);
    if (dp.dep_chains.empty()) {
      return std::deque<std::deque<Val*>>();
    }
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

std::deque<std::deque<Val*>> DependencyCheck::getAllUseChains(Val* producer) {
  return DependencyChains::getAllUseChains(producer);
}

std::unordered_set<Val*> DependencyCheck::getAllValsBetween(
    const std::unordered_set<Val*>& dependencies,
    const std::vector<Val*>& of) {
  return Dependencies::getAllVals(dependencies, of);
}

std::unordered_set<Val*> DependencyCheck::getAllOutputsOf(
    const std::unordered_set<Val*>& of) {
  if (of.empty()) {
    return std::unordered_set<Val*>();
  }
  FusionGuard fg((*of.begin())->fusion());
  return FindOutputs::getAllOutputsOf(of);
}

void ExprSort::handle(Expr* expr) {
  exprs.push_back(expr);
}

std::vector<Expr*> ExprSort::getExprs(Fusion* fusion, bool from_outputs_only) {
  ExprSort es;
  es.traverse(fusion, from_outputs_only);
  return es.exprs;
}

std::vector<Expr*> ExprSort::getExprs(
    Fusion* fusion,
    const std::vector<Val*>& from) {
  ExprSort es;
  es.traverseFrom(fusion, from, false);
  return es.exprs;
}

void InputsOf::handle(Val* v) {
  if (FusionGuard::getCurFusion()->origin(v) == nullptr)
    inputs.emplace(v);
}

std::unordered_set<Val*> InputsOf::output(Fusion* fusion, Val* output_) {
  InputsOf io;
  io.traverseFrom(FusionGuard::getCurFusion(), {output_}, false);
  return io.inputs;
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
