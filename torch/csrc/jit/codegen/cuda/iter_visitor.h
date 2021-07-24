#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>

#include <torch/csrc/jit/codegen/cuda/dispatch.h>

#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>
#include <torch/csrc/jit/codegen/cuda/type.h>

#include <deque>
#include <unordered_set>
#include <vector>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

/*
 * IterVisitor starts from leaf nodes, fusion outputs, or the provided values.
 * It walks the DAG bacwkards from the starting nodes, to roots. Each node in
 * the dag will be called with handle(Statement*) in topolgical order inputs of
 * the fusion to outputs of the fusion.
 *
 * TODO: We may want a BFS version of this code to extract ILP, not implemented
 * yet.
 *
 * TODO: We may want to have ordering of outputs to inputs. I'm not sure why we
 * would want this, but seems like it would be a reasonable request.
 */
// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
class TORCH_CUDA_CU_API IterVisitor : public OptOutDispatch {
 public:
  // NOLINTNEXTLINE(modernize-use-override)
  virtual ~IterVisitor() = default;

  IterVisitor() = default;

  IterVisitor(const IterVisitor& other) = default;
  IterVisitor& operator=(const IterVisitor& other) = default;

  IterVisitor(IterVisitor&& other) = default;
  IterVisitor& operator=(IterVisitor&& other) = default;

 protected:
  // Functions return nodes in reverse order to be added to the to_visit queue
  // These functions will start at outputs and propagate up through the DAG
  // to inputs based on depth first traversal. Next could be called on a node
  // multiple times.
  virtual std::vector<Statement*> next(Statement* stmt) {
    // NOLINTNEXTLINE(bugprone-branch-clone)
    if (stmt->isVal()) {
      return next(stmt->as<Val>());
    } else if (stmt->isExpr()) {
      return next(stmt->as<Expr>());
    } else {
      TORCH_INTERNAL_ASSERT(
          false, "IterVisitor could not detect type in next_dispatch.");
    }
  }

  virtual std::vector<Statement*> next(Val* v) {
    FusionGuard::getCurFusion()->assertInFusion(v, "Cannot traverse val, ");
    if (FusionGuard::getCurFusion()->origin(v) != nullptr) {
      return {FusionGuard::getCurFusion()->origin(v)};
    }
    return {};
  }

  virtual std::vector<Statement*> next(Expr* expr) {
    FusionGuard::getCurFusion()->assertInFusion(expr, "Cannot traverse expr, ");
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    std::vector<Statement*> next_stmts{
        expr->inputs().begin(), expr->inputs().end()};
    return next_stmts;
  }

  // This handle functions is called on every Statement* in topological order,
  // starting from outputs to inputs.
  void handle(Statement* s) override {
    OptOutDispatch::handle(s);
  }
  // This handle functions is called on every Expr* in topological order,
  // starting from outputs to inputs.
  void handle(Expr* e) override {
    OptOutDispatch::handle(e);
  }
  // This handle functions is called on every Val* in topological order,
  // starting from outputs to inputs.
  void handle(Val* v) override {
    OptOutDispatch::handle(v);
  }

  // The entire stack during traversal. stmt_stack.back().back() is the node
  // that is being called in handle(). stmt_stack.back() contains siblings (not
  // guarenteed to be all siblings throughout traversal). stmt_stack.front()
  // contains the outputs we started with (not guarenteed to be all outputs
  // throughout traversal).
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  std::vector<std::vector<Statement*>> stmt_stack;

  // Statements to stop traversal on if they're hit (pretends they're leaf
  // nodes in next)
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  std::unordered_set<Statement*> termination_stmts;

  void traverse_(
      Fusion* fusion,
      bool from_outputs_only = false,
      bool traverse_all_paths = false);

 public:
  // Starts at nodes provided in from, traverses from these nodes to inputs.
  // Calls handle on all Statement*s in topological sorted order.
  // traverseAllPaths = false only call handle on each Statement* once
  // traverseAllPaths = true traverses all paths from nodes in from to inputs.
  // Handle on a Statement* for every path from "from" nodes, to inputs.
  // to argument allows specification of nodes to stop at if we want to stop
  // beffore we hit all leaf nodes. This can be helpful when we want to traverse
  // from TensorView::domain(), to the rfactor domain, instead of root domain.
  void traverseFrom(
      Fusion* fusion,
      const std::vector<Val*>& from,
      bool traverseAllPaths = false);

  // from_outputs_only = true start from outputs registered with fusion,
  // from_outputs_only = false start from all leaf nodes. Calls into
  // traverseFrom.
  void traverse(Fusion* fusion, bool from_outputs_only = false);

  // from_outputs_only = true start from outputs registered with fusion,
  // from_outputs_only = false start from all leaf nodes. Calls into
  // traverseFrom.
  void traverseAllPaths(Fusion* fusion, bool from_outputs_only = false);

  static std::unordered_set<Val*> getInputsTo(const std::vector<Val*>& vals);
};

/*
 * Backward visitor IterVisitor calls handle in reverse order from outputs
 * to inputs It would be really nice to unify this with IterVisitor, however,
 * the challenge there is that we specify traversal from outputs towards inputs
 * because it implicitly provides DCE. However, if users are not careful, they
 * could miss necessary outputs to do a backward traversal.
 *
 * BackwardVisitor checks that all outputs of an Expr is visited before visiting
 * the Expr. If we don't provide nodes to start from on all backward paths of
 * those outputs we will never visit the Expr.
 *
 * The first step of BackwardVisitor is to make sure we've specified enough
 * outputs to guarentee that we will traverse all outputs of all exprs during
 * the backward traversal.
 */
// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
class TORCH_CUDA_CU_API BackwardVisitor : public OptOutDispatch {
 public:
  // NOLINTNEXTLINE(modernize-use-override)
  virtual ~BackwardVisitor() = default;

  BackwardVisitor() = default;

  BackwardVisitor(const BackwardVisitor& other) = default;
  BackwardVisitor& operator=(const BackwardVisitor& other) = default;

  BackwardVisitor(BackwardVisitor&& other) = default;
  BackwardVisitor& operator=(BackwardVisitor&& other) = default;

  // Functions return nodes in reverse order to be added to the to_visit queue
  // These functions will start at outputs and propagate up through the DAG
  // to inputs based on depth first traversal. Next could be called on a node
  // multiple times.
  virtual std::vector<Statement*> next(Statement* stmt);

  virtual std::vector<Statement*> next(Expr* expr);

  virtual std::vector<Statement*> next(Val* val);

  // This handle functions is called on every Statement* in topological order,
  // starting from outputs to inputs.
  // NOLINTNEXTLINE(modernize-use-override,cppcoreguidelines-explicit-virtual-functions)
  virtual void handle(Statement* stmt) override {
    OptOutDispatch::handle(stmt);
  }
  // This handle functions is called on every Expr* in topological order,
  // starting from outputs to inputs.
  // NOLINTNEXTLINE(modernize-use-override,cppcoreguidelines-explicit-virtual-functions)
  virtual void handle(Expr* expr) override {
    OptOutDispatch::handle(expr);
  }
  // This handle functions is called on every Val* in topological order,
  // starting from outputs to inputs.
  // NOLINTNEXTLINE(modernize-use-override,cppcoreguidelines-explicit-virtual-functions)
  virtual void handle(Val* val) override {
    OptOutDispatch::handle(val);
  }

  // All exprs that need to be visited in this traversal. Labeled in topological
  // order (size_t).
  std::unordered_map<Expr*, size_t> traversal_exprs_;

  // The entire stack during traversal. stmt_stack.back().back() is the node
  // that is being called in handle(). stmt_stack.back() contains siblings (not
  // guarenteed to be all siblings throughout traversal). stmt_stack.front()
  // contains the inputs we started with (not guarenteed to be all outputs
  // throughout traversal).
  std::deque<std::deque<Statement*>> stmt_stack_;

  // Starts at nodes provided in from, traverses from these nodes to inputs.
  // Calls handle on all Statement*s in topological sorted order.
  // traverseAllPaths = false only call handle on each Statement* once
  // traverseAllPaths = true traverses all paths from nodes in from to inputs.
  //   Handle on a Statement* for every path from "from" nodes, to inputs.
  void traverseFrom(
      Fusion* fusion,
      const std::vector<Val*>& from,
      bool traverseAllPaths = false);
};

class TORCH_CUDA_CU_API DependencyCheck {
 public:
  // Returns if "dependency" is a dependency of "of".
  static bool isDependencyOf(Val* dependency, Val* of);

  // Finds a Val* path from "of" to "dependency". Returns that path.
  // deque.back() is "of", deque[0] is dependency if a chain exists.
  static std::deque<Val*> getSingleDependencyChain(Val* dependency, Val* of);

  // Finds all Val* paths from "of" to "dependency". Returns those paths.
  // deque[i].back() is "of", and deque[i][0] is "dependency". Returns an
  // empty deque if no dependency found.
  static std::deque<std::deque<Val*>> getAllDependencyChains(
      Val* dependency,
      Val* of);

  // Finds all Val* paths from all leaf nodes to "dependency". Returns those
  // paths. deque[i].back() are leaf nodes, and deque[i][0] is "dependency".
  // Returns an empty deque if there are no uses of dependency found.
  static std::deque<std::deque<Val*>> getAllUseChains(Val* dependency);

  // Grab all values that exist between and including provided vals
  static std::unordered_set<Val*> getAllValsBetween(
      const std::unordered_set<Val*>& dependencies,
      const std::vector<Val*>& of);

  // Return registered outputs of the fusion that are a dependency of any val of
  static std::unordered_set<Val*> getAllOutputsOf(
      const std::unordered_set<Val*>& of);
};

// Expr sort will take a fusion and return a topologically sorted list of
// expressions.
class ExprSort : public IterVisitor {
 private:
  std::vector<Expr*> exprs;

  void handle(Expr* expr) override;

 public:
  static std::vector<Expr*> getExprs(Fusion* fusion, bool from_outputs_only);

  static std::vector<Expr*> getExprs(
      Fusion* fusion,
      const std::vector<Val*>& from);
};

class InputsOf : public IterVisitor {
 private:
  std::unordered_set<Val*> inputs;

  void handle(Val* v) final;

 public:
  static std::unordered_set<Val*> output(Fusion* fusion, Val* output_);
};

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
