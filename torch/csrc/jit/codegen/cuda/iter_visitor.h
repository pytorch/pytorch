#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>

#include <torch/csrc/jit/codegen/cuda/dispatch.h>

#include <deque>
#include <set>
#include <vector>

namespace torch {
namespace jit {
namespace fuser {

struct Statement;
struct Val;
struct Expr;

struct Fusion;

enum class ValType;

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
struct TORCH_CUDA_API IterVisitor : public OptOutDispatch {
  virtual ~IterVisitor() = default;

  IterVisitor() = default;

  IterVisitor(const IterVisitor& other) = default;
  IterVisitor& operator=(const IterVisitor& other) = default;

  IterVisitor(IterVisitor&& other) = default;
  IterVisitor& operator=(IterVisitor&& other) = default;

  // Functions return nodes in reverse order to be added to the to_visit queue
  // These functions will start at outputs and propagate up through the DAG
  // to inputs based on depth first traversal. Next could be called on a node
  // multiple times.
  virtual std::vector<Statement*> next(Statement* stmt);
  virtual std::vector<Statement*> next(Expr* expr);
  virtual std::vector<Statement*> next(Val* v);

  // This handle functions is called on every Statement* in topological order,
  // starting from outputs to inputs.
  virtual void handle(Statement* s) override {
    OptOutDispatch::handle(s);
  }
  // This handle functions is called on every Expr* in topological order,
  // starting from outputs to inputs.
  virtual void handle(Expr* e) override {
    OptOutDispatch::handle(e);
  }
  // This handle functions is called on every Val* in topological order,
  // starting from outputs to inputs.
  virtual void handle(Val* v) override {
    OptOutDispatch::handle(v);
  }

  // The entire stack during traversal. stmt_stack.back().back() is the node
  // that is being called in handle(). stmt_stack.back() contains siblings (not
  // guarenteed to be all siblings throughout traversal). stmt_stack.front()
  // contains the outputs we started with (not guarenteed to be all outputs
  // throughout traversal).
  std::vector<std::vector<Statement*>> stmt_stack;

  void traverse_(
      Fusion* const fusion,
      bool from_outputs_only = false,
      bool breadth_first = false,
      bool traverse_all_paths = false);

 public:
  // Starts at nodes provided in from, traverses from these nodes to inputs.
  // Calls handle on all Statement*s in topological sorted order.
  // traverseAllPaths = false only call handle on each Statement* once
  // traverseAllPaths = true traverses all paths from nodes in from to inputs.
  //   Handle on a Statement* for every path from "from" nodes, to inputs.
  void traverseFrom(
      Fusion* const fusion,
      const std::vector<Val*>& from,
      bool traverseAllPaths = false);

  // from_outputs_only = true start from outputs registered with fusion,
  // from_outputs_only = false start from all leaf nodes,
  // bool breadth_first = true is not implemented yet
  void traverse(
      Fusion* const fusion,
      bool from_outputs_only = false,
      bool breadth_first = false);

  // from_outputs_only = true start from outputs registered with fusion,
  // from_outputs_only = false start from all leaf nodes,
  // bool breadth_first = true is not implemented yet
  void traverseAllPaths(
      Fusion* const fusion,
      bool from_outputs_only = false,
      bool breadth_first = false);
};

struct TORCH_CUDA_API DependencyCheck {
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
  static std::deque<std::deque<Val*>> getAllDependencyChainsTo(Val* dependency);

  static std::vector<Val*> getTerminatingOutputs(Fusion* const);
};

} // namespace fuser
} // namespace jit
} // namespace torch
