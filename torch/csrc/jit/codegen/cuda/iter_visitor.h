#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>

#include <torch/csrc/jit/codegen/cuda/dispatch.h>

#include <stack>
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
 * IterVisitor walks a Fusion topologically ordered from outputs of the fusion.
 * By default outputs are any leaf Vars that don't have uses, but can be set to
 * registered outputs of the Fusion. On every node handle(NodeType*) will be
 * called (topologically ordered).
 *
 * stopCondition can be overridden if it is desired to stop the traversal at any
 * particular point. toVisitCallback can also be overridden and will be called
 * when a node is added to the to_visit queue. The use of these two functions
 * can be seen in DependencyCheck which uses them to find if a value is in the
 * dependency chain of another value. stopCondition is called when the value is
 * found to stop traversal. toVisitCallback is used to maintain a dependency
 * stack.
 */
struct TORCH_CUDA_API IterVisitor : public OptOutDispatch {
  virtual ~IterVisitor() = default;

  using OptOutDispatch::handle;

  IterVisitor() = default;

  IterVisitor(const IterVisitor& other) = default;
  IterVisitor& operator=(const IterVisitor& other) = default;

  IterVisitor(IterVisitor&& other) = default;
  IterVisitor& operator=(IterVisitor&& other) = default;

  // Functions return nodes in reverse order to be added to the to_visit queue
  // These functions will start at outputs and propagate op through the DAG
  // in depth first traversal. Next could be called on nodes multiple times,
  // however, once handle is called on a node next will not be called.
  std::vector<Statement*> next(Statement* stmt);
  std::vector<Statement*> next(Expr* expr);
  std::vector<Statement*> next(Val* v);

  virtual void handle(Statement* s) {
    OptOutDispatch::handle(s);
  }
  virtual void handle(Expr* e) {
    OptOutDispatch::handle(e);
  }
  virtual void handle(Val* v) {
    OptOutDispatch::handle(v);
  }

  // Stop condition allows users to stop iteration if a certain condition is met
  virtual bool stopCondition() {
    return false;
  }

  // Callback function when a Stmt is added to the "to_visit" queue
  virtual void toVisitCallback(Statement* stmt) {}

 public:
  // This version of traverse collects the points of the graph to start from
  // The "from_outputs_only" argument forces the graph to start from outputs
  // instead of search for Val typed nodes that have no uses.
  // The output type set limits further the set of Val nodes to search by type.
  void traverse(
      Fusion* const fusion,
      bool from_outputs_only = false,
      bool breadth_first = false);

  // Starts at from, traverses backwards through DAG, calls handle on nodes
  // in depth first topological sorted order.
  void traverseFrom(Fusion* const fusion, const std::vector<Val*>& from);
};

// Class to check if nodes are in the dependency chain of another node.
struct TORCH_CUDA_API DependencyCheck : public IterVisitor {
 private:
  // Class constructor checking if _dependency is a dependency of _of.
  DependencyCheck(Val* _dependency, Val* _of)
      : dependency_{_dependency}, of_{_of}, is_dependency{false} {}

  // when handle is called on val, we know 2 things. Val is a dependency of of.
  // and dep_chain contains the values in between of and dependency.
  void handle(Val* val) override;

  // When we handle an expr we pop off its outputs from the dep_chain
  void handle(Expr* expr) override;

  // When we visit an Expr we place its outputs on the dep_chain
  void toVisitCallback(Statement* stmt);

  // Traverse the dep chain from of, return if dependency was found in it
  bool check();

  Val* const dependency_;
  Val* const of_;
  bool is_dependency;
  std::stack<Val*> dep_chain;

  // Stop once we've found the dependency
  bool stopCondition() {
    return is_dependency;
  }

 public:
  // Returns if dependency is a dependency of of.
  static bool isDependencyOf(Val* dependency, Val* of) {
    DependencyCheck dp(dependency, of);
    return dp.check();
  }

  // Return the dependency chain, including dependency and of. If no dependency
  // was found, returns an empty stack.
  static std::stack<Val*> getDependencyChain(Val* dependency, Val* of);
};

} // namespace fuser
} // namespace jit
} // namespace torch
