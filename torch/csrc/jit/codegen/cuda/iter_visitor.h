#pragma once

#include <c10/macros/Export.h>

#include <torch/csrc/jit/codegen/cuda/dispatch.h>
#include <torch/csrc/jit/codegen/cuda/type.h>

#include <deque>
#include <unordered_set>
#include <vector>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

class Fusion;
class Statement;
class Expr;
class Val;

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
  ~IterVisitor() override = default;

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
  virtual std::vector<Statement*> next(Statement* stmt);

  virtual std::vector<Statement*> next(Val* v);

  virtual std::vector<Statement*> next(Expr* expr);

  // This handle functions is called on every Statement* in topological order,
  // starting from outputs to inputs.
  void handle(Statement* s) override;

  // This handle functions is called on every Expr* in topological order,
  // starting from outputs to inputs.
  void handle(Expr* e) override;

  // This handle functions is called on every Val* in topological order,
  // starting from outputs to inputs.
  void handle(Val* v) override;

  // The entire stack during traversal. stmt_stack.back().back() is the node
  // that is being called in handle(). stmt_stack.back() contains siblings (not
  // guarenteed to be all siblings throughout traversal). stmt_stack.front()
  // contains the outputs we started with (not guarenteed to be all outputs
  // throughout traversal).
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  std::vector<std::vector<Statement*>> stmt_stack;

  void traverseHelper(Fusion* fusion, bool traverse_all_paths = false);

 public:
  //! Traverses nodes in Fusion from inputs in topological order to "to". i.e.
  //! from inputs towards outputs.
  //! \param traverseAllPaths = false only call handle on each Statement* once
  //!    traverseAllPaths = true traverses all paths between expressions/values.
  //!    Calls handle on a Statement* for every path from inputs to "to".
  //! \param traverseIntoMembers = When hitting nodes like TensorView,
  //! TensorDomain, or IterDomain where there are members of the nodes that are
  //! Val's a value of "true" will also traverse into those member Val's, a
  //! value of "false" will not traverse into the members.
  void traverseTo(
      Fusion* fusion,
      const std::vector<Val*>& to,
      bool traverse_all_paths = false,
      bool traverse_into_members = false);

  //! Traverses nodes in Fusion from inputs in topological order to "to". i.e.
  //! from inputs towards outputs.
  //! \param traverseAllPaths = false only call handle on each Statement* once
  //!    traverseAllPaths = true traverses all paths between expressions/values.
  //!    Calls handle on a Statement* for every path from inputs to "to".
  //! \param traverseIntoMembers = When hitting nodes like TensorView,
  //! TensorDomain, or IterDomain where there are members of the nodes that are
  //! Val's a value of "true" will also traverse into those member Val's, a
  //! value of "false" will not traverse into the members.
  //! \param from: Specified values to start traversing. If a "from" Val is not
  //! on path from inputs to "to" node it will not be visited. If there's a path
  //! from inputs to "to" that doesn't go through "from" that input and the path
  //! from it will also be traversed.
  void traverseBetween(
      Fusion* fusion,
      const std::unordered_set<Val*>& from,
      const std::vector<Val*>& to,
      bool traverse_all_paths = false,
      bool traverse_into_members = false);

  // Iterates from terminating outputs registered with the fusion. Terminating
  // means value is not used to generate any other value used in producing
  // registered outputs.
  void traverse(Fusion* fusion);

  // Same as traverse put it traverses every edge, meaning it will traverse
  // values more than once.
  void traverseAllPaths(Fusion* fusion);

  //! Get inputs to vals. Possible input vals can be optionally
  //! given. If not, vals with no producers are returned.
  //
  // TODO: This doesn't seem to fit with IterVisitor. Should probably be moved
  // out of the class.
  static std::vector<Val*> getInputsTo(
      const std::vector<Val*>& vals,
      const std::vector<Val*>& inputs = {});
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
 * the backward traversal. In the case where we don't require visiting all
 * outputs of some exprs, example being the `N` output of welford ops.
 * `must_cover_all_expr_outputs` is added to disable the check, and in
 * this case the visitor pass need be aware
 *  1. Exprs with any output that has a use chain that ends with a final
 * consumer in the `from` list `will be` visited.
 *  2. Vals that doesn't have a use chain that ends with a final
 * consumer in the `from` list `will not be` visited, even though its
 * definition expr might be visited. An example is if the `N` output
 * of an welford op is unused, but other outputs are, the welford op
 * will be visited but the `N` output will not.
 *
 */
// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
class TORCH_CUDA_CU_API BackwardVisitor : public OptOutDispatch {
 protected:
  // NOLINTNEXTLINE(modernize-use-override)
  virtual ~BackwardVisitor() = default;

  BackwardVisitor(bool must_cover_all_expr_outputs = true)
      : must_cover_all_expr_outputs_(must_cover_all_expr_outputs) {}

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
  virtual void handle(Statement* stmt) override;

  // This handle functions is called on every Expr* in topological order,
  // starting from outputs to inputs.
  // NOLINTNEXTLINE(modernize-use-override,cppcoreguidelines-explicit-virtual-functions)
  virtual void handle(Expr* expr) override;

  // This handle functions is called on every Val* in topological order,
  // starting from outputs to inputs.
  // NOLINTNEXTLINE(modernize-use-override,cppcoreguidelines-explicit-virtual-functions)
  virtual void handle(Val* val) override;

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
  void traverseTo(
      Fusion* fusion,
      const std::vector<Val*>& from,
      bool traverseAllPaths = false);

  bool must_cover_all_expr_outputs_ = true;
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

  // Grab all values that exist between and including provided
  // vals. Returned values are topologicaly ordered, and unique.
  static std::vector<Val*> getAllValsBetween(
      const std::unordered_set<Val*>& dependencies,
      const std::vector<Val*>& of);

  // Returns all dependent exprs that exist between
  //  the provided vals
  static std::vector<Expr*> getAllExprsBetween(
      const std::unordered_set<Val*>& dependencies,
      const std::vector<Val*>& of);

  // Return registered outputs of the fusion that are a dependency of any val of
  static std::unordered_set<Val*> getAllOutputsOf(
      const std::unordered_set<Val*>& of);

  // Return all Vals that depend on the given Vals
  static std::unordered_set<Val*> getAllDependentVals(
      const std::unordered_set<Val*>& of);
};

// Expr sort will take a fusion and return a topologically sorted list of
// expressions.
class StmtSort : public IterVisitor {
 protected:
  StmtSort() = default;

  std::vector<Statement*> stmts;

  void handle(Statement* stmt) override;

 public:
  // If traverse_members it will also extract all member nodes in the sorted
  // statement list in the fusion. i.e. all IterDomains, extents, and associated
  // expressions of them
  static std::vector<Statement*> getStmts(
      Fusion* fusion,
      bool traverse_members = false);

  // Returns ordered Statements required to produce from, including from.
  static std::vector<Statement*> getStmts(
      Fusion* fusion,
      const std::vector<Val*>& to,
      bool traverse_members = false);

  // Returns ordered Statements required to produce from, including from.
  // Stops traversal once hiting any Statements in to. Includes Statements in
  // to.
  //
  // Warning: this doesn't necessarily prevent statements before `to` from being
  // returned. e.g.
  // i1 = i0
  // i2 = i1
  // i3 = i2
  // i4 = i3 + i1
  // getExprs(fusion, {i4}, {i3})
  // will return the definition and values {i0, i1, i4}
  // i3 is dependent on i1, but since i4 also is then the traversal will go down
  // the i4->i1->i0 path, even though the i4->i3-//>i2->i1 path is blocked.
  //
  // If traverse_members it will also extract all member nodes in the sorted
  // expr list in the fusion. i.e. all expressions on IterDomains, extents, etc
  static std::vector<Statement*> getStmtsBetween(
      Fusion* fusion,
      const std::vector<Val*>& from,
      const std::vector<Val*>& to,
      bool traverse_members = false);

  // Same as getStmts version but filters to only return the Expr*s
  static std::vector<Expr*> getExprs(
      Fusion* fusion,
      bool traverse_members = false);

  // Same as getStmts version but filters to only return the Expr*s
  static std::vector<Expr*> getExprs(
      Fusion* fusion,
      const std::vector<Val*>& to,
      bool traverse_members = false);

  // Same as getStmts version but filters to only return the Expr*s
  static std::vector<Expr*> getExprsBetween(
      Fusion* fusion,
      const std::vector<Val*>& from,
      const std::vector<Val*>& to,
      bool traverse_members = false);
};

class InputsOf : public IterVisitor {
 private:
  std::unordered_set<Val*> grabbed_inputs;
  std::vector<Val*> ordered_inputs;

  void handle(Val* v) final;

 public:
  static std::vector<Val*> output(Fusion* fusion, Val* output_);
  static std::vector<Val*> outputs(
      Fusion* fusion,
      const std::vector<Val*>& outputs_);
};

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
