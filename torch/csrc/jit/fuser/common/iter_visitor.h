#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>

#include <vector>
#include <stack>

namespace torch {
namespace jit {
namespace fuser {

struct Statement;
struct Val;
struct Expr;

struct TensorDomain;
struct TensorView;
struct IterDomain;
struct Tensor;
struct Float;
struct Int;

struct UnaryOp;
struct BinaryOp;
struct Split;
struct Merge;
struct Reorder;

struct FusionGurad;
struct Fusion;

struct TORCH_API IterVisitor {
  virtual ~IterVisitor() = default;

public:
  IterVisitor() = default;

  IterVisitor(const IterVisitor& other) = default;
  IterVisitor& operator=(const IterVisitor& other) = default;

  IterVisitor(IterVisitor&& other) = default;
  IterVisitor& operator=(IterVisitor&& other) = default;

  //Functions return nodes in reverse order to be added to the to_visit queue
  //These functions will start at outputs and propagate op through the DAG
  //in depth first traversal. Next could be called on nodes multiple times,
  //however, once handle is called on a node next will not be called.
  std::vector<Statement*> next(Statement* stmt);
  std::vector<Statement*> next(Expr* expr);
  std::vector<Statement*> next(Val* v);


  //Hierarchal dispatch functions for handle
  virtual void handle(Statement* s) { Statement::dispatch(this, s); }
  virtual void handle(Expr* e) { Expr::dispatch(this, e); }
  virtual void handle(Val* v) { Val::dispatch(this, v); }

  //Handle functions to be over written by inheriting classes. These functions
  //Will be called on nodes in topological order
  virtual void handle(TensorDomain*) {}
  virtual void handle(TensorView*) {}
  virtual void handle(IterDomain*) {}
  virtual void handle(Tensor*) {}

  virtual void handle(Float*) {}
  virtual void handle(Int*) {}

  virtual void handle(UnaryOp*) {}
  virtual void handle(BinaryOp*) {}
  virtual void handle(Split*) {}
  virtual void handle(Merge*) {}
  virtual void handle(Reorder*) {}

  //Stop condition allows users to stop iteration if a certain condition is met
  virtual bool stopCondition() { return false; }

  //Callback function when a Stmt is added to the "to_visit" queue
  virtual void toVisitCallback(Statement* stmt) {}

  void traverse(
      const Fusion* const _fusion,
      bool from_outputs_only,
      bool breadth_first);

  // Starts at from, traverses backwards through DAG, calls handle on nodes
  // in depth first topological sorted order.
  void traverse(const Fusion* const fusion, std::vector<Val*> from);
};


//Class to check if nodes are in the dependency chain of another node.
struct TORCH_API DependencyCheck : public IterVisitor{
private:
  
  //Class constructor checking if _dependency is a dependency of _of.
  DependencyCheck(Val* _dependency, Val* _of):dependency_{_dependency}, of_{_of}, is_dependency{false}{}

  //Run through nodes in topological order starting from _of looking for _dependency.
  void traverse(const Fusion* const, std::vector<Val*>);

  //when handle is called on val, we know 2 things. Val is a dependency of of.
  //and dep_chain contains the values in between of and dependency.
  void handle(Val* val);

  //When we handle an expr we pop off its outputs from the dep_chain
  void handle(Expr* expr);

  //When we visit an Expr we place its outputs on the dep_chain
  void toVisitCallback(Statement* stmt);

  //Traverse the dep chain from of, return if dependency was found in it
  bool check();

  Val* const dependency_;
  Val* const of_;
  bool is_dependency;
  std::stack<Val*> dep_chain;
 
  //Stop once we've found the dependency
  bool stopCondition() { return is_dependency; }

public:

  //Returns if dependency is a dependency of of.
  static bool isDependencyOf(Val* dependency, Val* of){
    DependencyCheck dp(dependency, of);
    return dp.check();
  }

  //Return the dependency chain, including dependency and of. If no dependency
  //was found, returns an empty stack.
  static std::stack<Val*> getDependencyChain(Val* dependency, Val* of);

};

} // namespace fuser
} // namespace jit
} // namespace torch
