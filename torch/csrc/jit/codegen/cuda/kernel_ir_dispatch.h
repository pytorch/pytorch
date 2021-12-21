#pragma once

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {
namespace kir {

// Hierarchal dispatch functions for handle
class Node;
class Expr;
class Val;

// Vals
class IterDomain;
class TensorDomain;
class TensorView;
class Bool;
class Double;
class Int;
class NamedScalar;
class Predicate;
class TensorIndex;

// Exprs
class UnaryOp;
class BinaryOp;
class TernaryOp;
class ReductionOp;
class WelfordOp;
class BroadcastOp;
class Allocate;
class Sync;
class InitMagicZero;
class UpdateMagicZero;
class ForLoop;
class IfThenElse;
class GridReduction;
class GridBroadcast;
class GridWelford;

// By default, all IR nodes are handled in this dispatch, and will call an empty
// function on all nodes.
class TORCH_CUDA_CU_API OptOutConstDispatch : public PolymorphicBase {
 protected:
  virtual void unhandled(const Node*) {}

 public:
  // Hierarchal dispatch functions for handle
  virtual void handle(const Node*);
  virtual void handle(const Expr*);
  virtual void handle(const Val*);

  // Vals
  virtual void handle(const IterDomain*);
  virtual void handle(const TensorDomain*);
  virtual void handle(const TensorView*);
  virtual void handle(const Bool*);
  virtual void handle(const Double*);
  virtual void handle(const Int*);
  virtual void handle(const NamedScalar*);
  virtual void handle(const Predicate*);
  virtual void handle(const TensorIndex*);

  // Exprs
  virtual void handle(const UnaryOp*);
  virtual void handle(const BinaryOp*);
  virtual void handle(const TernaryOp*);
  virtual void handle(const ReductionOp*);
  virtual void handle(const WelfordOp*);
  virtual void handle(const BroadcastOp*);
  virtual void handle(const Allocate*);
  virtual void handle(const Sync*);
  virtual void handle(const InitMagicZero*);
  virtual void handle(const UpdateMagicZero*);
  virtual void handle(const ForLoop*);
  virtual void handle(const IfThenElse*);
  virtual void handle(const GridReduction*);
  virtual void handle(const GridBroadcast*);
  virtual void handle(const GridWelford*);
};

class TORCH_CUDA_CU_API OptOutDispatch : public PolymorphicBase {
 protected:
  virtual void unhandled(Node*) {}

 public:
  // Hierarchal dispatch functions for handle
  virtual void handle(Node*);
  virtual void handle(Expr*);
  virtual void handle(Val*);

  // Vals

  virtual void handle(IterDomain*);
  virtual void handle(TensorDomain*);
  virtual void handle(TensorView*);
  virtual void handle(Bool*);
  virtual void handle(Double*);
  virtual void handle(Int*);
  virtual void handle(NamedScalar*);
  virtual void handle(Predicate*);
  virtual void handle(TensorIndex*);

  // Exprs
  virtual void handle(UnaryOp*);
  virtual void handle(BinaryOp*);
  virtual void handle(TernaryOp*);
  virtual void handle(ReductionOp*);
  virtual void handle(WelfordOp*);
  virtual void handle(BroadcastOp*);
  virtual void handle(Allocate*);
  virtual void handle(Sync*);
  virtual void handle(InitMagicZero*);
  virtual void handle(UpdateMagicZero*);
  virtual void handle(ForLoop*);
  virtual void handle(IfThenElse*);
  virtual void handle(GridReduction*);
  virtual void handle(GridBroadcast*);
  virtual void handle(GridWelford*);
};

class TORCH_CUDA_CU_API OptInConstDispatch : public OptOutConstDispatch {
 public:
  using OptOutConstDispatch::handle;

 protected:
  virtual void unhandled(const Node* stmt) final;
};

class TORCH_CUDA_CU_API OptInDispatch : public OptOutDispatch {
 public:
  using OptOutDispatch::handle;

 protected:
  virtual void unhandled(Node* stmt) final;
};

// Base visitor class that visits all nodes in provided vector<Expr*>.
//
// Includes visiting through scopes like IfThenElse and ForLoop, and tracks them
// in scopes_ and for_loops_.
//
// Makes a copy of exprs at exprs_ which could be used to modify and return.
//
// When traversing through ITE/FLs it will use a copy
// of the provided expressions to make it safe to insert/delete nodes.
//
// Provides a simple base class to inherit from for typical kir passes
class IrVisitor : public OptOutDispatch {
 public:
  std::vector<Expr*> handle(const std::vector<Expr*>& expr);

 protected:
  using OptOutDispatch::handle;

  virtual void handle(ForLoop*) override;
  virtual void handle(IfThenElse*) override;

 protected:
  std::vector<ForLoop*> for_loops_;
  std::vector<Scope*> scope_;
  std::vector<Expr*> exprs_;
};

// Base Expr Mutator class that visits all nodes with IrVisitor, and then
// inserts new expressions or replaces expressions based on insertion/replace
// maps provided. These replacement maps are expected to accumulate during an
// initial traversal, then runs an insertion based on them after the overloaded
// traversal.
//
// Order of mutations may be important, mutations are ordered according to the
// following rules:
//   Before/After insertions are ordered as registered when reverse_order ==
//   false,
//
//   Before/After insertions are in reverse order as registered when
//   reverse_order == true,
//
//   Before/After insertions are done before Expr replacements, so reference for
//   insertions must be on pre-replaced Exprs
//
// To place in a scope that is empty, simply provide a nullptr reference
// Since insertions are done in order, it's possible to insert an expression in
// an empty scope, and then use that inserted scope as a reference for
// subsequent mutations.
class ExprMutator : public IrVisitor {
 protected:
  std::vector<Expr*> traverseAndInsert(
      const std::vector<Expr*>& expr,
      bool reverse_order = false);

  std::vector<Expr*> mutate(bool reverse_order = false);

  using IrVisitor::handle;
  // Registration function which *don't* need to be called "in place" during
  // visiting.
  void registerInsertBefore(Expr* reference, Expr* new_expr, Scope* scope);
  void registerInsertAfter(Expr* reference, Expr* new_expr, Scope* scope);
  void registerReplace(Expr* reference, Expr* new_expr, Scope* scope);

  // Registration function which need to be called "in place" during visiting.
  // I.E.
  // if you want to insert before/after or replace an Expr, you must register
  // when in handle(Expr*) of that expr.
  void registerInsertBefore(Expr* reference, Expr* new_expr);
  void registerInsertAfter(Expr* reference, Expr* new_expr);
  void registerReplace(Expr* reference, Expr* new_expr);

 private:
  enum class MutationMode { BEFORE, AFTER, REPLACE };

  void registerMutation(
      Expr* ref,
      Expr* new_expr,
      Scope* scope,
      MutationMode mode);

  struct MutationInformation {
    Expr* reference = nullptr;
    Expr* new_expr = nullptr;
    Scope* scope = nullptr;
    MutationMode mode = MutationMode::BEFORE;
  };

  // Track insertions as they're registered
  std::vector<MutationInformation> insertions_;

  // Track replacements as they're registered
  std::vector<MutationInformation> replacements_;
};

} // namespace kir
} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
