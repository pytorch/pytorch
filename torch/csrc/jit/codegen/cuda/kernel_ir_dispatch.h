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
class KirVisitor : public OptOutDispatch {
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

} // namespace kir
} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
