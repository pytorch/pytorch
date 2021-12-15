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
  virtual void handle(const IterDomain* stmt);
  virtual void handle(const TensorDomain* stmt);
  virtual void handle(const TensorView* stmt);
  virtual void handle(const Bool* stmt);
  virtual void handle(const Double* stmt);
  virtual void handle(const Int* stmt);
  virtual void handle(const NamedScalar* stmt);
  virtual void handle(const Predicate* stmt);
  virtual void handle(const TensorIndex* stmt);

  // Exprs
  virtual void handle(const UnaryOp* stmt);
  virtual void handle(const BinaryOp* stmt);
  virtual void handle(const TernaryOp* stmt);
  virtual void handle(const ReductionOp* stmt);
  virtual void handle(const WelfordOp* stmt);
  virtual void handle(const BroadcastOp* stmt);
  virtual void handle(const Allocate* stmt);
  virtual void handle(const Sync* stmt);
  virtual void handle(const InitMagicZero* stmt);
  virtual void handle(const UpdateMagicZero* stmt);
  virtual void handle(const ForLoop* stmt);
  virtual void handle(const IfThenElse* stmt);
  virtual void handle(const GridReduction* stmt);
  virtual void handle(const GridBroadcast* stmt);
  virtual void handle(const GridWelford* stmt);
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

  virtual void handle(IterDomain* stmt);
  virtual void handle(TensorDomain* stmt);
  virtual void handle(TensorView* stmt);
  virtual void handle(Bool* stmt);
  virtual void handle(Double* stmt);
  virtual void handle(Int* stmt);
  virtual void handle(NamedScalar* stmt);
  virtual void handle(Predicate* stmt);
  virtual void handle(TensorIndex* stmt);

  // Exprs
  virtual void handle(UnaryOp* stmt);
  virtual void handle(BinaryOp* stmt);
  virtual void handle(TernaryOp* stmt);
  virtual void handle(ReductionOp* stmt);
  virtual void handle(WelfordOp* stmt);
  virtual void handle(BroadcastOp* stmt);
  virtual void handle(Allocate* stmt);
  virtual void handle(Sync* stmt);
  virtual void handle(InitMagicZero* stmt);
  virtual void handle(UpdateMagicZero* stmt);
  virtual void handle(ForLoop* stmt);
  virtual void handle(IfThenElse* stmt);
  virtual void handle(GridReduction* stmt);
  virtual void handle(GridBroadcast* stmt);
  virtual void handle(GridWelford* stmt);
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

} // namespace kir
} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
