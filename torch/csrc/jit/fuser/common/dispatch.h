#pragma once

namespace torch {
namespace jit {
namespace fuser {

// Hierarchal dispatch functions for handle
struct Statement;
struct Expr;
struct Val;

// Vals
struct IterDomain;
struct TensorDomain;
struct Tensor;
struct TensorView;
struct Float;
struct Int;

// Exprs
struct Split;
struct Merge;
struct Reorder;
struct UnaryOp;
struct BinaryOp;
struct ForLoop;
struct IfThenElse;

/*
 * By default, all IR nodes are handled in this dispatch, and will call an empty
 * function on all nodes.
 */
struct TORCH_API OptOutDispatch {
  virtual ~OptOutDispatch() = default;
  OptOutDispatch() = default;

  OptOutDispatch(const OptOutDispatch& other) = default;
  OptOutDispatch& operator=(const OptOutDispatch& other) = default;

  OptOutDispatch(OptOutDispatch&& other) = default;
  OptOutDispatch& operator=(OptOutDispatch&& other) = default;

  // Hierarchal dispatch functions for handle
  virtual void handle(Statement*);
  virtual void handle(Expr*);
  virtual void handle(Val*);

  // Vals
  virtual void handle(IterDomain*) { }
  virtual void handle(TensorDomain*) { }
  virtual void handle(Tensor*) { }
  virtual void handle(TensorView*) { }
  virtual void handle(Float*) { }
  virtual void handle(Int*) { }

  // Exprs
  virtual void handle(Split*) { }
  virtual void handle(Merge*) { }
  virtual void handle(Reorder*) { }
  virtual void handle(UnaryOp*) { }
  virtual void handle(BinaryOp*) { }
  virtual void handle(ForLoop*) { }
  virtual void handle(IfThenElse*) { }
};

struct TORCH_API OptInConstDispatch {
  virtual ~OptInConstDispatch() = default;
  OptInConstDispatch() = default;

  OptInConstDispatch(const OptInConstDispatch& other) = default;
  OptInConstDispatch& operator=(const OptInConstDispatch& other) = default;

  OptInConstDispatch(OptInConstDispatch&& other) = default;
  OptInConstDispatch& operator=(OptInConstDispatch&& other) = default;

  // Hierarchal dispatch functions for handle
  virtual void handle(const Statement* const);
  virtual void handle(const Expr* const);
  virtual void handle(const Val* const);

  // Vals
  virtual void handle(const IterDomain* const) { AT_ERROR("Handle not overriden for IterDomain."); }
  virtual void handle(const TensorDomain* const) { AT_ERROR("Handle not overriden for TensorDomain."); }
  virtual void handle(const Tensor* const) { AT_ERROR("Handle not overriden for Tensor."); }
  virtual void handle(const TensorView* const) { AT_ERROR("Handle not overriden for TensorView."); }
  virtual void handle(const Float* const) { AT_ERROR("Handle not overriden for Float."); }
  virtual void handle(const Int* const) { AT_ERROR("Handle not overriden for Int."); }

  // Exprs
  virtual void handle(const Split* const) { AT_ERROR("Handle not overriden for Split."); }
  virtual void handle(const Merge* const) { AT_ERROR("Handle not overriden for Merge."); }
  virtual void handle(const Reorder* const) { AT_ERROR("Handle not overriden for Reorder."); }
  virtual void handle(const UnaryOp* const) { AT_ERROR("Handle not overriden for UnaryOp."); }
  virtual void handle(const BinaryOp* const) { AT_ERROR("Handle not overriden for BinaryOp."); }
  virtual void handle(const ForLoop* const) { AT_ERROR("Handle not overriden for ForLoop."); }
  virtual void handle(const IfThenElse* const) { AT_ERROR("Handle not overriden for IfThenElse."); }
};

struct TORCH_API OptInDispatch {
  virtual ~OptInDispatch() = default;
  OptInDispatch() = default;

  OptInDispatch(const OptInDispatch& other) = default;
  OptInDispatch& operator=(const OptInDispatch& other) = default;

  OptInDispatch(OptInDispatch&& other) = default;
  OptInDispatch& operator=(OptInDispatch&& other) = default;

  // Hierarchal dispatch functions for handle
  virtual void handle(Statement* s);
  virtual void handle(Expr* e);
  virtual void handle(Val* v);

  // Vals
  virtual void handle(IterDomain*) { AT_ERROR("Handle not overriden for IterDomain."); }
  virtual void handle(TensorDomain*) { AT_ERROR("Handle not overriden for TensorDomain."); }
  virtual void handle(Tensor*) { AT_ERROR("Handle not overriden for Tensor."); }
  virtual void handle(TensorView*) { AT_ERROR("Handle not overriden for TensorView."); }
  virtual void handle(Float*) { AT_ERROR("Handle not overriden for Float."); }
  virtual void handle(Int*) { AT_ERROR("Handle not overriden for Int."); }

  // Exprs
  virtual void handle(Split*) { AT_ERROR("Handle not overriden for Split."); }
  virtual void handle(Merge*) { AT_ERROR("Handle not overriden for Merge."); }
  virtual void handle(Reorder*) { AT_ERROR("Handle not overriden for Reorder."); }
  virtual void handle(UnaryOp*) { AT_ERROR("Handle not overriden for UnaryOp."); }
  virtual void handle(BinaryOp*) { AT_ERROR("Handle not overriden for BinaryOp."); }
  virtual void handle(ForLoop*) { AT_ERROR("Handle not overriden for ForLoop."); }
  virtual void handle(IfThenElse*) { AT_ERROR("Handle not overriden for IfThenElse."); }
};

struct TORCH_API OptOutMutator {
virtual ~OptOutMutator() = default;
  OptOutMutator() = default;

  OptOutMutator(const OptOutMutator& other) = default;
  OptOutMutator& operator=(const OptOutMutator& other) = default;

  OptOutMutator(OptOutMutator&& other) = default;
  OptOutMutator& operator=(OptOutMutator&& other) = default;

  // Hierarchal dispatch functions for handle
  virtual Statement* mutate(Statement* s);
  virtual Statement* mutate(Expr* e);
  virtual Statement* mutate(Val* v);

  // Vals
  virtual Statement* mutate(IterDomain*);
  virtual Statement* mutate(TensorDomain*);
  virtual Statement* mutate(Tensor*);
  virtual Statement* mutate(TensorView*);
  virtual Statement* mutate(Float*);
  virtual Statement* mutate(Int*);

  // Exprs
  virtual Statement* mutate(Split*);
  virtual Statement* mutate(Merge*);
  virtual Statement* mutate(Reorder*);
  virtual Statement* mutate(UnaryOp*);
  virtual Statement* mutate(BinaryOp*);
  virtual Statement* mutate(ForLoop*);
  virtual Statement* mutate(IfThenElse*);
};

struct TORCH_API OptInMutator {
virtual ~OptInMutator() = default;
  OptInMutator() = default;

  OptInMutator(const OptInMutator& other) = default;
  OptInMutator& operator=(const OptInMutator& other) = default;

  OptInMutator(OptInMutator&& other) = default;
  OptInMutator& operator=(OptInMutator&& other) = default;

  // Hierarchal dispatch functions for mutate
  virtual Statement* mutate(Statement*);
  virtual Statement* mutate(Expr*);
  virtual Statement* mutate(Val*);

  // Vals
  virtual Statement* mutate(IterDomain*) { AT_ERROR("Mutate not overriden for IterDomain."); }
  virtual Statement* mutate(TensorDomain*) { AT_ERROR("Mutate not overriden for TensorDomain."); }
  virtual Statement* mutate(Tensor*) { AT_ERROR("Mutate not overriden for Tensor."); }
  virtual Statement* mutate(TensorView*) { AT_ERROR("Mutate not overriden for TensorView."); }
  virtual Statement* mutate(Float*) { AT_ERROR("Mutate not overriden for Float."); }
  virtual Statement* mutate(Int*) { AT_ERROR("Mutate not overriden for Int."); }

  // Exprs
  virtual Statement* mutate(Split*) { AT_ERROR("Mutate not overriden for Split."); }
  virtual Statement* mutate(Merge*) { AT_ERROR("Mutate not overriden for Merge."); }
  virtual Statement* mutate(Reorder*) { AT_ERROR("Mutate not overriden for Reorder."); }
  virtual Statement* mutate(UnaryOp*) { AT_ERROR("Mutate not overriden for UnaryOp."); }
  virtual Statement* mutate(BinaryOp*) { AT_ERROR("Mutate not overriden for BinaryOp."); }
  virtual Statement* mutate(ForLoop*) { AT_ERROR("Mutate not overriden for ForLoop."); }
  virtual Statement* mutate(IfThenElse*) { AT_ERROR("Mutate not overriden for IfThenElse."); }
};

} // namespace fuser
} // namespace jit
} // namespace torch