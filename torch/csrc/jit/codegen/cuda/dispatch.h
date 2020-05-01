#pragma once

#include <c10/util/Exception.h>
#include <torch/csrc/WindowsTorchApiMacro.h>

#include <unordered_map>

/*
 * dispatch.h prevents the need from adding manual dispatch in every class that
 * wants to define how to process a series of nodes. dispatch.h provides 4
 * classes that can be inherited providing a means to override functions on a
 * per-node basis. There are currently 4 provided dispatch mechanisms:
 *
 * OptOutDispatch:
 *
 * provides the functions:
 * virtual void handle(ValType* irnode){}
 *
 * This provides a mechanisms to override this handle for particular node
 * types. For example if we only wanted to actually run a function on
 * BinaryOps, we could inherit OptOutDispatch and simply override: void
 * handle(BinaryOp*) { doSomething; } Then we could run through all our
 * Statement* and call OptOutDispatch::handle(statement). When a BinaryOp is
 * encountered our override function will be called. For every other node,
 * nothing will be done.
 *
 * OptInDispatch:
 *
 * This class is similar to OptOutDispatch, however if we encounter a node
 * that we haven't specified an override for in the derived class, an error
 * will be thrown. This is useful if we create a class that is expected to
 * handle any type of node it encounters.
 *
 * OptOutMutator:
 *
 * This class is similar to OptOutDispatch except the functions provided are of
 * type: virtual Statement* mutate(Statement*) this is useful for when we want
 * to have an IR node result from our overloaded functions.
 *
 * OptInMutator:
 *
 * This class is similar to OptInDispatch except the functions provided are of
 * type: virtual Statement* mutate(Statement*) this is useful for when we want
 * to have an IR node result from our overloaded functions.
 */

namespace torch {
namespace jit {
namespace fuser {

struct Fusion;

// Hierarchal dispatch functions for handle
struct Statement;
struct Expr;
struct Val;

// Vals
struct IterDomain;
struct TensorDomain;
struct TensorView;
struct TensorIndex;
struct Float;
struct Int;
struct NamedScalar;

// Exprs
struct Split;
struct Merge;
struct Reorder;
struct UnaryOp;
struct BinaryOp;
struct ForLoop;
struct IfThenElse;
struct Allocate;

/*
 * By default, all IR nodes are handled in this dispatch, and will call an empty
 * function on all nodes.
 */
struct TORCH_CUDA_API OptOutConstDispatch {
  virtual ~OptOutConstDispatch() = default;
  OptOutConstDispatch() = default;

  OptOutConstDispatch(const OptOutConstDispatch& other) = default;
  OptOutConstDispatch& operator=(const OptOutConstDispatch& other) = default;

  OptOutConstDispatch(OptOutConstDispatch&& other) = default;
  OptOutConstDispatch& operator=(OptOutConstDispatch&& other) = default;

  // Hierarchal dispatch functions for handle
  virtual void handle(const Statement* const);
  virtual void handle(const Expr* const);
  virtual void handle(const Val* const);

  // Vals
  virtual void handle(const IterDomain* const) {}
  virtual void handle(const TensorDomain* const) {}
  virtual void handle(const TensorView* const) {}
  virtual void handle(const TensorIndex* const) {}
  virtual void handle(const Float* const) {}
  virtual void handle(const Int* const) {}
  virtual void handle(const NamedScalar* const) {}

  // Exprs
  virtual void handle(const Split* const) {}
  virtual void handle(const Merge* const) {}
  virtual void handle(const Reorder* const) {}
  virtual void handle(const UnaryOp* const) {}
  virtual void handle(const BinaryOp* const) {}
  virtual void handle(const ForLoop* const) {}
  virtual void handle(const IfThenElse* const) {}
  virtual void handle(const Allocate* const) {}
};

struct TORCH_CUDA_API OptOutDispatch {
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
  virtual void handle(IterDomain*) {}
  virtual void handle(TensorDomain*) {}
  virtual void handle(TensorView*) {}
  virtual void handle(TensorIndex*) {}
  virtual void handle(Float*) {}
  virtual void handle(Int*) {}
  virtual void handle(NamedScalar*) {}

  // Exprs
  virtual void handle(Split*) {}
  virtual void handle(Merge*) {}
  virtual void handle(Reorder*) {}
  virtual void handle(UnaryOp*) {}
  virtual void handle(BinaryOp*) {}
  virtual void handle(ForLoop*) {}
  virtual void handle(IfThenElse*) {}
  virtual void handle(Allocate*) {}
};

struct TORCH_CUDA_API OptInConstDispatch {
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
  virtual void handle(const IterDomain* const) {
    TORCH_INTERNAL_ASSERT(false, "Handle not overriden for IterDomain.");
  }
  virtual void handle(const TensorDomain* const) {
    TORCH_INTERNAL_ASSERT(false, "Handle not overriden for TensorDomain.");
  }
  virtual void handle(const TensorView* const) {
    TORCH_INTERNAL_ASSERT(false, "Handle not overriden for TensorView.");
  }
  virtual void handle(const TensorIndex* const) {
    AT_ERROR("Handle not overriden for TensorIndex.");
  }
  virtual void handle(const Float* const) {
    TORCH_INTERNAL_ASSERT(false, "Handle not overriden for Float.");
  }
  virtual void handle(const Int* const) {
    TORCH_INTERNAL_ASSERT(false, "Handle not overriden for Int.");
  }
  virtual void handle(const NamedScalar* const) {
    AT_ERROR("Handle not overriden for NamedScalar.");
  }

  // Exprs
  virtual void handle(const Split* const) {
    TORCH_INTERNAL_ASSERT(false, "Handle not overriden for Split.");
  }
  virtual void handle(const Merge* const) {
    TORCH_INTERNAL_ASSERT(false, "Handle not overriden for Merge.");
  }
  virtual void handle(const Reorder* const) {
    TORCH_INTERNAL_ASSERT(false, "Handle not overriden for Reorder.");
  }
  virtual void handle(const UnaryOp* const) {
    TORCH_INTERNAL_ASSERT(false, "Handle not overriden for UnaryOp.");
  }
  virtual void handle(const BinaryOp* const) {
    TORCH_INTERNAL_ASSERT(false, "Handle not overriden for BinaryOp.");
  }
  virtual void handle(const ForLoop* const) {
    AT_ERROR("Handle not overriden for ForLoop.");
  }
  virtual void handle(const Allocate* const) {
    AT_ERROR("Handle not overriden for Allocate.");
  }
  virtual void handle(const IfThenElse* const) {
    AT_ERROR("Handle not overriden for IfThenElse.");
  }
};

struct TORCH_CUDA_API OptInDispatch {
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
  virtual void handle(IterDomain*) {
    TORCH_INTERNAL_ASSERT(false, "Handle not overriden for IterDomain.");
  }
  virtual void handle(TensorDomain*) {
    TORCH_INTERNAL_ASSERT(false, "Handle not overriden for TensorDomain.");
  }
  virtual void handle(TensorView*) {
    TORCH_INTERNAL_ASSERT(false, "Handle not overriden for TensorView.");
  }
  virtual void handle(TensorIndex*) {
    AT_ERROR("Handle not overriden for TensorIndex.");
  }
  virtual void handle(Float*) {
    TORCH_INTERNAL_ASSERT(false, "Handle not overriden for Float.");
  }
  virtual void handle(Int*) {
    TORCH_INTERNAL_ASSERT(false, "Handle not overriden for Int.");
  }
  virtual void handle(NamedScalar*) {
    AT_ERROR("Handle not overriden for NamedScalar.");
  }

  // Exprs
  virtual void handle(Split*) {
    TORCH_INTERNAL_ASSERT(false, "Handle not overriden for Split.");
  }
  virtual void handle(Merge*) {
    TORCH_INTERNAL_ASSERT(false, "Handle not overriden for Merge.");
  }
  virtual void handle(Reorder*) {
    TORCH_INTERNAL_ASSERT(false, "Handle not overriden for Reorder.");
  }
  virtual void handle(UnaryOp*) {
    TORCH_INTERNAL_ASSERT(false, "Handle not overriden for UnaryOp.");
  }
  virtual void handle(BinaryOp*) {
    TORCH_INTERNAL_ASSERT(false, "Handle not overriden for BinaryOp.");
  }
  virtual void handle(ForLoop*) {
    TORCH_INTERNAL_ASSERT(false, "Handle not overriden for ForLoop.");
  }
  virtual void handle(Allocate*) {
    AT_ERROR("Handle not overriden for Allocate.");
  }
  virtual void handle(IfThenElse*) {
    TORCH_INTERNAL_ASSERT(false, "Handle not overriden for IfThenElse.");
  }
};

struct TORCH_CUDA_API OptOutMutator {
  virtual ~OptOutMutator() = default;
  OptOutMutator() = default;

  OptOutMutator(const OptOutMutator& other) = default;
  OptOutMutator& operator=(const OptOutMutator& other) = default;

  OptOutMutator(OptOutMutator&& other) = default;
  OptOutMutator& operator=(OptOutMutator&& other) = default;

  virtual void mutate(Fusion* fusion);

  // Hierarchal dispatch functions for handle
  virtual Statement* mutate(Statement* s);
  virtual Statement* mutate(Expr* e);
  virtual Statement* mutate(Val* v);

  /*
   * We always want to dispatch through a Val, so we can capture and dispatch
   * correctly members of nodes like Split->TensorDomain If we don't call the
   * below function or manually cast to use mutate(Val* v) we can't intercept
   * and mutate by capturing mutate(Val* v), which is what we do when we want to
   * replace all instances of a value.
   */
  Statement* mutateAsVal(Val* v) {
    return mutate(v);
  }

  void registerMutation(Val* val, Val* mutation) {
    TORCH_INTERNAL_ASSERT(
        mutations.find(val) == mutations.end(),
        " The same value is incorrectly being mutated twice.",
        " One mutation per mutation pass is allowed.");
    mutations[val] = mutation;
  }

  std::unordered_map<Val*, Val*> mutations;

  //****Functions below defined in mutator.cpp*****///
  // Vals
  virtual Statement* mutate(IterDomain*);
  virtual Statement* mutate(TensorDomain*);
  virtual Statement* mutate(TensorView*);
  virtual Statement* mutate(TensorIndex*);
  virtual Statement* mutate(Float*);
  virtual Statement* mutate(Int*);
  virtual Statement* mutate(NamedScalar*);

  // Exprs
  virtual Statement* mutate(Split*);
  virtual Statement* mutate(Merge*);
  virtual Statement* mutate(Reorder*);
  virtual Statement* mutate(UnaryOp*);
  virtual Statement* mutate(BinaryOp*);
  virtual Statement* mutate(ForLoop*);
  virtual Statement* mutate(IfThenElse*);
  virtual Statement* mutate(Allocate*);
};

struct TORCH_CUDA_API OptInMutator {
  virtual ~OptInMutator() = default;
  OptInMutator() = default;

  OptInMutator(const OptInMutator& other) = default;
  OptInMutator& operator=(const OptInMutator& other) = default;

  OptInMutator(OptInMutator&& other) = default;
  OptInMutator& operator=(OptInMutator&& other) = default;

  void registerMutation(Val* val, Val* mutation) {
    TORCH_INTERNAL_ASSERT(
        mutations.find(val) == mutations.end(),
        " The same value is incorrectly being mutated twice.",
        " One mutation per mutation pass is allowed.");
    mutations[val] = mutation;
  }

  std::unordered_map<Val*, Val*> mutations;

  // Hierarchal dispatch functions for mutate
  virtual Statement* mutate(Statement*);
  virtual Statement* mutate(Expr*);
  virtual Statement* mutate(Val*);

  // Vals
  virtual Statement* mutate(IterDomain*) {
    TORCH_INTERNAL_ASSERT(false, "Mutate not overriden for IterDomain.");
  }
  virtual Statement* mutate(TensorDomain*) {
    TORCH_INTERNAL_ASSERT(false, "Mutate not overriden for TensorDomain.");
  }
  virtual Statement* mutate(TensorView*) {
    TORCH_INTERNAL_ASSERT(false, "Mutate not overriden for TensorView.");
  }
  virtual Statement* mutate(TensorIndex*) {
    AT_ERROR("Mutate not overriden for TensorIndex.");
  }
  virtual Statement* mutate(Float*) {
    TORCH_INTERNAL_ASSERT(false, "Mutate not overriden for Float.");
  }
  virtual Statement* mutate(Int*) {
    TORCH_INTERNAL_ASSERT(false, "Mutate not overriden for Int.");
  }
  virtual Statement* mutate(NamedScalar*) {
    AT_ERROR("Mutate not overriden for NamedScalar.");
  }

  // Exprs
  virtual Statement* mutate(Split*) {
    TORCH_INTERNAL_ASSERT(false, "Mutate not overriden for Split.");
  }
  virtual Statement* mutate(Merge*) {
    TORCH_INTERNAL_ASSERT(false, "Mutate not overriden for Merge.");
  }
  virtual Statement* mutate(Reorder*) {
    TORCH_INTERNAL_ASSERT(false, "Mutate not overriden for Reorder.");
  }
  virtual Statement* mutate(UnaryOp*) {
    TORCH_INTERNAL_ASSERT(false, "Mutate not overriden for UnaryOp.");
  }
  virtual Statement* mutate(BinaryOp*) {
    TORCH_INTERNAL_ASSERT(false, "Mutate not overriden for BinaryOp.");
  }
  virtual Statement* mutate(ForLoop*) {
    TORCH_INTERNAL_ASSERT(false, "Mutate not overriden for ForLoop.");
  }
  virtual Statement* mutate(Allocate*) {
    AT_ERROR("Mutate not overriden for Allocate.");
  }
  virtual Statement* mutate(IfThenElse*) {
    TORCH_INTERNAL_ASSERT(false, "Mutate not overriden for IfThenElse.");
  }
};

} // namespace fuser
} // namespace jit
} // namespace torch
