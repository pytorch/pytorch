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
namespace cuda {

class Fusion;

// Hierarchal dispatch functions for handle
class Statement;
class Expr;
class Val;

// Vals
class IterDomain;
class TensorDomain;
class TensorView;
class Bool;
class Float;
class Half;
class Int;
class NamedScalar;

// Exprs
class Split;
class Merge;
class UnaryOp;
class BinaryOp;
class TernaryOp;
class ReductionOp;
class BroadcastOp;

// Kernel IR
namespace kir {

class Bool;
class Float;
class Half;
class Int;
class NamedScalar;

class IterDomain;
class TensorDomain;
class TensorView;

class UnaryOp;
class BinaryOp;
class TernaryOp;
class ReductionOp;
class BroadcastOp;

class TensorIndex;
class Allocate;
class ForLoop;
class IfThenElse;
class GridReduction;
class Sync;

} // namespace kir

/*
 * By default, all IR nodes are handled in this dispatch, and will call an empty
 * function on all nodes.
 */
class TORCH_CUDA_CU_API OptOutConstDispatch {
 public:
  virtual ~OptOutConstDispatch() = default;
  OptOutConstDispatch() = default;

  OptOutConstDispatch(const OptOutConstDispatch& other) = default;
  OptOutConstDispatch& operator=(const OptOutConstDispatch& other) = default;

  OptOutConstDispatch(OptOutConstDispatch&& other) = default;
  OptOutConstDispatch& operator=(OptOutConstDispatch&& other) = default;

  // Hierarchal dispatch functions for handle
  virtual void handle(const Statement*);
  virtual void handle(const Expr*);
  virtual void handle(const Val*);

  // Vals
  virtual void handle(const IterDomain*) {}
  virtual void handle(const TensorDomain*) {}
  virtual void handle(const TensorView*) {}
  virtual void handle(const Bool*) {}
  virtual void handle(const Float*) {}
  virtual void handle(const Half*) {}
  virtual void handle(const Int*) {}
  virtual void handle(const NamedScalar*) {}

  // Exprs
  virtual void handle(const Split*) {}
  virtual void handle(const Merge*) {}
  virtual void handle(const UnaryOp*) {}
  virtual void handle(const BinaryOp*) {}
  virtual void handle(const TernaryOp*) {}
  virtual void handle(const ReductionOp*) {}
  virtual void handle(const BroadcastOp*) {}

  // Kernel IR nodes
  virtual void handle(const kir::Bool*) {}
  virtual void handle(const kir::Float*) {}
  virtual void handle(const kir::Half*) {}
  virtual void handle(const kir::Int*) {}
  virtual void handle(const kir::NamedScalar*) {}

  virtual void handle(const kir::IterDomain*) {}
  virtual void handle(const kir::TensorDomain*) {}
  virtual void handle(const kir::TensorView*) {}

  virtual void handle(const kir::UnaryOp*) {}
  virtual void handle(const kir::BinaryOp*) {}
  virtual void handle(const kir::TernaryOp*) {}
  virtual void handle(const kir::ReductionOp*) {}
  virtual void handle(const kir::BroadcastOp*) {}

  virtual void handle(const kir::TensorIndex*) {}
  virtual void handle(const kir::GridReduction*) {}
  virtual void handle(const kir::ForLoop*) {}
  virtual void handle(const kir::IfThenElse*) {}
  virtual void handle(const kir::Allocate*) {}
  virtual void handle(const kir::Sync*) {}
};

class TORCH_CUDA_CU_API OptOutDispatch {
 public:
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
  virtual void handle(Bool*) {}
  virtual void handle(Float*) {}
  virtual void handle(Half*) {}
  virtual void handle(Int*) {}
  virtual void handle(NamedScalar*) {}

  // Exprs
  virtual void handle(Split*) {}
  virtual void handle(Merge*) {}
  virtual void handle(UnaryOp*) {}
  virtual void handle(BinaryOp*) {}
  virtual void handle(TernaryOp*) {}
  virtual void handle(ReductionOp*) {}
  virtual void handle(BroadcastOp*) {}

  // Kernel IR nodes
  virtual void handle(kir::Bool*) {}
  virtual void handle(kir::Float*) {}
  virtual void handle(kir::Half*) {}
  virtual void handle(kir::Int*) {}
  virtual void handle(kir::NamedScalar*) {}

  virtual void handle(kir::IterDomain*) {}
  virtual void handle(kir::TensorDomain*) {}
  virtual void handle(kir::TensorView*) {}

  virtual void handle(kir::UnaryOp*) {}
  virtual void handle(kir::BinaryOp*) {}
  virtual void handle(kir::TernaryOp*) {}
  virtual void handle(kir::ReductionOp*) {}
  virtual void handle(kir::BroadcastOp*) {}

  virtual void handle(kir::TensorIndex*) {}
  virtual void handle(kir::GridReduction*) {}
  virtual void handle(kir::ForLoop*) {}
  virtual void handle(kir::IfThenElse*) {}
  virtual void handle(kir::Allocate*) {}
  virtual void handle(kir::Sync*) {}
};

class TORCH_CUDA_CU_API OptInConstDispatch {
 public:
  virtual ~OptInConstDispatch() = default;
  OptInConstDispatch() = default;

  OptInConstDispatch(const OptInConstDispatch& other) = default;
  OptInConstDispatch& operator=(const OptInConstDispatch& other) = default;

  OptInConstDispatch(OptInConstDispatch&& other) = default;
  OptInConstDispatch& operator=(OptInConstDispatch&& other) = default;

  // Hierarchal dispatch functions for handle
  virtual void handle(const Statement*);
  virtual void handle(const Expr*);
  virtual void handle(const Val*);

  // Vals
  virtual void handle(const IterDomain*) {
    TORCH_INTERNAL_ASSERT(false, "Handle not overriden for IterDomain.");
  }
  virtual void handle(const TensorDomain*) {
    TORCH_INTERNAL_ASSERT(false, "Handle not overriden for TensorDomain.");
  }
  virtual void handle(const TensorView*) {
    TORCH_INTERNAL_ASSERT(false, "Handle not overriden for TensorView.");
  }
  virtual void handle(const Bool*) {
    TORCH_INTERNAL_ASSERT(false, "Handle not overriden for Bool.");
  }
  virtual void handle(const Float*) {
    TORCH_INTERNAL_ASSERT(false, "Handle not overriden for Float.");
  }
  virtual void handle(const Half*) {
    TORCH_INTERNAL_ASSERT(false, "Handle not overriden for Half.");
  }
  virtual void handle(const Int*) {
    TORCH_INTERNAL_ASSERT(false, "Handle not overriden for Int.");
  }
  virtual void handle(const NamedScalar*) {
    TORCH_INTERNAL_ASSERT(false, "Handle not overriden for NamedScalar.");
  }

  // Exprs
  virtual void handle(const Split*) {
    TORCH_INTERNAL_ASSERT(false, "Handle not overriden for Split.");
  }
  virtual void handle(const Merge*) {
    TORCH_INTERNAL_ASSERT(false, "Handle not overriden for Merge.");
  }
  virtual void handle(const UnaryOp*) {
    TORCH_INTERNAL_ASSERT(false, "Handle not overriden for UnaryOp.");
  }
  virtual void handle(const BinaryOp*) {
    TORCH_INTERNAL_ASSERT(false, "Handle not overriden for BinaryOp.");
  }
  virtual void handle(const TernaryOp*) {
    TORCH_INTERNAL_ASSERT(false, "Handle not overriden for TernaryOp.");
  }
  virtual void handle(const ReductionOp*) {
    TORCH_INTERNAL_ASSERT(false, "Handle not overriden for ReductionOp.");
  }
  virtual void handle(const BroadcastOp*) {
    TORCH_INTERNAL_ASSERT(false, "Handle not overriden for BroadcastOp.");
  }

  // Kernel IR
  //
  // TODO: move to a specialized visitor
  //

  virtual void handle(const kir::Bool*) {
    TORCH_INTERNAL_ASSERT(false, "Handle not overriden for kir::Bool.");
  }
  virtual void handle(const kir::Float*) {
    TORCH_INTERNAL_ASSERT(false, "Handle not overriden for kir::Float.");
  }
  virtual void handle(const kir::Half*) {
    TORCH_INTERNAL_ASSERT(false, "Handle not overriden for kir::Half.");
  }
  virtual void handle(const kir::Int*) {
    TORCH_INTERNAL_ASSERT(false, "Handle not overriden for kir::Int.");
  }
  virtual void handle(const kir::NamedScalar*) {
    TORCH_INTERNAL_ASSERT(false, "Handle not overriden for kir::NamedScalar.");
  }

  virtual void handle(const kir::IterDomain*) {
    TORCH_INTERNAL_ASSERT(false, "Handle not overriden for kir::IterDomain.");
  }
  virtual void handle(const kir::TensorDomain*) {
    TORCH_INTERNAL_ASSERT(false, "Handle not overriden for kir::TensorDomain.");
  }
  virtual void handle(const kir::TensorView*) {
    TORCH_INTERNAL_ASSERT(false, "Handle not overriden for kir::TensorView.");
  }

  virtual void handle(const kir::UnaryOp*) {
    TORCH_INTERNAL_ASSERT(false, "Handle not overriden for kir::UnaryOp.");
  }
  virtual void handle(const kir::BinaryOp*) {
    TORCH_INTERNAL_ASSERT(false, "Handle not overriden for kir::BinaryOp.");
  }
  virtual void handle(const kir::TernaryOp*) {
    TORCH_INTERNAL_ASSERT(false, "Handle not overriden for kir::TernaryOp.");
  }
  virtual void handle(const kir::ReductionOp*) {
    TORCH_INTERNAL_ASSERT(false, "Handle not overriden for kir::ReductionOp.");
  }
  virtual void handle(const kir::BroadcastOp*) {
    TORCH_INTERNAL_ASSERT(false, "Handle not overriden for kir::BroadcastOp.");
  }

  virtual void handle(const kir::GridReduction*) {
    TORCH_INTERNAL_ASSERT(
        false, "Handle not overriden for kir::GridReduction.");
  }
  virtual void handle(const kir::ForLoop*) {
    TORCH_INTERNAL_ASSERT(false, "Handle not overriden for kir::ForLoop.");
  }
  virtual void handle(const kir::Allocate*) {
    TORCH_INTERNAL_ASSERT(false, "Handle not overriden for kir::Allocate.");
  }
  virtual void handle(const kir::Sync*) {
    TORCH_INTERNAL_ASSERT(false, "Handle not overriden for kir::Sync.");
  }
  virtual void handle(const kir::IfThenElse*) {
    TORCH_INTERNAL_ASSERT(false, "Handle not overriden for kir::IfThenElse.");
  }

  virtual void handle(const kir::TensorIndex*) {
    TORCH_INTERNAL_ASSERT(false, "Handle not overriden for kir::TensorIndex.");
  }
};

class TORCH_CUDA_CU_API OptInDispatch {
 public:
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
  virtual void handle(Bool*) {
    TORCH_INTERNAL_ASSERT(false, "Handle not overriden for Bool.");
  }
  virtual void handle(Float*) {
    TORCH_INTERNAL_ASSERT(false, "Handle not overriden for Float.");
  }
  virtual void handle(Half*) {
    TORCH_INTERNAL_ASSERT(false, "Handle not overriden for Half.");
  }
  virtual void handle(Int*) {
    TORCH_INTERNAL_ASSERT(false, "Handle not overriden for Int.");
  }
  virtual void handle(NamedScalar*) {
    TORCH_INTERNAL_ASSERT(false, "Handle not overriden for NamedScalar.");
  }

  // Exprs
  virtual void handle(Split*) {
    TORCH_INTERNAL_ASSERT(false, "Handle not overriden for Split.");
  }
  virtual void handle(Merge*) {
    TORCH_INTERNAL_ASSERT(false, "Handle not overriden for Merge.");
  }
  virtual void handle(UnaryOp*) {
    TORCH_INTERNAL_ASSERT(false, "Handle not overriden for UnaryOp.");
  }
  virtual void handle(BinaryOp*) {
    TORCH_INTERNAL_ASSERT(false, "Handle not overriden for BinaryOp.");
  }
  virtual void handle(TernaryOp*) {
    TORCH_INTERNAL_ASSERT(false, "Handle not overriden for TernaryOp.");
  }
  virtual void handle(ReductionOp*) {
    TORCH_INTERNAL_ASSERT(false, "Handle not overriden for ReductionOp.");
  }
  virtual void handle(BroadcastOp*) {
    TORCH_INTERNAL_ASSERT(false, "Handle not overriden for BroadcastOp.");
  }

  // Kernel IR
  //
  // TODO: move to a specialized visitor
  //

  virtual void handle(kir::Bool*) {
    TORCH_INTERNAL_ASSERT(false, "Handle not overriden for Bool.");
  }
  virtual void handle(kir::Float*) {
    TORCH_INTERNAL_ASSERT(false, "Handle not overriden for Float.");
  }
  virtual void handle(kir::Half*) {
    TORCH_INTERNAL_ASSERT(false, "Handle not overriden for Half.");
  }
  virtual void handle(kir::Int*) {
    TORCH_INTERNAL_ASSERT(false, "Handle not overriden for Int.");
  }
  virtual void handle(kir::NamedScalar*) {
    TORCH_INTERNAL_ASSERT(false, "Handle not overriden for kir::NamedScalar.");
  }
  virtual void handle(kir::TensorIndex*) {
    TORCH_INTERNAL_ASSERT(false, "Handle not overriden for kir::TensorIndex.");
  }

  virtual void handle(kir::IterDomain*) {
    TORCH_INTERNAL_ASSERT(false, "Handle not overriden for kir::IterDomain.");
  }
  virtual void handle(kir::TensorDomain*) {
    TORCH_INTERNAL_ASSERT(false, "Handle not overriden for kir::TensorDomain.");
  }
  virtual void handle(kir::TensorView*) {
    TORCH_INTERNAL_ASSERT(false, "Handle not overriden for kir::TensorView.");
  }

  virtual void handle(kir::UnaryOp*) {
    TORCH_INTERNAL_ASSERT(false, "Handle not overriden for kir::UnaryOp.");
  }
  virtual void handle(kir::BinaryOp*) {
    TORCH_INTERNAL_ASSERT(false, "Handle not overriden for kir::BinaryOp.");
  }
  virtual void handle(kir::TernaryOp*) {
    TORCH_INTERNAL_ASSERT(false, "Handle not overriden for kir::TernaryOp.");
  }
  virtual void handle(kir::ReductionOp*) {
    TORCH_INTERNAL_ASSERT(false, "Handle not overriden for kir::ReductionOp.");
  }
  virtual void handle(kir::BroadcastOp*) {
    TORCH_INTERNAL_ASSERT(false, "Handle not overriden for kir::BroadcastOp.");
  }

  virtual void handle(kir::GridReduction*) {
    TORCH_INTERNAL_ASSERT(
        false, "Handle not overriden for kir::GridReduction.");
  }
  virtual void handle(kir::ForLoop*) {
    TORCH_INTERNAL_ASSERT(false, "Handle not overriden for kir::ForLoop.");
  }
  virtual void handle(kir::Allocate*) {
    TORCH_INTERNAL_ASSERT(false, "Handle not overriden for kir::Allocate.");
  }
  virtual void handle(kir::Sync*) {
    TORCH_INTERNAL_ASSERT(false, "Handle not overriden for kir::Sync.");
  }
  virtual void handle(kir::IfThenElse*) {
    TORCH_INTERNAL_ASSERT(false, "Handle not overriden for kir::IfThenElse.");
  }
};

// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
class TORCH_CUDA_CU_API OptOutMutator {
 public:
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

  // We always want to dispatch through a Val, so we can capture and dispatch
  // correctly members of nodes like Split->TensorDomain If we don't call the
  // below function or manually cast to use mutate(Val* v) we can't intercept
  // and mutate by capturing mutate(Val* v), which is what we do when we want to
  // replace all instances of a value.
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

  //****Functions below defined in mutator.cpp*****

  // Vals
  virtual Statement* mutate(IterDomain*);
  virtual Statement* mutate(TensorDomain*);
  virtual Statement* mutate(TensorView*);
  virtual Statement* mutate(kir::TensorIndex*);
  virtual Statement* mutate(Bool*);
  virtual Statement* mutate(Float*);
  virtual Statement* mutate(Half*);
  virtual Statement* mutate(Int*);
  virtual Statement* mutate(NamedScalar*);

  // Exprs
  virtual Statement* mutate(Split*);
  virtual Statement* mutate(Merge*);
  virtual Statement* mutate(UnaryOp*);
  virtual Statement* mutate(BinaryOp*);
  virtual Statement* mutate(TernaryOp*);
  virtual Statement* mutate(ReductionOp*);
  virtual Statement* mutate(kir::GridReduction*);
  virtual Statement* mutate(BroadcastOp*);
  virtual Statement* mutate(kir::ForLoop*);
  virtual Statement* mutate(kir::IfThenElse*);
  virtual Statement* mutate(kir::Allocate*);
  virtual Statement* mutate(kir::Sync*);
};

// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
class TORCH_CUDA_CU_API OptInMutator {
 public:
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
  virtual Statement* mutate(kir::TensorIndex*) {
    TORCH_INTERNAL_ASSERT(false, "Mutate not overriden for TensorIndex.");
  }
  virtual Statement* mutate(Bool*) {
    TORCH_INTERNAL_ASSERT(false, "Mutate not overriden for Bool.");
  }
  virtual Statement* mutate(Float*) {
    TORCH_INTERNAL_ASSERT(false, "Mutate not overriden for Float.");
  }
  virtual Statement* mutate(Int*) {
    TORCH_INTERNAL_ASSERT(false, "Mutate not overriden for Int.");
  }
  virtual Statement* mutate(NamedScalar*) {
    TORCH_INTERNAL_ASSERT(false, "Mutate not overriden for NamedScalar.");
  }

  // Exprs
  virtual Statement* mutate(Split*) {
    TORCH_INTERNAL_ASSERT(false, "Mutate not overriden for Split.");
  }
  virtual Statement* mutate(Merge*) {
    TORCH_INTERNAL_ASSERT(false, "Mutate not overriden for Merge.");
  }
  virtual Statement* mutate(UnaryOp*) {
    TORCH_INTERNAL_ASSERT(false, "Mutate not overriden for UnaryOp.");
  }
  virtual Statement* mutate(BinaryOp*) {
    TORCH_INTERNAL_ASSERT(false, "Mutate not overriden for BinaryOp.");
  }
  virtual Statement* mutate(TernaryOp*) {
    TORCH_INTERNAL_ASSERT(false, "Mutate not overriden for TernaryOp.");
  }
  virtual Statement* mutate(ReductionOp*) {
    TORCH_INTERNAL_ASSERT(false, "Mutate not overriden for ReductionOp.");
  }
  virtual Statement* mutate(kir::GridReduction*) {
    TORCH_INTERNAL_ASSERT(false, "Mutate not overriden for GridReduction.");
  }
  virtual Statement* mutate(BroadcastOp*) {
    TORCH_INTERNAL_ASSERT(false, "Mutate not overriden for BroadcastOp.");
  }
  virtual Statement* mutate(kir::ForLoop*) {
    TORCH_INTERNAL_ASSERT(false, "Mutate not overriden for ForLoop.");
  }
  virtual Statement* mutate(kir::Allocate*) {
    TORCH_INTERNAL_ASSERT(false, "Mutate not overriden for Allocate.");
  }
  virtual Statement* mutate(kir::Sync*) {
    TORCH_INTERNAL_ASSERT(false, "Mutate not overriden for Sync.");
  }
  virtual Statement* mutate(kir::IfThenElse*) {
    TORCH_INTERNAL_ASSERT(false, "Mutate not overriden for IfThenElse.");
  }
};

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
