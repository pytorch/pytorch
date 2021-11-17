#pragma once

#include <torch/csrc/jit/codegen/cuda/utils.h>

#include <c10/util/Exception.h>
#include <torch/csrc/WindowsTorchApiMacro.h>

#include <unordered_map>

// dispatch.h prevents the need from adding manual dispatch in every class that
// wants to define how to process a series of nodes. dispatch.h provides 4
// classes that can be inherited providing a means to override functions on a
// per-node basis. There are currently 4 provided dispatch mechanisms:
//
// OptOutDispatch:
//
// provides the functions:
// virtual void handle(ValType* irnode){}
//
// This provides a mechanisms to override this handle for particular node
// types. For example if we only wanted to actually run a function on
// BinaryOps, we could inherit OptOutDispatch and simply override: void
// handle(BinaryOp*) { doSomething; } Then we could run through all our
// Statement* and call OptOutDispatch::handle(statement). When a BinaryOp is
// encountered our override function will be called. For every other node,
// nothing will be done.
//
// OptInDispatch:
//
// This class is similar to OptOutDispatch, however if we encounter a node
// that we haven't specified an override for in the derived class, an error
// will be thrown. This is useful if we create a class that is expected to
// handle any type of node it encounters.
//
// OptOutMutator:
//
// This class is similar to OptOutDispatch except the functions provided are of
// type: virtual Statement* mutate(Statement*) this is useful for when we want
// to have an IR node result from our overloaded functions.
//
// OptInMutator:
//
// This class is similar to OptInDispatch except the functions provided are of
// type: virtual Statement* mutate(Statement*) this is useful for when we want
// to have an IR node result from our overloaded functions.

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
class Double;
class Int;
class NamedScalar;

// Exprs
class Split;
class Merge;
class UnaryOp;
class BinaryOp;
class TernaryOp;
class ReductionOp;
class WelfordOp;
class BroadcastOp;
class TransposeOp;
class ShiftOp;
class GatherOp;

// By default, all IR nodes are handled in this dispatch, and will call an empty
// function on all nodes.
class TORCH_CUDA_CU_API OptOutConstDispatch : public PolymorphicBase {
 public:
  // Hierarchal dispatch functions for handle
  virtual void handle(const Statement*);
  virtual void handle(const Expr*);
  virtual void handle(const Val*);

  // Vals
  virtual void handle(const IterDomain*) {}
  virtual void handle(const TensorDomain*) {}
  virtual void handle(const TensorView*) {}
  virtual void handle(const Bool*) {}
  virtual void handle(const Double*) {}
  virtual void handle(const Int*) {}
  virtual void handle(const NamedScalar*) {}

  // Exprs
  virtual void handle(const Split*) {}
  virtual void handle(const Merge*) {}
  virtual void handle(const UnaryOp*) {}
  virtual void handle(const BinaryOp*) {}
  virtual void handle(const TernaryOp*) {}
  virtual void handle(const ReductionOp*) {}
  virtual void handle(const WelfordOp*) {}
  virtual void handle(const BroadcastOp*) {}
  virtual void handle(const TransposeOp*) {}
  virtual void handle(const ShiftOp*) {}
  virtual void handle(const GatherOp*) {}
};

class TORCH_CUDA_CU_API OptOutDispatch : public PolymorphicBase {
 public:
  // Hierarchal dispatch functions for handle
  virtual void handle(Statement*);
  virtual void handle(Expr*);
  virtual void handle(Val*);

  // Vals
  virtual void handle(IterDomain*) {}
  virtual void handle(TensorDomain*) {}
  virtual void handle(TensorView*) {}
  virtual void handle(Bool*) {}
  virtual void handle(Double*) {}
  virtual void handle(Int*) {}
  virtual void handle(NamedScalar*) {}

  // Exprs
  virtual void handle(Split*) {}
  virtual void handle(Merge*) {}
  virtual void handle(UnaryOp*) {}
  virtual void handle(BinaryOp*) {}
  virtual void handle(TernaryOp*) {}
  virtual void handle(ReductionOp*) {}
  virtual void handle(WelfordOp*) {}
  virtual void handle(BroadcastOp*) {}
  virtual void handle(TransposeOp*) {}
  virtual void handle(ShiftOp*) {}
  virtual void handle(GatherOp*) {}
};

class TORCH_CUDA_CU_API OptInConstDispatch : public PolymorphicBase {
 public:
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
  virtual void handle(const Double*) {
    TORCH_INTERNAL_ASSERT(false, "Handle not overriden for Double.");
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
  virtual void handle(const WelfordOp*) {
    TORCH_INTERNAL_ASSERT(false, "Handle not overriden for WelfordOp.");
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
  virtual void handle(const TransposeOp*) {
    TORCH_INTERNAL_ASSERT(false, "Handle not overriden for TransposeOp.");
  }
  virtual void handle(const ShiftOp*) {
    TORCH_INTERNAL_ASSERT(false, "Handle not overriden for ShiftOp.");
  }
  virtual void handle(const GatherOp*) {
    TORCH_INTERNAL_ASSERT(false, "Handle not overriden for GatherOp.");
  }
};

class TORCH_CUDA_CU_API OptInDispatch : public PolymorphicBase {
 public:
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
  virtual void handle(Double*) {
    TORCH_INTERNAL_ASSERT(false, "Handle not overriden for Double.");
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
  virtual void handle(WelfordOp*) {
    TORCH_INTERNAL_ASSERT(false, "Handle not overriden for WelfordOp.");
  }
  virtual void handle(BroadcastOp*) {
    TORCH_INTERNAL_ASSERT(false, "Handle not overriden for BroadcastOp.");
  }
  virtual void handle(TransposeOp*) {
    TORCH_INTERNAL_ASSERT(false, "Handle not overriden for TransposeOp.");
  }
  virtual void handle(ShiftOp*) {
    TORCH_INTERNAL_ASSERT(false, "Handle not overriden for ShiftOp.");
  }
  virtual void handle(GatherOp*) {
    TORCH_INTERNAL_ASSERT(false, "Handle not overriden for GatherOp.");
  }
};

// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
class TORCH_CUDA_CU_API OptOutMutator : public PolymorphicBase {
 public:
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
  virtual Statement* mutate(Bool*);
  virtual Statement* mutate(Double*);
  virtual Statement* mutate(Int*);
  virtual Statement* mutate(NamedScalar*);

  // Exprs
  virtual Statement* mutate(Split*);
  virtual Statement* mutate(Merge*);
  virtual Statement* mutate(UnaryOp*);
  virtual Statement* mutate(BinaryOp*);
  virtual Statement* mutate(TernaryOp*);
  virtual Statement* mutate(ReductionOp*);
  virtual Statement* mutate(WelfordOp*);
  virtual Statement* mutate(BroadcastOp*);
  virtual Statement* mutate(TransposeOp*);
  virtual Statement* mutate(ShiftOp*);
  virtual Statement* mutate(GatherOp*);
};

// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
class TORCH_CUDA_CU_API OptInMutator : public PolymorphicBase {
 public:
  std::unordered_map<Val*, Val*> mutations;

 public:
  void registerMutation(Val* val, Val* mutation) {
    TORCH_INTERNAL_ASSERT(
        mutations.find(val) == mutations.end(),
        " The same value is incorrectly being mutated twice.",
        " One mutation per mutation pass is allowed.");
    mutations[val] = mutation;
  }

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
  virtual Statement* mutate(Bool*) {
    TORCH_INTERNAL_ASSERT(false, "Mutate not overriden for Bool.");
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
  virtual Statement* mutate(WelfordOp*) {
    TORCH_INTERNAL_ASSERT(false, "Mutate not overriden for WelfordOp.");
  }
  virtual Statement* mutate(BroadcastOp*) {
    TORCH_INTERNAL_ASSERT(false, "Mutate not overriden for BroadcastOp.");
  }
  virtual Statement* mutate(TransposeOp*) {
    TORCH_INTERNAL_ASSERT(false, "Mutate not overriden for TransposeOp.");
  }
  virtual Statement* mutate(ShiftOp*) {
    TORCH_INTERNAL_ASSERT(false, "Mutate not overriden for ShiftOp.");
  }
  virtual Statement* mutate(GatherOp*) {
    TORCH_INTERNAL_ASSERT(false, "Mutate not overriden for GatherOp.");
  }
};

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
