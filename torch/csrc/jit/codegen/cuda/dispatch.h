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
class ViewOp;

// By default, all IR nodes are handled in this dispatch, and will call an empty
// function on all nodes.
class TORCH_CUDA_CU_API OptOutConstDispatch : public PolymorphicBase {
 protected:
  virtual void unhandled(const Statement*) {}

 public:
  // Hierarchal dispatch functions for handle
  virtual void handle(const Statement*);
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

  // Exprs
  virtual void handle(const Split* stmt);
  virtual void handle(const Merge* stmt);
  virtual void handle(const UnaryOp* stmt);
  virtual void handle(const BinaryOp* stmt);
  virtual void handle(const TernaryOp* stmt);
  virtual void handle(const ReductionOp* stmt);
  virtual void handle(const WelfordOp* stmt);
  virtual void handle(const BroadcastOp* stmt);
  virtual void handle(const TransposeOp* stmt);
  virtual void handle(const ShiftOp* stmt);
  virtual void handle(const GatherOp* stmt);
  virtual void handle(const ViewOp* stmt);
};

class TORCH_CUDA_CU_API OptOutDispatch : public PolymorphicBase {
 protected:
  virtual void unhandled(Statement*) {}

 public:
  // Hierarchal dispatch functions for handle
  virtual void handle(Statement*);
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

  // Exprs
  virtual void handle(Split* stmt);
  virtual void handle(Merge* stmt);
  virtual void handle(UnaryOp* stmt);
  virtual void handle(BinaryOp* stmt);
  virtual void handle(TernaryOp* stmt);
  virtual void handle(ReductionOp* stmt);
  virtual void handle(WelfordOp* stmt);
  virtual void handle(BroadcastOp* stmt);
  virtual void handle(TransposeOp* stmt);
  virtual void handle(ShiftOp* stmt);
  virtual void handle(GatherOp* stmt);
  virtual void handle(ViewOp* stmt);
};

class TORCH_CUDA_CU_API OptInConstDispatch : public OptOutConstDispatch {
 public:
  using OptOutConstDispatch::handle;

 protected:
  virtual void unhandled(const Statement* stmt) final;
};

class TORCH_CUDA_CU_API OptInDispatch : public OptOutDispatch {
 public:
  using OptOutDispatch::handle;

 protected:
  virtual void unhandled(Statement* stmt) final;
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
  virtual Statement* mutate(ViewOp*);
};

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
