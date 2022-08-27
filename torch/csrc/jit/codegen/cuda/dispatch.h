#pragma once

#include <c10/macros/Export.h>
#include <c10/util/Exception.h>

#include <torch/csrc/jit/codegen/cuda/utils.h>

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
class IrContainer;
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
class ComplexDouble;
class NamedScalar;

// Exprs
class ARangeOp;
class UnaryOp;
class BinaryOp;
class TernaryOp;
class RNGOp;
class ReductionOp;
class GroupedReductionOp;
class WelfordOp;
class GroupedWelfordOp;
class LoadStoreOp;
class MmaOp;
class BroadcastOp;
class TransposeOp;
class ExpandOp;
class ShiftOp;
class GatherOp;
class ViewAsScalar;
class ViewOp;

// Exprs
class Split;
class Merge;
class Swizzle2D;

namespace kir {
class Predicate;
class TensorIndex;
class IntPair;

class Allocate;
class BlockSync;
class GridSync;
class CpAsyncWait;
class CpAsyncCommit;
class ForLoop;
class IfThenElse;
class GridReduction;
class GroupedGridReduction;
class GridBroadcast;
class GridWelford;
class GroupedGridWelford;
class AllocateFusedReduction;
class InitMagicZero;
class UpdateMagicZero;
class Swizzle2DInt;
class PairSelect;

} // namespace kir

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
  virtual void handle(const ComplexDouble* stmt);
  virtual void handle(const NamedScalar* stmt);

  virtual void handle(const kir::Predicate*);
  virtual void handle(const kir::TensorIndex*);
  virtual void handle(const kir::IntPair*);

  // Exprs
  virtual void handle(const ARangeOp* stmt);
  virtual void handle(const UnaryOp* stmt);
  virtual void handle(const BinaryOp* stmt);
  virtual void handle(const TernaryOp* stmt);
  virtual void handle(const RNGOp* stmt);
  virtual void handle(const ReductionOp* stmt);
  virtual void handle(const GroupedReductionOp* stmt);
  virtual void handle(const WelfordOp* stmt);
  virtual void handle(const GroupedWelfordOp* stmt);
  virtual void handle(const LoadStoreOp* stmt);
  virtual void handle(const MmaOp* stmt);
  virtual void handle(const BroadcastOp* stmt);

  virtual void handle(const Split* stmt);
  virtual void handle(const Merge* stmt);
  virtual void handle(const Swizzle2D* stmt);
  virtual void handle(const TransposeOp* stmt);
  virtual void handle(const ExpandOp* stmt);
  virtual void handle(const ShiftOp* stmt);
  virtual void handle(const GatherOp* stmt);
  virtual void handle(const ViewAsScalar* stmt);
  virtual void handle(const ViewOp* stmt);

  virtual void handle(const kir::Allocate*);
  virtual void handle(const kir::BlockSync*);
  virtual void handle(const kir::GridSync*);
  virtual void handle(const kir::CpAsyncWait*);
  virtual void handle(const kir::CpAsyncCommit*);
  virtual void handle(const kir::InitMagicZero*);
  virtual void handle(const kir::UpdateMagicZero*);
  virtual void handle(const kir::ForLoop*);
  virtual void handle(const kir::IfThenElse*);
  virtual void handle(const kir::GridReduction*);
  virtual void handle(const kir::GroupedGridReduction*);
  virtual void handle(const kir::GridBroadcast*);
  virtual void handle(const kir::GridWelford*);
  virtual void handle(const kir::GroupedGridWelford*);
  virtual void handle(const kir::AllocateFusedReduction*);
  virtual void handle(const kir::Swizzle2DInt*);
  virtual void handle(const kir::PairSelect*);
};

class TORCH_CUDA_CU_API OptOutDispatch : public PolymorphicBase {
 protected:
  virtual void unhandled(Statement*);

 public:
  // Hierarchal dispatch functions for handle
  virtual void handle(Statement*);
  virtual void handle(Expr*);
  virtual void handle(Val*);

  // Vals
  virtual void handle(Bool* stmt);
  virtual void handle(Double* stmt);
  virtual void handle(Int* stmt);
  virtual void handle(ComplexDouble* stmt);
  virtual void handle(NamedScalar* stmt);
  virtual void handle(IterDomain* stmt);
  virtual void handle(TensorDomain* stmt);
  virtual void handle(TensorView* stmt);

  virtual void handle(kir::Predicate*);
  virtual void handle(kir::TensorIndex*);
  virtual void handle(kir::IntPair*);

  // Exprs
  virtual void handle(ARangeOp* stmt);
  virtual void handle(UnaryOp* stmt);
  virtual void handle(BinaryOp* stmt);
  virtual void handle(TernaryOp* stmt);
  virtual void handle(RNGOp* stmt);
  virtual void handle(ReductionOp* stmt);
  virtual void handle(GroupedReductionOp* stmt);
  virtual void handle(WelfordOp* stmt);
  virtual void handle(GroupedWelfordOp* stmt);
  virtual void handle(LoadStoreOp* stmt);
  virtual void handle(MmaOp* stmt);
  virtual void handle(BroadcastOp* stmt);

  virtual void handle(Split* stmt);
  virtual void handle(Merge* stmt);
  virtual void handle(Swizzle2D* stmt);
  virtual void handle(TransposeOp* stmt);
  virtual void handle(ExpandOp* stmt);
  virtual void handle(ShiftOp* stmt);
  virtual void handle(GatherOp* stmt);
  virtual void handle(ViewAsScalar* stmt);
  virtual void handle(ViewOp* stmt);

  virtual void handle(kir::Allocate* stmt);
  virtual void handle(kir::BlockSync* stmt);
  virtual void handle(kir::GridSync* stmt);
  virtual void handle(kir::CpAsyncWait* stmt);
  virtual void handle(kir::CpAsyncCommit* stmt);
  virtual void handle(kir::InitMagicZero* stmt);
  virtual void handle(kir::UpdateMagicZero* stmt);
  virtual void handle(kir::ForLoop* stmt);
  virtual void handle(kir::IfThenElse* stmt);
  virtual void handle(kir::GridReduction* stmt);
  virtual void handle(kir::GroupedGridReduction* stmt);
  virtual void handle(kir::GridBroadcast* stmt);
  virtual void handle(kir::GridWelford* stmt);
  virtual void handle(kir::GroupedGridWelford* stmt);
  virtual void handle(kir::AllocateFusedReduction* stmt);
  virtual void handle(kir::Swizzle2DInt* stmt);
  virtual void handle(kir::PairSelect* stmt);
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

// Class to perform mutations on Fusion IR. Exprs can simply be redefined, but
// when mutating values they have to be registered through registerMutation so
// that exprs can detect there's been a muatation and know to modify all
// instances of that Val. This means each Val should be mutated "consistently".
// Otherwise behavior may be difficult to understand as it depends on which
// order mutate is called in. This class expects user to topologically call the
// statments of interest so inputs are called and mutated before exprs depending
// on them.
//
// Warning: TensorViews need to be treated carefully. As we don't generally
// register their mutation when their tensor domains only change. If a TV needs
// to be swapped out, it needs to be registered as a "proper" mutation like
// other vals, on top of TensorDomain being updated in the mutated TensorView.
//
// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
class TORCH_CUDA_CU_API OptOutMutator : public PolymorphicBase {
 public:
  // Hierarchal dispatch functions for handle
  virtual void mutate(Statement* s);
  virtual void mutate(Expr* e);
  virtual void mutate(Val* v);

  void registerMutation(Val* val, Val* mutation);

  Val* maybeMutated(Val* val) {
    if (mutations.find(val) == mutations.end()) {
      return val;
    }
    return mutations.at(val);
  }

  std::unordered_map<Val*, Val*> mutations;

  //****Functions below defined in mutator.cpp*****

  // Vals
  virtual void mutate(Bool*);
  virtual void mutate(Double*);
  virtual void mutate(Int*);
  virtual void mutate(ComplexDouble*);
  virtual void mutate(NamedScalar*);
  virtual void mutate(IterDomain*);
  virtual void mutate(TensorDomain*);
  virtual void mutate(TensorView*);

  virtual void mutate(kir::Predicate*);
  virtual void mutate(kir::TensorIndex*);
  virtual void mutate(kir::IntPair*);

  // Exprs
  virtual void mutate(ARangeOp*);
  virtual void mutate(UnaryOp*);
  virtual void mutate(BinaryOp*);
  virtual void mutate(TernaryOp*);
  virtual void mutate(RNGOp*);
  virtual void mutate(ReductionOp*);
  virtual void mutate(GroupedReductionOp*);
  virtual void mutate(WelfordOp*);
  virtual void mutate(GroupedWelfordOp*);
  virtual void mutate(LoadStoreOp*);
  virtual void mutate(MmaOp*);
  virtual void mutate(BroadcastOp*);

  virtual void mutate(Split*);
  virtual void mutate(Merge*);
  virtual void mutate(Swizzle2D*);
  virtual void mutate(TransposeOp*);
  virtual void mutate(ExpandOp*);
  virtual void mutate(ShiftOp*);
  virtual void mutate(GatherOp*);
  virtual void mutate(ViewAsScalar*);
  virtual void mutate(ViewOp*);

  virtual void mutate(kir::Allocate*);
  virtual void mutate(kir::BlockSync*);
  virtual void mutate(kir::GridSync*);
  virtual void mutate(kir::CpAsyncWait*);
  virtual void mutate(kir::CpAsyncCommit*);
  virtual void mutate(kir::InitMagicZero*);
  virtual void mutate(kir::UpdateMagicZero*);
  virtual void mutate(kir::ForLoop*);
  virtual void mutate(kir::IfThenElse*);
  virtual void mutate(kir::GridReduction*);
  virtual void mutate(kir::GroupedGridReduction*);
  virtual void mutate(kir::GridBroadcast*);
  virtual void mutate(kir::GridWelford*);
  virtual void mutate(kir::GroupedGridWelford*);
  virtual void mutate(kir::AllocateFusedReduction*);
  virtual void mutate(kir::Swizzle2DInt*);
  virtual void mutate(kir::PairSelect*);

 protected:
  void removeExpr(IrContainer*, Expr*);
};

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
