#include <ir_cloner.h>

#include <fusion.h>
#include <ir_all_nodes.h>
#include <ir_builder.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

IrCloner::IrCloner(IrContainer* container) : ir_container_(container) {}

Statement* IrCloner::clone(const Statement* statement) {
  if (statement == nullptr) {
    return nullptr;
  }

  // Have we already cloned this node?
  const auto it = clones_map_.find(statement);
  if (it != clones_map_.end()) {
    return it->second;
  } else {
    // Clone the new node, saving/restoring this->clone_
    // since the cloning can be reentrant
    auto saved_clone = clone_;
    handle(statement);
    auto new_node = clone_;
    clone_ = saved_clone;

    // The base cloning constructor (Statement) should have
    // registered the new node. Failure to do so indicates
    // that something went horribly wrong.
    TORCH_INTERNAL_ASSERT(new_node != nullptr);
    TORCH_INTERNAL_ASSERT(clones_map_[statement] == new_node);

    return new_node;
  }
}

void IrCloner::registerClone(const Statement* src, Statement* clone) {
  TORCH_CHECK(src != nullptr);
  TORCH_CHECK(clone != nullptr);
  TORCH_CHECK(clones_map_.insert({src, clone}).second);
}

void IrCloner::handle(const Statement* s) {
  OptInConstDispatch::handle(s);
}

void IrCloner::handle(const Val* v) {
  OptInConstDispatch::handle(v);
}

void IrCloner::handle(const Expr* e) {
  OptInConstDispatch::handle(e);
}

void IrCloner::handle(const TensorDomain* td) {
  clone_ = IrBuilder::clone(td, this);
}

void IrCloner::handle(const IterDomain* id) {
  clone_ = IrBuilder::clone(id, this);
}

void IrCloner::handle(const Bool* b) {
  clone_ = IrBuilder::clone(b, this);
}

void IrCloner::handle(const Double* d) {
  clone_ = IrBuilder::clone(d, this);
}

void IrCloner::handle(const Int* i) {
  clone_ = IrBuilder::clone(i, this);
}

void IrCloner::handle(const ComplexDouble* c) {
  clone_ = IrBuilder::clone(c, this);
}

void IrCloner::handle(const NamedScalar* named_scalar) {
  clone_ = IrBuilder::clone(named_scalar, this);
}

void IrCloner::handle(const TensorView* tv) {
  clone_ = IrBuilder::clone(tv, this);
}

void IrCloner::handle(const FullOp* op) {
  clone_ = IrBuilder::clone(op, this);
}

void IrCloner::handle(const ARangeOp* op) {
  clone_ = IrBuilder::clone(op, this);
}

void IrCloner::handle(const EyeOp* op) {
  clone_ = IrBuilder::clone(op, this);
}

void IrCloner::handle(const UnaryOp* op) {
  clone_ = IrBuilder::clone(op, this);
}

void IrCloner::handle(const BinaryOp* op) {
  clone_ = IrBuilder::clone(op, this);
}

void IrCloner::handle(const TernaryOp* op) {
  clone_ = IrBuilder::clone(op, this);
}

void IrCloner::handle(const RNGOp* op) {
  clone_ = IrBuilder::clone(op, this);
}

void IrCloner::handle(const BroadcastOp* op) {
  clone_ = IrBuilder::clone(op, this);
}

void IrCloner::handle(const ReductionOp* op) {
  clone_ = IrBuilder::clone(op, this);
}

void IrCloner::handle(const GroupedReductionOp* op) {
  clone_ = IrBuilder::clone(op, this);
}

void IrCloner::handle(const WelfordOp* op) {
  clone_ = IrBuilder::clone(op, this);
}

void IrCloner::handle(const LoadStoreOp* op) {
  clone_ = IrBuilder::clone(op, this);
}

void IrCloner::handle(const MmaOp* op) {
  clone_ = IrBuilder::clone(op, this);
}

void IrCloner::handle(const TransposeOp* op) {
  clone_ = IrBuilder::clone(op, this);
}

void IrCloner::handle(const ExpandOp* op) {
  clone_ = IrBuilder::clone(op, this);
}

void IrCloner::handle(const ShiftOp* op) {
  clone_ = IrBuilder::clone(op, this);
}

void IrCloner::handle(const GatherOp* op) {
  clone_ = IrBuilder::clone(op, this);
}

void IrCloner::handle(const ViewAsScalar* op) {
  clone_ = IrBuilder::clone(op, this);
}

void IrCloner::handle(const ViewOp* op) {
  clone_ = IrBuilder::clone(op, this);
}

void IrCloner::handle(const Split* split) {
  clone_ = IrBuilder::clone(split, this);
}

void IrCloner::handle(const Merge* merge) {
  clone_ = IrBuilder::clone(merge, this);
}

void IrCloner::handle(const Swizzle2D* swizzle) {
  clone_ = IrBuilder::clone(swizzle, this);
}

TensorView* RecomputeTv::recompute(TensorView* tv) {
  FusionGuard fg(tv->fusion());

  // Disallow recomputation of inputs or outputs. User would have to be aware of
  // these changes and informed they happened somehow.
  TORCH_INTERNAL_ASSERT(
      !tv->isFusionInput(),
      "Cannot recompute buffers that are inputs of the fusion.");

  // Grab all the expressions used to generate the TensorView
  auto exprs = StmtSort::getExprs(tv->fusion(), {tv}, false);

  // Run the replicator
  RecomputeTv replicator(tv->fusion(), exprs);

  // Make const version of pointer for lookup
  const auto const_tv = tv;
  // Find the recomputed tensor from the cloner
  auto clone_it = replicator.clones_map_.find(const_tv);
  TORCH_INTERNAL_ASSERT(clone_it != replicator.clones_map_.end());
  auto cloned_val = clone_it->second;
  TORCH_INTERNAL_ASSERT(
      cloned_val->isA<TensorView>(),
      "Cloned value is somehow not a tensor view.");

  // Return the cloned value
  return cloned_val->as<TensorView>();
}

RecomputeTv::RecomputeTv(Fusion* fusion, std::vector<Expr*> exprs)
    : IrCloner(fusion), fusion_(fusion) {
  // Add inputs to the clones map to prevent cloning them.
  for (const auto inp : fusion->inputs()) {
    clones_map_[inp] = inp;
  }
  // Adds all scalar values to clones map to prevent cloning them
  for (const auto val : fusion->vals()) {
    if (val->getValType().value() == ValType::Scalar ||
        val->getValType().value() == ValType::NamedScalar) {
      clones_map_[val] = val;
    }
  }
  // Clone the expressions
  for (auto expr : exprs) {
    IrCloner::handle(expr);
  }
}

void RecomputeTv::handle(const TensorDomain* td) {
  // Make sure to recompute the history of the iteration domains, explicitly go
  // through the expressions and send them to IrCloner.
  auto exprs =
      StmtSort::getExprs(fusion_, {td->domain().begin(), td->domain().end()});

  for (auto expr : exprs) {
    IrCloner::handle(expr);
  }
  IrCloner::handle(td);
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
