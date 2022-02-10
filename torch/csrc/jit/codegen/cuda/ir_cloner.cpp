#include <torch/csrc/jit/codegen/cuda/ir_cloner.h>

#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

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
    TORCH_INTERNAL_ASSERT(new_node->fusion() == fusion_);

    return new_node;
  }
}

void IrCloner::registerClone(const Statement* src, Statement* clone) {
  TORCH_CHECK(src != nullptr);
  TORCH_CHECK(clone != nullptr);
  TORCH_CHECK(clone->fusion() == fusion_);
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
  clone_ = new TensorDomain(td, this);
}

void IrCloner::handle(const IterDomain* id) {
  clone_ = new IterDomain(id, this);
}

void IrCloner::handle(const Bool* b) {
  clone_ = new Bool(b, this);
}

void IrCloner::handle(const Double* d) {
  clone_ = new Double(d, this);
}

void IrCloner::handle(const Int* i) {
  clone_ = new Int(i, this);
}

void IrCloner::handle(const NamedScalar* named_scalar) {
  clone_ = new NamedScalar(named_scalar, this);
}

void IrCloner::handle(const TensorView* tv) {
  clone_ = new TensorView(tv, this);
}

void IrCloner::handle(const UnaryOp* op) {
  clone_ = new UnaryOp(op, this);
}

void IrCloner::handle(const BinaryOp* op) {
  clone_ = new BinaryOp(op, this);
}

void IrCloner::handle(const TernaryOp* op) {
  clone_ = new TernaryOp(op, this);
}

void IrCloner::handle(const BroadcastOp* op) {
  clone_ = new BroadcastOp(op, this);
}

void IrCloner::handle(const ReductionOp* op) {
  clone_ = new ReductionOp(op, this);
}

void IrCloner::handle(const WelfordOp* op) {
  clone_ = new WelfordOp(op, this);
}

void IrCloner::handle(const TransposeOp* op) {
  clone_ = new TransposeOp(op, this);
}

void IrCloner::handle(const ShiftOp* op) {
  clone_ = new ShiftOp(op, this);
}

void IrCloner::handle(const GatherOp* op) {
  clone_ = new GatherOp(op, this);
}

void IrCloner::handle(const ViewOp* op) {
  clone_ = new ViewOp(op, this);
}

void IrCloner::handle(const Split* split) {
  clone_ = new Split(split, this);
}

void IrCloner::handle(const Merge* merge) {
  clone_ = new Merge(merge, this);
}

TensorView* RecomputeTv::recompute(TensorView* tv) {
  FusionGuard fg(tv->fusion());

  // Disallow recomputation of inputs or outputs. User would have to be aware of
  // these changes and informed they happened somehow.
  TORCH_INTERNAL_ASSERT(
      !tv->isFusionInput(),
      "Cannot recompute buffers that are inputs of the fusion.");

  // Grab all the expressions used to generate the TensorView
  auto exprs = ExprSort::getExprs(tv->fusion(), {tv});

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
    : IrCloner(fusion) {
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
      ExprSort::getExprs(fusion(), {td->domain().begin(), td->domain().end()});

  for (auto expr : exprs) {
    IrCloner::handle(expr);
  }
  IrCloner::handle(td);
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
