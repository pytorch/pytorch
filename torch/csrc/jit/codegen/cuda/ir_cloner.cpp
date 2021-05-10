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

void IrCloner::handle(const Float* f) {
  clone_ = new Float(f, this);
}

void IrCloner::handle(const Half* h) {
  clone_ = new Half(h, this);
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

void IrCloner::handle(const Split* split) {
  clone_ = new Split(split, this);
}

void IrCloner::handle(const Merge* merge) {
  clone_ = new Merge(merge, this);
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
