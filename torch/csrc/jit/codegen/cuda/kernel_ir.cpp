#include <torch/csrc/jit/codegen/cuda/ir_builder.h>
#include <torch/csrc/jit/codegen/cuda/kernel.h>
#include <torch/csrc/jit/codegen/cuda/kernel_expr_evaluator.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir.h>
#include <torch/csrc/jit/codegen/cuda/lower2device.h>
#include <torch/csrc/jit/codegen/cuda/lower_utils.h>
#include <torch/csrc/jit/codegen/cuda/type.h>

#include <iostream>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {
namespace kir {

Predicate::Predicate(
    IrBuilderPasskey passkey,
    PredicateType ptype,
    const Expr* expr,
    Bool* thread_pred)
    : Val(passkey, ValType::Predicate, DataType::Bool),
      ptype_(ptype),
      expr_(expr),
      thread_pred_(thread_pred) {
  TORCH_INTERNAL_ASSERT(
      passkey.ir_container_->isA<kir::Kernel>(),
      "IR type only valid for Kernel container.");
  TORCH_INTERNAL_ASSERT(
      ptype != PredicateType::Unswitch && ptype != PredicateType::Manual);
}

Predicate::Predicate(IrBuilderPasskey passkey, ForLoop* unrolled_loop)
    : Val(passkey, ValType::Predicate, DataType::Bool),
      ptype_(PredicateType::Unswitch),
      unrolled_loop_(unrolled_loop) {
  TORCH_INTERNAL_ASSERT(
      passkey.ir_container_->isA<kir::Kernel>(),
      "IR type only valid for Kernel container.");
  TORCH_INTERNAL_ASSERT(unrolled_loop != nullptr);
}

Predicate::Predicate(IrBuilderPasskey passkey, Bool* value)
    : Val(passkey, ValType::Predicate, DataType::Bool),
      ptype_(PredicateType::Manual),
      value_(value) {
  TORCH_INTERNAL_ASSERT(
      passkey.ir_container_->isA<kir::Kernel>(),
      "IR type only valid for Kernel container.");
  TORCH_INTERNAL_ASSERT(value != nullptr);
}

TensorIndex::TensorIndex(
    IrBuilderPasskey passkey,
    const TensorView* view,
    std::vector<Val*> indices)
    : Val(passkey, ValType::TensorIndex, view->getDataType().value()),
      view_(view),
      indices_(indices) {
  TORCH_INTERNAL_ASSERT(
      passkey.ir_container_->isA<kir::Kernel>(),
      "IR type only valid for Kernel container.");
  TORCH_INTERNAL_ASSERT(
      std::all_of(
          indices.begin(),
          indices.end(),
          [](Val* v) { return v->dtype() == DataType::Int; }),
      "Cannot index with a value other than an int.");
  indices_.erase(
      std::remove_if(
          indices_.begin(),
          indices_.end(),
          [](Val* index) { return index->isZeroInt(); }),
      indices_.end());
  // If indices becomes empty, just put one ZeroInt
  if (indices_.empty()) {
    indices_.push_back(FusionGuard::getCurFusion()->zeroVal());
  }
}

Sync::Sync(IrBuilderPasskey passkey, bool war_sync)
    : Expr(passkey, ExprType::Sync), war_sync_(war_sync) {
  TORCH_INTERNAL_ASSERT(
      passkey.ir_container_->isA<kir::Kernel>(),
      "IR type only valid for Kernel container.");
}

InitMagicZero::InitMagicZero(IrBuilderPasskey passkey)
    : Expr(passkey, ExprType::InitMagicZero) {
  TORCH_INTERNAL_ASSERT(
      passkey.ir_container_->isA<kir::Kernel>(),
      "IR type only valid for Kernel container.");
}

UpdateMagicZero::UpdateMagicZero(IrBuilderPasskey passkey)
    : Expr(passkey, ExprType::UpdateMagicZero) {
  TORCH_INTERNAL_ASSERT(
      passkey.ir_container_->isA<kir::Kernel>(),
      "IR type only valid for Kernel container.");
}

void Scope::insert(std::vector<Expr*>::const_iterator pos, Expr* expr) {
  exprs_.insert(pos, expr);
}

void Scope::insert_before(Expr* ref, Expr* expr) {
  const auto it = std::find(exprs_.begin(), exprs_.end(), ref);
  TORCH_INTERNAL_ASSERT(
      it != exprs_.end(),
      "Tried to insert ",
      expr,
      " before the reference: ",
      ref,
      " however the reference was not found in this scope.");
  insert(it, expr);
}

void Scope::insert_after(Expr* ref, Expr* expr) {
  const auto it = std::find(exprs_.begin(), exprs_.end(), ref);
  TORCH_INTERNAL_ASSERT(
      it != exprs_.end(),
      "Tried to insert ",
      expr,
      " after the reference: ",
      ref,
      " however the reference was not found in this scope.");
  insert(it + 1, expr);
}

void Scope::insert(size_t pos, Expr* expr) {
  const auto it = exprs_.begin() + pos;
  insert(it, expr);
}

void Scope::erase(std::vector<Expr*>::const_iterator pos) {
  // Remove the scope of the expr if this is the scope
  auto expr = *pos;
  exprs_.erase(pos);
}

void Scope::erase(Expr* ref) {
  const auto it = std::find(exprs_.begin(), exprs_.end(), ref);
  if (it != exprs_.end()) {
    erase(it);
  }
}

void Scope::erase(size_t pos) {
  TORCH_INTERNAL_ASSERT(pos < size());
  erase(exprs_.begin() + pos);
}

bool Scope::contains(Expr* expr) const {
  const auto it = std::find(exprs_.begin(), exprs_.end(), expr);
  return it != exprs_.end();
}

void Scope::clear() {
  exprs_.clear();
}

ForLoop::ForLoop(
    IrBuilderPasskey passkey,
    IterDomain* iter_domain,
    Val* index,
    Val* start,
    Val* stop,
    Val* step,
    bool vectorize,
    Val* vectorize_shift,
    bool unroll_required)
    : Expr(passkey, ExprType::ForLoop),
      iter_domain_{iter_domain},
      index_(index),
      start_(start),
      stop_(stop),
      step_(step),
      vectorize_(vectorize),
      vectorize_shift_(vectorize_shift),
      unroll_required_(unroll_required),
      body_(this) {
  TORCH_INTERNAL_ASSERT(
      passkey.ir_container_->isA<kir::Kernel>(),
      "IR type only valid for Kernel container.");
  TORCH_INTERNAL_ASSERT(index->dtype() == DataType::Int);
  addInput(index);
  addInput(iter_domain);
  if (start_ == nullptr && iter_domain->isThread()) {
    start_ = NamedScalar::getParallelIndex(iter_domain->getParallelType());
  }
  if (step_ == nullptr) {
    if (iter_domain->isThread()) {
      step_ = NamedScalar::getParallelDim(iter_domain->getParallelType());
    } else {
      step_ = FusionGuard::getCurFusion()->oneVal();
    }
  }
}

ForLoop::ForLoop(IrBuilderPasskey passkey, IterDomain* iter_domain)
    : ForLoop(
          passkey,
          iter_domain,
          iter_domain->isBroadcast() ? FusionGuard::getCurFusion()->zeroVal()
                                     : IrBuilder::create<Int>(c10::nullopt),
          nullptr,
          nullptr,
          nullptr,
          isParallelTypeVectorize(iter_domain->getParallelType()),
          nullptr,
          false) {
  TORCH_INTERNAL_ASSERT(
      passkey.ir_container_->isA<kir::Kernel>(),
      "IR type only valid for Kernel container.");
}

ForLoop::ForLoop(IrBuilderPasskey passkey, const ForLoop* other)
    : ForLoop(
          passkey,
          other->iter_domain(),
          other->index(),
          other->start(),
          other->stop(),
          other->step(),
          other->vectorize(),
          other->vectorize_shift(),
          other->isUnrollRequired()) {
  TORCH_INTERNAL_ASSERT(
      passkey.ir_container_->isA<kir::Kernel>(),
      "IR type only valid for Kernel container.");
}

bool ForLoop::isUnrollable() const {
  // Start and stop must be constant, must not be a broadcast
  // dimension, cannot be bound to a parallel dimension, must not be
  // vectorized.
  return start()->isConstScalar() && stop()->isConstScalar() &&
      !iter_domain()->isThread() && !iter_domain()->isBroadcast() &&
      !vectorize();
}

bool ForLoop::isUnrolled() const {
  if (isUnrollRequired() && !isUnrollable()) {
    TORCH_WARN(
        "Unroll required but not possible. Register allocation disabled. Loop index: ",
        index_->toString());
    return false;
  }

  // Size-one loop will not be materialized as a loop, so return false
  if (start()->isZeroInt() && stop()->isOneInt()) {
    return false;
  }

  // Unroll if required.
  if (isUnrollRequired()) {
    return true;
  }

  // Don't unroll if not possible
  if (!isUnrollable()) {
    return false;
  }

  // Unrolling is technically possible but avoided
  if (iter_domain()->getParallelType() == ParallelType::Unswitch) {
    // Use ParallelType::Unroll if unrolling is desired. Note that
    // unswitched size-one loops are not unrolled as they are not
    // materialized as actual for-loops.
    return false;
  }

  return true;
}

Val* ForLoop::start() const {
  if (start_ != nullptr) {
    return start_;
  } else {
    // clang-tidy complains without this
    TORCH_INTERNAL_ASSERT(iter_domain_ != nullptr);
    return iter_domain_->start();
  }
}

Val* ForLoop::stop() const {
  if (stop_ != nullptr) {
    return stop_;
  } else {
    // clang-tidy complains without this
    TORCH_INTERNAL_ASSERT(iter_domain_ != nullptr);
    return iter_domain_->extent();
  }
}

Val* ForLoop::step() const {
  TORCH_INTERNAL_ASSERT(step_ != nullptr);
  return step_;
}

IfThenElse::IfThenElse(IrBuilderPasskey passkey, Predicate* cond)
    : Expr(passkey, ExprType::IfThenElse), then_body_(this), else_body_(this) {
  setPredicate(cond);
  addInput(cond);
}

Val* TensorIndex::index(int i) const {
  TORCH_INTERNAL_ASSERT(
      nDims() > 0, "Tried to get an index of a 0-dim TensorIndex");
  if (i < 0)
    i += nDims();
  TORCH_INTERNAL_ASSERT(i >= 0 && i < int(nDims()));
  return indices_[i];
}

Allocate::Allocate(
    IrBuilderPasskey passkey,
    Val* buffer,
    MemoryType memory_type,
    std::vector<Val*> shape,
    bool zero_init)
    : Expr(passkey, ExprType::Allocate),
      buffer_(buffer),
      memory_type_(memory_type),
      shape_(std::move(shape)),
      zero_init_(zero_init) {
  TORCH_INTERNAL_ASSERT(
      passkey.ir_container_->isA<kir::Kernel>(),
      "IR type only valid for Kernel container.");
  if (!shape_.empty()) {
    TORCH_INTERNAL_ASSERT(
        (shape_.size() == 1 && shape_[0]->isOneInt()) ||
        buffer_->isA<TensorView>());
  } else {
    TORCH_INTERNAL_ASSERT(buffer_->isA<TensorView>());
    TORCH_INTERNAL_ASSERT(
        buffer_->as<TensorView>()->getMemoryType() == memory_type_);
    const auto domain = buffer_->as<TensorView>()->domain();
    for (auto axis : domain->noReductions()) {
      shape_.push_back(axis->extent());
    }
  }

  for (auto s : shape_) {
    if (size_ == nullptr) {
      size_ = s;
    } else {
      size_ = IrBuilder::mulExpr(size_, s);
    }
  }

  if (size_ == nullptr) {
    size_ = FusionGuard::getCurFusion()->oneVal();
  }

  addInput(size_);
}

Allocate::Allocate(
    IrBuilderPasskey passkey,
    Val* buffer,
    MemoryType memory_type,
    Val* size,
    bool zero_init)
    : Allocate(
          passkey,
          buffer,
          memory_type,
          size == nullptr ? std::vector<Val*>{} : std::vector<Val*>{size},
          zero_init) {
  TORCH_INTERNAL_ASSERT(
      passkey.ir_container_->isA<kir::Kernel>(),
      "IR type only valid for Kernel container.");
}

GridReduction::GridReduction(
    IrBuilderPasskey passkey,
    ReductionOp* reduction_op,
    Allocate* reduction_buffer,
    Allocate* sync_buffer)
    : Expr(passkey, ExprType::GridReduction),
      reduction_op_(reduction_op),
      reduction_buffer_(reduction_buffer),
      sync_buffer_(sync_buffer) {
  TORCH_INTERNAL_ASSERT(
      passkey.ir_container_->isA<kir::Kernel>(),
      "IR type only valid for Kernel container.");
}

GridBroadcast::GridBroadcast(
    IrBuilderPasskey passkey,
    BroadcastOp* broadcast_op,
    Allocate* broadcast_buffer,
    Allocate* sync_buffer)
    : Expr(passkey, ExprType::GridBroadcast),
      broadcast_op_(broadcast_op),
      broadcast_buffer_(broadcast_buffer),
      sync_buffer_(sync_buffer) {
  TORCH_INTERNAL_ASSERT(
      passkey.ir_container_->isA<kir::Kernel>(),
      "IR type only valid for Kernel container.");
}

GridWelford::GridWelford(
    IrBuilderPasskey passkey,
    WelfordOp* welford_op,
    Allocate* var_buffer,
    Allocate* avg_buffer,
    Allocate* n_buffer,
    Allocate* sync_buffer)
    : Expr(passkey, ExprType::GridWelford),
      welford_op_(welford_op),
      var_buffer_(var_buffer),
      avg_buffer_(avg_buffer),
      n_buffer_(n_buffer),
      sync_buffer_(sync_buffer) {
  TORCH_INTERNAL_ASSERT(
      passkey.ir_container_->isA<kir::Kernel>(),
      "IR type only valid for Kernel container.");
}

} // namespace kir
} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
