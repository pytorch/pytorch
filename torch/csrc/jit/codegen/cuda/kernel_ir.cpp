#include <torch/csrc/jit/codegen/cuda/expr_evaluator.h>
#include <torch/csrc/jit/codegen/cuda/ir_builder.h>
#include <torch/csrc/jit/codegen/cuda/kernel.h>
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

Val* TensorIndex::index(int i) const {
  TORCH_INTERNAL_ASSERT(
      nDims() > 0, "Tried to get an index of a 0-dim TensorIndex");
  if (i < 0)
    i += nDims();
  TORCH_INTERNAL_ASSERT(i >= 0 && i < int(nDims()));
  return indices_[i];
}

BlockSync::BlockSync(IrBuilderPasskey passkey, bool war_sync)
    : Expr(passkey, ExprType::BlockSync), war_sync_(war_sync) {
  TORCH_INTERNAL_ASSERT(
      passkey.ir_container_->isA<kir::Kernel>(),
      "IR type only valid for Kernel container.");
}

Expr* BlockSync::shallowCopy() const {
  auto result = IrBuilder::create<BlockSync>(war_sync_);
  result->copyPredicatesFrom(this);
  return result;
}

GridSync::GridSync(
    IrBuilderPasskey passkey,
    ParallelTypeBitmap sync_dims,
    Val* sync_buffer)
    : Expr(passkey, ExprType::GridSync),
      sync_dims_(sync_dims),
      sync_buffer_(sync_buffer) {}

Expr* GridSync::shallowCopy() const {
  auto result = IrBuilder::create<GridSync>(sync_dims_, sync_buffer_);
  result->copyPredicatesFrom(this);
  return result;
}

CpAsyncWait::CpAsyncWait(IrBuilderPasskey passkey, unsigned int keep_stages)
    : Expr(passkey, ExprType::CpAsyncWait), keep_stages_(keep_stages) {
  TORCH_INTERNAL_ASSERT(
      passkey.ir_container_->isA<kir::Kernel>(),
      "IR type only valid for Kernel container.");
}

Expr* CpAsyncWait::shallowCopy() const {
  auto result = IrBuilder::create<CpAsyncWait>(keep_stages_);
  result->copyPredicatesFrom(this);
  return result;
}

CpAsyncCommit::CpAsyncCommit(IrBuilderPasskey passkey)
    : Expr(passkey, ExprType::CpAsyncCommit) {
  TORCH_INTERNAL_ASSERT(
      passkey.ir_container_->isA<kir::Kernel>(),
      "IR type only valid for Kernel container.");
}

Expr* CpAsyncCommit::shallowCopy() const {
  auto result = IrBuilder::create<CpAsyncCommit>();
  result->copyPredicatesFrom(this);
  return result;
}

InitMagicZero::InitMagicZero(IrBuilderPasskey passkey)
    : Expr(passkey, ExprType::InitMagicZero) {
  TORCH_INTERNAL_ASSERT(
      passkey.ir_container_->isA<kir::Kernel>(),
      "IR type only valid for Kernel container.");
}

Expr* InitMagicZero::shallowCopy() const {
  auto result = IrBuilder::create<InitMagicZero>();
  result->copyPredicatesFrom(this);
  return result;
}

UpdateMagicZero::UpdateMagicZero(IrBuilderPasskey passkey)
    : Expr(passkey, ExprType::UpdateMagicZero) {
  TORCH_INTERNAL_ASSERT(
      passkey.ir_container_->isA<kir::Kernel>(),
      "IR type only valid for Kernel container.");
}

Expr* UpdateMagicZero::shallowCopy() const {
  auto result = IrBuilder::create<UpdateMagicZero>();
  result->copyPredicatesFrom(this);
  return result;
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
  C10_UNUSED auto expr = *pos;
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
    bool unroll_required,
    DoubleBufferLoopStage double_buffer_loop_stage)
    : Expr(passkey, ExprType::ForLoop),
      iter_domain_{iter_domain},
      index_(index),
      start_(start),
      stop_(stop),
      step_(step),
      vectorize_(vectorize),
      vectorize_shift_(vectorize_shift),
      unroll_required_(unroll_required),
      body_(this),
      double_buffer_loop_stage_(double_buffer_loop_stage) {
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
          GpuLower::current()->caMap()->getIndexVariable(iter_domain),
          nullptr,
          nullptr,
          nullptr,
          !iter_domain->isBroadcast() &&
              isParallelTypeVectorize(iter_domain->getParallelType()),
          nullptr,
          false,
          DoubleBufferLoopStage::NotApplicable) {
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
          other->isUnrollRequired(),
          other->doubleBufferLoopStage()) {
  TORCH_INTERNAL_ASSERT(
      passkey.ir_container_->isA<kir::Kernel>(),
      "IR type only valid for Kernel container.");
}

Expr* ForLoop::shallowCopy() const {
  auto result = IrBuilder::create<ForLoop>(
      iter_domain_,
      index_,
      start_,
      stop_,
      step_,
      vectorize_,
      vectorize_shift_,
      unroll_required_,
      double_buffer_loop_stage_);
  result->body_ = body_;
  result->copyPredicatesFrom(this);
  return result;
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

bool ForLoop::isTrivial() const {
  // These loops are not materialized
  if (vectorize() || iter_domain()->isBroadcast() ||
      iter_domain()->isStride() || iter_domain()->isMma()) {
    return true;
  }

  // By default, a parallelized loop would look like:
  //
  //   for (int x = threadIdx.x; x < stop; x += blockDim.x) {
  //     do_some_comp(x);
  //   }
  //
  // When stop is guaranteed to be smaller or equal to the number of
  // threads, the for-loop is not necessary. In the above case, we
  // would just generate the loop body without the for clause but
  // references to the loop index replaced by the loop start value.
  //
  // When the loop end is the same as the IterDomain extent, the
  // assumption can be safely made. This is more conservative than
  // necessary since the loop stop value just needs to be <= the
  // IterDomain extent. However, at this point, this conservative
  // analysis seems sufficient.
  if (stop() == iter_domain()->extent() && iter_domain()->isThread()) {
    return true;
  }

  // Extent-1 loop: for (int i = 0; i < 1; ++i) {
  if (start()->isZeroInt() && stop()->isOneInt() && step()->isOneInt()) {
    return true;
  }

  // Another extent-1 loop: for (int i = N - 1; i < N; ++i) {
  if (start()->definition() != nullptr &&
      start()->definition()->isA<BinaryOp>() &&
      start()->definition()->as<BinaryOp>()->getBinaryOpType() ==
          BinaryOpType::Sub &&
      start()->definition()->as<BinaryOp>()->lhs() == stop() &&
      start()->definition()->as<BinaryOp>()->rhs()->isOneInt()) {
    return true;
  }

  return false;
}

IfThenElse::IfThenElse(IrBuilderPasskey passkey, Predicate* cond)
    : Expr(passkey, ExprType::IfThenElse), then_body_(this), else_body_(this) {
  setPredicate(cond);
  addInput(cond);
}

Expr* IfThenElse::shallowCopy() const {
  auto result = IrBuilder::create<IfThenElse>(predicate());
  result->then_body_ = then_body_;
  result->else_body_ = else_body_;
  result->setWritePredicate(writePredicate());
  return result;
}

Allocate::Allocate(
    IrBuilderPasskey passkey,
    Val* buffer,
    MemoryType memory_type,
    std::vector<Val*> shape,
    bool zero_init,
    const Allocate* alias)
    : Expr(passkey, ExprType::Allocate),
      buffer_(buffer),
      memory_type_(memory_type),
      shape_(std::move(shape)),
      zero_init_(zero_init),
      alias_(alias) {
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

  if (alias_ != nullptr) {
    TORCH_INTERNAL_ASSERT(alias_ != this, "Invalid alias");
    TORCH_INTERNAL_ASSERT(
        alias_->memoryType() == memory_type_, "Invalid alias");
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

Expr* Allocate::shallowCopy() const {
  auto result =
      IrBuilder::create<Allocate>(buffer_, memory_type_, shape_, zero_init_);
  result->copyPredicatesFrom(this);
  return result;
}

GridReduction::GridReduction(
    IrBuilderPasskey passkey,
    BinaryOpType reduction_op_type,
    Val* init,
    Val* out,
    Val* in,
    Allocate* reduction_buffer,
    Allocate* sync_buffer,
    Val* entrance_index,
    Val* entrances,
    bool is_allreduce)
    : ReductionOp(
          passkey,
          reduction_op_type,
          init,
          out,
          in,
          is_allreduce,
          ExprType::GridReduction),
      reduction_buffer_(reduction_buffer),
      sync_buffer_(sync_buffer),
      entrance_index_(entrance_index),
      entrances_(entrances) {
  TORCH_INTERNAL_ASSERT(
      passkey.ir_container_->isA<kir::Kernel>(),
      "IR type only valid for Kernel container.");
}

Expr* GridReduction::shallowCopy() const {
  auto result = IrBuilder::create<GridReduction>(
      getReductionOpType(),
      init(),
      out(),
      in(),
      reduction_buffer_,
      sync_buffer_,
      entrance_index_,
      entrances_,
      isAllreduce());
  result->copyPredicatesFrom(this);
  result->thread_predicate_ = thread_predicate_;
  return result;
}

GroupedGridReduction::GroupedGridReduction(
    IrBuilderPasskey passkey,
    std::vector<BinaryOpType> reduction_op_types,
    std::vector<Val*> init_vals,
    std::vector<Val*> outputs,
    std::vector<Val*> inputs,
    std::vector<Allocate*> reduction_buffers,
    Allocate* sync_buffer,
    Val* entrance_index,
    Val* entrances,
    Val* buffer_stride,
    bool is_allreduce)
    : GroupedReductionOp(
          passkey,
          std::move(reduction_op_types),
          std::move(init_vals),
          std::move(outputs),
          std::move(inputs),
          is_allreduce,
          ExprType::GroupedGridReduction),
      reduction_buffers_(std::move(reduction_buffers)),
      sync_buffer_(sync_buffer),
      entrance_index_(entrance_index),
      entrances_(entrances),
      buffer_stride_(buffer_stride) {
  TORCH_INTERNAL_ASSERT(
      passkey.ir_container_->isA<kir::Kernel>(),
      "IR type only valid for Kernel container.");
}

Expr* GroupedGridReduction::shallowCopy() const {
  auto result = IrBuilder::create<GroupedGridReduction>(
      getReductionOpTypes(),
      initVals(),
      outputs(),
      inputs(),
      reduction_buffers_,
      sync_buffer_,
      entrance_index_,
      entrances_,
      buffer_stride_,
      isAllreduce());
  result->copyPredicatesFrom(this);
  result->thread_predicate_ = thread_predicate_;
  return result;
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

Expr* GridBroadcast::shallowCopy() const {
  auto result = IrBuilder::create<GridBroadcast>(
      broadcast_op_, broadcast_buffer_, sync_buffer_);
  result->copyPredicatesFrom(this);
  return result;
}

GridWelford::GridWelford(
    IrBuilderPasskey passkey,
    WelfordOp* welford_op,
    Allocate* var_buffer,
    Allocate* avg_buffer,
    Allocate* n_buffer,
    Allocate* sync_buffer,
    Val* entrance_index,
    Val* entrances)
    : Expr(passkey, ExprType::GridWelford),
      welford_op_(welford_op),
      var_buffer_(var_buffer),
      avg_buffer_(avg_buffer),
      n_buffer_(n_buffer),
      sync_buffer_(sync_buffer),
      entrance_index_(entrance_index),
      entrances_(entrances) {
  TORCH_INTERNAL_ASSERT(
      passkey.ir_container_->isA<kir::Kernel>(),
      "IR type only valid for Kernel container.");
}

Expr* GridWelford::shallowCopy() const {
  auto result = IrBuilder::create<GridWelford>(
      welford_op_,
      var_buffer_,
      avg_buffer_,
      n_buffer_,
      sync_buffer_,
      entrance_index_,
      entrances_);
  result->copyPredicatesFrom(this);
  result->thread_predicate_ = thread_predicate_;
  return result;
}

GroupedGridWelford::GroupedGridWelford(
    IrBuilderPasskey passkey,
    std::vector<WelfordTriplet> output_vals,
    std::vector<WelfordTriplet> input_vals,
    std::vector<WelfordTriplet> init_vals,
    std::array<std::vector<Allocate*>, 3> reduction_buffers,
    Allocate* sync_buffer,
    Val* entrance_index,
    Val* entrances,
    Val* buffer_stride,
    bool is_allreduce)
    : GroupedWelfordOp(
          passkey,
          std::move(output_vals),
          std::move(input_vals),
          std::move(init_vals),
          is_allreduce,
          ExprType::GroupedGridWelford),
      reduction_buffers_(std::move(reduction_buffers)),
      sync_buffer_(sync_buffer),
      entrance_index_(entrance_index),
      entrances_(entrances),
      buffer_stride_(buffer_stride) {
  TORCH_INTERNAL_ASSERT(
      passkey.ir_container_->isA<kir::Kernel>(),
      "IR type only valid for Kernel container.");
}

Expr* GroupedGridWelford::shallowCopy() const {
  auto result = IrBuilder::create<GroupedGridWelford>(
      outputVals(),
      inputVals(),
      initVals(),
      reduction_buffers_,
      sync_buffer_,
      entrance_index_,
      entrances_,
      buffer_stride_,
      isAllreduce());
  result->copyPredicatesFrom(this);
  result->thread_predicate_ = thread_predicate_;
  return result;
}

AllocateFusedReduction::AllocateFusedReduction(
    IrBuilderPasskey passkey,
    GridReduction* grid_reduction)
    : Expr(passkey, ExprType::AllocateFusedReduction),
      grid_expr_(grid_reduction) {
  TORCH_INTERNAL_ASSERT(
      passkey.ir_container_->isA<kir::Kernel>(),
      "IR type only valid for Kernel container.");
}

AllocateFusedReduction::AllocateFusedReduction(
    IrBuilderPasskey passkey,
    GridWelford* grid_welford)
    : Expr(passkey, ExprType::AllocateFusedReduction),
      grid_expr_(grid_welford) {
  TORCH_INTERNAL_ASSERT(
      passkey.ir_container_->isA<kir::Kernel>(),
      "IR type only valid for Kernel container.");
}

AllocateFusedReduction::AllocateFusedReduction(
    IrBuilderPasskey passkey,
    GroupedGridReduction* grouped_grid_reduction)
    : Expr(passkey, ExprType::AllocateFusedReduction),
      grid_expr_(grouped_grid_reduction) {
  TORCH_INTERNAL_ASSERT(
      passkey.ir_container_->isA<kir::Kernel>(),
      "IR type only valid for Kernel container.");
}

AllocateFusedReduction::AllocateFusedReduction(
    IrBuilderPasskey passkey,
    GroupedGridWelford* grouped_grid_welford)
    : Expr(passkey, ExprType::AllocateFusedReduction),
      grid_expr_(grouped_grid_welford) {
  TORCH_INTERNAL_ASSERT(
      passkey.ir_container_->isA<kir::Kernel>(),
      "IR type only valid for Kernel container.");
}

Expr* AllocateFusedReduction::shallowCopy() const {
  if (grid_expr_->isA<GridReduction>()) {
    auto result = IrBuilder::create<AllocateFusedReduction>(
        grid_expr_->as<GridReduction>());
    result->setPredicate(predicate());
    result->setWritePredicate(writePredicate());
    return result;
  } else if (grid_expr_->isA<GridWelford>()) {
    auto result = IrBuilder::create<AllocateFusedReduction>(
        grid_expr_->as<GridWelford>());
    result->setPredicate(predicate());
    result->setWritePredicate(writePredicate());
    return result;
  } else if (grid_expr_->isA<GroupedGridReduction>()) {
    auto result = IrBuilder::create<AllocateFusedReduction>(
        grid_expr_->as<GroupedGridReduction>());
    result->setPredicate(predicate());
    result->setWritePredicate(writePredicate());
    return result;
  } else if (grid_expr_->isA<GroupedGridWelford>()) {
    auto result = IrBuilder::create<AllocateFusedReduction>(
        grid_expr_->as<GroupedGridWelford>());
    result->setPredicate(predicate());
    result->setWritePredicate(writePredicate());
    return result;
  }
  TORCH_INTERNAL_ASSERT(
      false, "Unknown reduction type in AllocateFusedReduction::shallowCopy");
}

TensorIndex* AllocateFusedReduction::out() const {
  TORCH_INTERNAL_ASSERT(grid_expr_ != nullptr);
  if (grid_expr_->isA<GridReduction>() ||
      grid_expr_->isA<GroupedGridReduction>()) {
    return grid_expr_->outputs().at(0)->as<kir::TensorIndex>();
  } else if (auto grid_welford = dynamic_cast<GridWelford*>(grid_expr_)) {
    return grid_welford->welford_op()->out()->as<kir::TensorIndex>();
  } else if (
      auto grouped_grid_welford =
          dynamic_cast<GroupedGridWelford*>(grid_expr_)) {
    return grouped_grid_welford->out(0)->as<kir::TensorIndex>();
  } else {
    TORCH_INTERNAL_ASSERT(
        false, "Invalid grid expression: ", grid_expr_->toString());
  }
}

const ParallelTypeBitmap& AllocateFusedReduction::threadPredicate() const {
  TORCH_INTERNAL_ASSERT(grid_expr_ != nullptr);
  if (auto grid_reduction = dynamic_cast<GridReduction*>(grid_expr_)) {
    return grid_reduction->threadPredicate();
  } else if (auto grid_welford = dynamic_cast<GridWelford*>(grid_expr_)) {
    return grid_welford->threadPredicate();
  } else if (
      auto grouped_grid_reduction =
          dynamic_cast<GroupedGridReduction*>(grid_expr_)) {
    return grouped_grid_reduction->threadPredicate();
  } else if (
      auto grouped_grid_welford =
          dynamic_cast<GroupedGridWelford*>(grid_expr_)) {
    return grouped_grid_welford->threadPredicate();
  } else {
    TORCH_INTERNAL_ASSERT(
        false, "Invalid grid expression: ", grid_expr_->toString());
  }
}

} // namespace kir
} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
