#include <torch/csrc/jit/codegen/cuda/expr_evaluator.h>
#include <torch/csrc/jit/codegen/cuda/ir_builder.h>
#include <torch/csrc/jit/codegen/cuda/ir_cloner.h>
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

Allocate::Allocate(
    IrBuilderPasskey passkey,
    Val* buffer,
    MemoryType memory_type,
    std::vector<Val*> shape,
    bool zero_init,
    Allocate* alias)
    : Expr(passkey) {
  TORCH_INTERNAL_ASSERT(
      passkey.ir_container_->isA<kir::Kernel>(),
      "IR type only valid for Kernel container.");
  if (!shape.empty()) {
    TORCH_INTERNAL_ASSERT(
        (shape.size() == 1 && shape[0]->isOneInt()) ||
        buffer->isA<TensorView>());
  } else {
    TORCH_INTERNAL_ASSERT(buffer->isA<TensorView>());
    TORCH_INTERNAL_ASSERT(
        buffer->as<TensorView>()->getMemoryType() == memory_type);
    const auto domain = buffer->as<TensorView>()->domain();
    for (auto axis : domain->noReductions()) {
      shape.push_back(axis->extent());
    }
  }

  Val* size = nullptr;
  for (auto s : shape) {
    if (size == nullptr) {
      size = s;
    } else {
      size = IrBuilder::mulExpr(size, s);
    }
  }

  if (size == nullptr) {
    size = FusionGuard::getCurFusion()->oneVal();
  }

  if (alias != nullptr) {
    TORCH_INTERNAL_ASSERT(alias != this, "Invalid alias");
    TORCH_INTERNAL_ASSERT(alias->memoryType() == memory_type, "Invalid alias");
  }

  addInput(size);
  addAttribute(buffer);
  addAttribute(IrBuilder::create<Attribute<MemoryType>>(
      passkey.ir_container_, memory_type));
  addAttribute(
      IrBuilder::create<Attribute<bool>>(passkey.ir_container_, zero_init));

  addAttribute(alias);

  for (auto s : shape) {
    addAttribute(s);
  }
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

NVFUSER_DEFINE_CLONE_AND_CREATE(Allocate)

BlockSync::BlockSync(IrBuilderPasskey passkey, bool war_sync) : Expr(passkey) {
  TORCH_INTERNAL_ASSERT(
      passkey.ir_container_->isA<kir::Kernel>(),
      "IR type only valid for Kernel container.");
  addAttribute(
      IrBuilder::create<Attribute<bool>>(passkey.ir_container_, war_sync));
}

NVFUSER_DEFINE_CLONE_AND_CREATE(BlockSync)

GridSync::GridSync(
    IrBuilderPasskey passkey,
    ParallelTypeBitmap sync_dims,
    Val* sync_buffer)
    : Expr(passkey) {
  addAttribute(IrBuilder::create<Attribute<ParallelTypeBitmap>>(
      passkey.ir_container_, sync_dims));
  addAttribute(sync_buffer);
}

NVFUSER_DEFINE_CLONE_AND_CREATE(GridSync)

CpAsyncWait::CpAsyncWait(IrBuilderPasskey passkey, unsigned int keep_stages)
    : Expr(passkey) {
  TORCH_INTERNAL_ASSERT(
      passkey.ir_container_->isA<kir::Kernel>(),
      "IR type only valid for Kernel container.");
  addAttribute(IrBuilder::create<Attribute<unsigned int>>(
      passkey.ir_container_, keep_stages));
}

NVFUSER_DEFINE_CLONE_AND_CREATE(CpAsyncWait)

CpAsyncCommit::CpAsyncCommit(IrBuilderPasskey passkey) : Expr(passkey) {
  TORCH_INTERNAL_ASSERT(
      passkey.ir_container_->isA<kir::Kernel>(),
      "IR type only valid for Kernel container.");
}

NVFUSER_DEFINE_CLONE_AND_CREATE(CpAsyncCommit)

InitMagicZero::InitMagicZero(IrBuilderPasskey passkey) : Expr(passkey) {
  TORCH_INTERNAL_ASSERT(
      passkey.ir_container_->isA<kir::Kernel>(),
      "IR type only valid for Kernel container.");
}

NVFUSER_DEFINE_CLONE_AND_CREATE(InitMagicZero)

UpdateMagicZero::UpdateMagicZero(IrBuilderPasskey passkey) : Expr(passkey) {
  TORCH_INTERNAL_ASSERT(
      passkey.ir_container_->isA<kir::Kernel>(),
      "IR type only valid for Kernel container.");
}

NVFUSER_DEFINE_CLONE_AND_CREATE(UpdateMagicZero)

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
    : Expr(passkey) {
  TORCH_INTERNAL_ASSERT(
      passkey.ir_container_->isA<kir::Kernel>(),
      "IR type only valid for Kernel container.");
  TORCH_INTERNAL_ASSERT(index->dtype() == DataType::Int);
  addInput(index);
  addInput(iter_domain);
  if (start == nullptr && iter_domain->isThread()) {
    start = NamedScalar::getParallelIndex(iter_domain->getParallelType());
  }
  if (step == nullptr) {
    if (iter_domain->isThread()) {
      step = NamedScalar::getParallelDim(iter_domain->getParallelType());
    } else {
      step = FusionGuard::getCurFusion()->oneVal();
    }
  }
  addAttribute(start);
  addAttribute(stop);
  addAttribute(step);
  addAttribute(
      IrBuilder::create<Attribute<bool>>(passkey.ir_container_, vectorize));
  addAttribute(vectorize_shift);
  addAttribute(IrBuilder::create<Attribute<bool>>(
      passkey.ir_container_, unroll_required));
  addAttribute(IrBuilder::create<Attribute<DoubleBufferLoopStage>>(
      passkey.ir_container_, double_buffer_loop_stage));
  // Storing IR nodes as Attribute is not safe with IrCloner, but fortunately
  // kernel IR does not need this feature.
  addAttribute(
      IrBuilder::create<Attribute<Scope>>(passkey.ir_container_, this));
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

NVFUSER_DEFINE_CLONE_AND_CREATE(ForLoop)

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
        index()->toString());
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
  if (attributeVal(0) != nullptr) {
    return attributeVal(0);
  } else {
    // clang-tidy complains without this
    TORCH_INTERNAL_ASSERT(iter_domain() != nullptr);
    return iter_domain()->start();
  }
}

Val* ForLoop::stop() const {
  if (attributeVal(1) != nullptr) {
    return attributeVal(1);
  } else {
    // clang-tidy complains without this
    TORCH_INTERNAL_ASSERT(iter_domain() != nullptr);
    return iter_domain()->extent();
  }
}

Val* ForLoop::step() const {
  TORCH_INTERNAL_ASSERT(attributeVal(2) != nullptr);
  return attributeVal(2);
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
    : Expr(passkey) {
  setPredicate(cond);
  addInput(cond);
  // Storing IR nodes as Attribute is not safe with IrCloner, but fortunately
  // kernel IR does not need this feature.
  addAttribute(
      IrBuilder::create<Attribute<Scope>>(passkey.ir_container_, this));
  addAttribute(
      IrBuilder::create<Attribute<Scope>>(passkey.ir_container_, this));
}

NVFUSER_DEFINE_CLONE_AND_CREATE(IfThenElse)

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
    : ReductionOp(passkey, reduction_op_type, init, out, in, is_allreduce) {
  TORCH_INTERNAL_ASSERT(
      passkey.ir_container_->isA<kir::Kernel>(),
      "IR type only valid for Kernel container.");
  TORCH_INTERNAL_ASSERT(
      attributes().size() == num_reduction_op_attr,
      "The num_reduction_op_attr does not match the number of attributes ReductionOp has."
      "If you changed ReductionOp, please change num_reduction_op_attr accordingly.");
  addAttribute(reduction_buffer);
  addAttribute(sync_buffer);
  addAttribute(entrance_index);
  addAttribute(entrances);
  addAttribute(
      IrBuilder::create<Attribute<ParallelTypeBitmap>>(passkey.ir_container_));
}

NVFUSER_DEFINE_CLONE_AND_CREATE(GridReduction)

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
          is_allreduce) {
  TORCH_INTERNAL_ASSERT(
      passkey.ir_container_->isA<kir::Kernel>(),
      "IR type only valid for Kernel container.");
  TORCH_INTERNAL_ASSERT(
      attributes().size() == numGroupedReductionOpAttr(),
      "The numGroupedReductionOpAttr() does not match the number of attributes GroupedReductionOp has."
      "If you changed GroupedReductionOp, please change numGroupedReductionOpAttr() accordingly.");
  addAttribute(sync_buffer);
  addAttribute(entrance_index);
  addAttribute(entrances);
  addAttribute(buffer_stride);
  addAttribute(
      IrBuilder::create<Attribute<ParallelTypeBitmap>>(passkey.ir_container_));
  for (auto buffer : reduction_buffers) {
    addAttribute(buffer);
  }
}

NVFUSER_DEFINE_CLONE_AND_CREATE(GroupedGridReduction)

GridBroadcast::GridBroadcast(
    IrBuilderPasskey passkey,
    BroadcastOp* broadcast_op,
    Allocate* broadcast_buffer,
    Allocate* sync_buffer)
    : Expr(passkey) {
  TORCH_INTERNAL_ASSERT(
      passkey.ir_container_->isA<kir::Kernel>(),
      "IR type only valid for Kernel container.");
  addAttribute(broadcast_op);
  addAttribute(broadcast_buffer);
  addAttribute(sync_buffer);
}

NVFUSER_DEFINE_CLONE_AND_CREATE(GridBroadcast)

GridWelford::GridWelford(
    IrBuilderPasskey passkey,
    WelfordOp* welford_op,
    Allocate* var_buffer,
    Allocate* avg_buffer,
    Allocate* n_buffer,
    Allocate* sync_buffer,
    Val* entrance_index,
    Val* entrances)
    : Expr(passkey) {
  TORCH_INTERNAL_ASSERT(
      passkey.ir_container_->isA<kir::Kernel>(),
      "IR type only valid for Kernel container.");
  addAttribute(welford_op);
  addAttribute(var_buffer);
  addAttribute(avg_buffer);
  addAttribute(n_buffer);
  addAttribute(sync_buffer);
  addAttribute(entrance_index);
  addAttribute(entrances);
  addAttribute(
      IrBuilder::create<Attribute<ParallelTypeBitmap>>(passkey.ir_container_));
}

NVFUSER_DEFINE_CLONE_AND_CREATE(GridWelford)

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
          is_allreduce) {
  TORCH_INTERNAL_ASSERT(
      passkey.ir_container_->isA<kir::Kernel>(),
      "IR type only valid for Kernel container.");
  TORCH_INTERNAL_ASSERT(
      attributes().size() == numGroupedWelfordOpAttr(),
      "The numGroupedWelfordOpAttr() does not match the number of attributes GroupedWelfordOp has."
      "If you changed GroupedReductionOp, please change numGroupedWelfordOpAttr() accordingly.");
  addAttribute(sync_buffer);
  addAttribute(entrance_index);
  addAttribute(entrances);
  addAttribute(buffer_stride);
  addAttribute(
      IrBuilder::create<Attribute<ParallelTypeBitmap>>(passkey.ir_container_));
  TORCH_INTERNAL_ASSERT(
      reduction_buffers[0].size() == reduction_buffers[1].size());
  TORCH_INTERNAL_ASSERT(
      reduction_buffers[0].size() == reduction_buffers[2].size());
  for (auto i : c10::irange(reduction_buffers[0].size())) {
    addAttribute(reduction_buffers[0].at(i));
    addAttribute(reduction_buffers[1].at(i));
    addAttribute(reduction_buffers[2].at(i));
  }
}

NVFUSER_DEFINE_CLONE_AND_CREATE(GroupedGridWelford)

VectorizedWelfordOp::VectorizedWelfordOp(
    IrBuilderPasskey passkey,
    const WelfordTriplet& output,
    const WelfordTriplet& input,
    const WelfordTriplet& init,
    Val* count,
    Val* reciprocal_of_count,
    Bool* hoisted_predicate)
    : WelfordOp(passkey, output, input, init, false) {
  addAttribute(count);
  addAttribute(reciprocal_of_count);
  addAttribute(hoisted_predicate);
}

NVFUSER_DEFINE_CLONE_AND_CREATE(VectorizedWelfordOp)

AllocateFusedReduction::AllocateFusedReduction(
    IrBuilderPasskey passkey,
    Expr* grid_expr)
    : Expr(passkey) {
  TORCH_INTERNAL_ASSERT(
      passkey.ir_container_->isA<kir::Kernel>(),
      "IR type only valid for Kernel container.");
  addAttribute(grid_expr);
}

NVFUSER_DEFINE_CLONE_AND_CREATE(AllocateFusedReduction)

TensorIndex* AllocateFusedReduction::out() const {
  TORCH_INTERNAL_ASSERT(gridExpr() != nullptr);
  if (gridExpr()->isA<GridReduction>() ||
      gridExpr()->isA<GroupedGridReduction>()) {
    return gridExpr()->outputs().at(0)->as<kir::TensorIndex>();
  } else if (auto grid_welford = dynamic_cast<GridWelford*>(gridExpr())) {
    return grid_welford->welford_op()->out()->as<kir::TensorIndex>();
  } else if (
      auto grouped_grid_welford =
          dynamic_cast<GroupedGridWelford*>(gridExpr())) {
    return grouped_grid_welford->out(0)->as<kir::TensorIndex>();
  } else {
    TORCH_INTERNAL_ASSERT(
        false, "Invalid grid expression: ", gridExpr()->toString());
  }
}

const ParallelTypeBitmap& AllocateFusedReduction::threadPredicate() const {
  TORCH_INTERNAL_ASSERT(gridExpr() != nullptr);
  if (auto grid_reduction = dynamic_cast<GridReduction*>(gridExpr())) {
    return grid_reduction->threadPredicate();
  } else if (auto grid_welford = dynamic_cast<GridWelford*>(gridExpr())) {
    return grid_welford->threadPredicate();
  } else if (
      auto grouped_grid_reduction =
          dynamic_cast<GroupedGridReduction*>(gridExpr())) {
    return grouped_grid_reduction->threadPredicate();
  } else if (
      auto grouped_grid_welford =
          dynamic_cast<GroupedGridWelford*>(gridExpr())) {
    return grouped_grid_welford->threadPredicate();
  } else {
    TORCH_INTERNAL_ASSERT(
        false, "Invalid grid expression: ", gridExpr()->toString());
  }
}

} // namespace kir
} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
