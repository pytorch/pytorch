#include <torch/csrc/jit/codegen/cuda/arith.h>
#include <torch/csrc/jit/codegen/cuda/index_compute.h>
#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>
#include <torch/csrc/jit/codegen/cuda/lower2device.h>
#include <torch/csrc/jit/codegen/cuda/lower_utils.h>
#include <torch/csrc/jit/codegen/cuda/predicate_compute.h>

#include <torch/csrc/jit/codegen/cuda/lower_index.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

Val* IndexLowering::lowerSrcIndex(Val* src, Val* dst) const {
  if (auto tv = dynamic_cast<TensorView*>(src)) {
    TORCH_INTERNAL_ASSERT(dst->isA<TensorView>());
    return Index::getProducerIndex(tv, dst->as<TensorView>(), for_loops_);
  } else {
    return src;
  }
}

Val* IndexLowering::lowerDstIndex(Val* dst) const {
  if (auto tv = dynamic_cast<TensorView*>(dst)) {
    return Index::getConsumerIndex(tv, for_loops_);
  } else {
    return dst;
  }
}

void IndexLowering::pushBack(Expr* expr) {
  if (active_scope_ == nullptr) {
    lowered_exprs_.push_back(expr);
  } else {
    active_scope_->push_back(expr);
  }
}

void IndexLowering::insertAtTopLevel(Expr* expr) {
  TORCH_INTERNAL_ASSERT(!lowered_exprs_.empty());
  lowered_exprs_.insert(lowered_exprs_.end() - 1, expr);
}

void IndexLowering::handle(const kir::IfThenElse* ite) {
  const auto prev_scope = active_scope_;

  auto new_ite = IrBuilder::create<kir::IfThenElse>(ite->predicate());
  pushBack(new_ite);

  active_scope_ = &new_ite->thenBody();

  for (auto expr : ite->thenBody().exprs()) {
    OptOutConstDispatch::handle(expr);
  }

  active_scope_ = &new_ite->elseBody();

  for (auto expr : ite->elseBody().exprs()) {
    OptOutConstDispatch::handle(expr);
  }

  active_scope_ = prev_scope;
}

void IndexLowering::handle(const kir::ForLoop* for_loop) {
  const auto prev_scope = active_scope_;

  auto new_for_loop = IrBuilder::create<kir::ForLoop>(for_loop);
  pushBack(new_for_loop);

  active_scope_ = &new_for_loop->body();
  for_loops_.push_back(new_for_loop);

  for (auto expr : for_loop->body().exprs()) {
    OptOutConstDispatch::handle(expr);
  }

  for_loops_.pop_back();
  active_scope_ = prev_scope;
}

void IndexLowering::handle(const UnaryOp* uop) {
  const auto in = lowerSrcIndex(uop->in(), uop->out());
  const auto out = lowerDstIndex(uop->out());
  pushBack(IrBuilder::create<UnaryOp>(uop->getUnaryOpType(), out, in));
}

void IndexLowering::handle(const BinaryOp* bop) {
  const auto lhs = lowerSrcIndex(bop->lhs(), bop->out());
  const auto rhs = lowerSrcIndex(bop->rhs(), bop->out());
  const auto out = lowerDstIndex(bop->out());
  pushBack(IrBuilder::create<BinaryOp>(bop->getBinaryOpType(), out, lhs, rhs));
}

void IndexLowering::handle(const TernaryOp* top) {
  const auto in1 = lowerSrcIndex(top->in1(), top->out());
  const auto in2 = lowerSrcIndex(top->in2(), top->out());
  const auto in3 = lowerSrcIndex(top->in3(), top->out());
  const auto out = lowerDstIndex(top->out());
  pushBack(IrBuilder::create<TernaryOp>(
      top->getTernaryOpType(), out, in1, in2, in3));
}

namespace {

// Get the size of the temporary work buffer for grid communication, this can be
// grid reduction, broadcast, or grid welford.
// expansion_factor can be optionally passed to expand the allocation
// size. For example, FusedReduction should double the work buffer size.
Val* getGridCommWorkBufferSize(
    const TensorDomain* td,
    int expansion_factor = 1) {
  // The buffer size is the number of thread blocks multiplied by the
  // number of threads not used for reduction domains.
  // Note: Previously it was calculated based on the shape of the
  // tensor, but it makes more sense to compute the size based on the
  // shape of the thread block and grid since this buffer is used for
  // communications among them. Both methods should result in the same
  // size if the parallel dimensions are exact, but otherwise, just
  // computing the buffer size based on the tensor shape isn't
  // sufficient since there could be extra threads/blocks.
  TORCH_INTERNAL_ASSERT(
      expansion_factor >= 1, "Invalid expansion factor: ", expansion_factor);
  Val* buffer_size = expansion_factor == 1
      ? GpuLower::current()->kernel()->oneVal()
      : IrBuilder::create<Int>(expansion_factor);
  for (auto pt : kParallelTypeThreads) {
    auto pt_dim = GpuLower::current()->parallelDimensionMap().get(pt);
    if (pt_dim == nullptr || pt_dim->isOneInt()) {
      continue;
    }
    if (isParallelTypeThreadDim(pt) &&
        std::any_of(td->domain().begin(), td->domain().end(), [&](auto out_id) {
          return out_id->getParallelType() == pt &&
              (out_id->isReduction() || out_id->isBroadcast());
        })) {
      continue;
    }
    buffer_size = IrBuilder::mulExpr(buffer_size, pt_dim);
  }
  return buffer_size;
}

Val* getGridSyncBufferSize(const TensorDomain* td) {
  // See the comment above for getGridCommWorkBufferSize.
  Val* buffer_size = GpuLower::current()->kernel()->oneVal();
  for (auto pt : kParallelTypeBIDs) {
    auto pt_dim = GpuLower::current()->parallelDimensionMap().get(pt);
    if (pt_dim == nullptr || pt_dim->isOneInt()) {
      continue;
    }
    if (std::any_of(td->domain().begin(), td->domain().end(), [&](auto out_id) {
          return out_id->getParallelType() == pt &&
              (out_id->isReduction() || out_id->isBroadcast());
        })) {
      continue;
    }
    buffer_size = IrBuilder::mulExpr(buffer_size, pt_dim);
  }
  return buffer_size;
}

// Allocate global buffer for a grid communication calls, i.e. grid reduce, grid
// welford reduce, grid broadcast.
kir::Allocate* allocGlobalBufferForGridComm(
    Val* buffer_size,
    DataType dtype,
    bool zero_init) {
  const std::vector<IterDomain*> new_buffer_ids = {
      IrBuilder::create<IterDomain>(
          GpuLower::current()->kernel()->zeroVal(), buffer_size)};
  const auto buffer_domain = IrBuilder::create<TensorDomain>(new_buffer_ids);
  const auto buffer_tv =
      IrBuilder::create<TensorView>(buffer_domain, dtype, MemoryType::Global);
  return IrBuilder::create<kir::Allocate>(
      buffer_tv, buffer_tv->getMemoryType(), nullptr, zero_init);
}

} // namespace

void IndexLowering::handle(const ReductionOp* rop) {
  TORCH_INTERNAL_ASSERT(ir_utils::isTvOp(rop));

  const auto out_tv = rop->out()->as<TensorView>();
  const auto out_domain = out_tv->domain();

  const bool has_block_reduce = out_domain->hasBlockReduction();
  const bool has_grid_reduce = out_domain->hasGridReduction();

  const auto out = lowerDstIndex(rop->out());
  const auto in = lowerSrcIndex(rop->in(), rop->out());

  // Serial reduction
  if (!has_block_reduce && !has_grid_reduce) {
    pushBack(
        IrBuilder::create<BinaryOp>(rop->getReductionOpType(), out, out, in));
    return;
  }

  ReductionOp* indexed_rop = IrBuilder::create<ReductionOp>(
      rop->getReductionOpType(), rop->init(), out, in, rop->isFused());
  if (rop->predicate()) {
    indexed_rop->setPredicate(rop->predicate());
  }
  if (rop->writePredicate()) {
    indexed_rop->setWritePredicate(rop->writePredicate());
  }

  // If not grid reduction, just append the new ReductionOp node
  if (!has_grid_reduce) {
    pushBack(indexed_rop);
    return;
  }

  handleGridReduction(indexed_rop);
}

void IndexLowering::handleGridReduction(ReductionOp* indexed_rop) {
  const auto out_tv = indexed_rop->out()->as<kir::TensorIndex>()->view();
  const auto out_domain = out_tv->domain();

  TORCH_INTERNAL_ASSERT(out_domain->hasGridReduction());

  // If we do a grid reduction we can't have a reduction axis that is not bound
  // to a grid or block dim.
  TORCH_INTERNAL_ASSERT(
      std::none_of(
          out_domain->domain().begin(),
          out_domain->domain().end(),
          [](IterDomain* id) {
            return !id->isThread() && id->isReduction() &&
                !id->extent()->isOneInt();
          }),
      "Found a reduction stage that has both a non-parallelized ",
      "reduction and a grid reduction. This is not supported, ",
      "please use rfactor to do the serialized reduction first, ",
      "then the grid reduction.");

  // When using the fused reduction in a loop, the global work buffer
  // is double buffered to save global synchronizations.
  auto is_within_a_loop = std::any_of(
      out_domain->domain().begin(),
      out_domain->domain().end(),
      [](IterDomain* id) { return !isTrivialIterDomain(id); });

  const auto reduce_buffer = allocGlobalBufferForGridComm(
      getGridCommWorkBufferSize(
          out_domain, indexed_rop->isFused() && is_within_a_loop ? 2 : 1),
      indexed_rop->out()->dtype(),
      false);

  const auto sync_buffer = allocGlobalBufferForGridComm(
      getGridSyncBufferSize(out_domain), DataType::Int, true);

  const bool block_reduce_separated =
      out_domain->hasBlockReduction() && !indexed_rop->isFused();

  // The thread predicate for GridReduction needs to be set
  // separately from the main predicate. Do not combine them like
  // other expressions.
  const auto& thread_pred =
      GpuLower::current()->threadPredMap().getPredicatedParallelTypes(out_tv);

  auto grid_reduction = IrBuilder::create<kir::GridReduction>(
      indexed_rop, reduce_buffer, sync_buffer);

  grid_reduction->setThreadPredicate(thread_pred);

  // If preceded by a blockReduce, all thread blocks should have
  // valid inputs to gridReduce. In fact, using the original
  // predicate does not work when the write predicate of the
  // blockReduce is different from the read predicate.
  if (indexed_rop->predicate()) {
    if (block_reduce_separated) {
      grid_reduction->setPredicate(IrBuilder::create<kir::Predicate>(
          GpuLower::current()->kernel()->trueVal()));
    } else {
      grid_reduction->setPredicate(indexed_rop->predicate());
    }
  }

  if (indexed_rop->writePredicate()) {
    grid_reduction->setWritePredicate(indexed_rop->writePredicate());
  }

  // Push back the reduction op when block reduction is done
  // separately. Otherwise, the reduction op is just referenced from
  // the grid reduction op.
  if (block_reduce_separated) {
    pushBack(indexed_rop);
  }

  pushBack(reduce_buffer);
  pushBack(sync_buffer);
  pushBack(grid_reduction);

  if (indexed_rop->isFused()) {
    // When using the fused reduction, allocate the reduction object at
    // the outer-most scope
    auto fused_reduction_alloc_reduction =
        IrBuilder::create<kir::AllocateFusedReduction>(grid_reduction);
    insertAtTopLevel(fused_reduction_alloc_reduction);
  }
}

void IndexLowering::handle(const WelfordOp* wop) {
  TORCH_INTERNAL_ASSERT(ir_utils::isTvOp(wop));

  const auto out_tv = wop->outAvg()->as<TensorView>();
  const auto out_domain = out_tv->domain();

  const bool has_block_reduce = out_domain->hasBlockReduction();
  const bool has_grid_reduce = out_domain->hasGridReduction();

  // If we do a grid reduction we can't have a reduction axis that is not bound
  // to a grid or block dim ()
  if (has_grid_reduce) {
    TORCH_INTERNAL_ASSERT(
        std::none_of(
            out_domain->domain().begin(),
            out_domain->domain().end(),
            [](IterDomain* id) {
              return !id->isThread() && id->isReduction();
            }),
        "Found a reduction stage that has both a non-parallelized ",
        "reduction and a grid reduction.  This is not supported, ",
        "please use rfactor to do the serialized reduction first, ",
        "then the grid reduction.");
  }

  // lower IO tensors
  const auto in_var =
      wop->inVar() ? lowerSrcIndex(wop->inVar(), wop->outAvg()) : nullptr;
  const auto in_avg = lowerSrcIndex(wop->inAvg(), wop->outAvg());
  auto in_N = wop->inN();

  // in Rfactor-ed case, the input N is actually a TV
  if (!in_N->isScalar()) {
    in_N = lowerSrcIndex(in_N, wop->outN());
  }

  auto out_avg = lowerDstIndex(wop->outAvg());
  auto out_var = lowerDstIndex(wop->outVar());
  auto out_N = lowerDstIndex(wop->outN());

  WelfordOp* indexed_wop = IrBuilder::create<WelfordOp>(
      out_avg,
      out_var,
      out_N,
      wop->initAvg(),
      wop->initVar(),
      wop->initN(),
      in_avg,
      in_var,
      in_N,
      wop->isFused());

  if (wop->predicate()) {
    indexed_wop->setPredicate(wop->predicate());
  }
  if (wop->writePredicate()) {
    indexed_wop->setWritePredicate(wop->writePredicate());
  }

  // Serial welford
  if (!has_block_reduce && !has_grid_reduce) {
    pushBack(indexed_wop);
    return;
  }

  // Block-only welford
  if (!has_grid_reduce) {
    pushBack(indexed_wop);
    return;
  }

  handleGridWelford(indexed_wop);
}

void IndexLowering::handleGridWelford(WelfordOp* indexed_wop) {
  const auto out_tv = indexed_wop->out()->as<kir::TensorIndex>()->view();
  const auto out_domain = out_tv->domain();

  // Buffer allocation
  // When using the fused reduction in a loop, the global work buffer
  // is double buffered to save global synchronizations.
  auto is_within_a_loop = std::any_of(
      out_domain->domain().begin(),
      out_domain->domain().end(),
      [](IterDomain* id) { return !isTrivialIterDomain(id); });

  const auto work_buffer_size = getGridCommWorkBufferSize(
      out_domain, indexed_wop->isFused() && is_within_a_loop ? 2 : 1);

  const auto out_var_buffer = allocGlobalBufferForGridComm(
      work_buffer_size, indexed_wop->outVar()->dtype(), false);
  const auto out_avg_buffer = allocGlobalBufferForGridComm(
      work_buffer_size, indexed_wop->outAvg()->dtype(), false);
  const auto out_N_buffer = allocGlobalBufferForGridComm(
      work_buffer_size, indexed_wop->outN()->dtype(), false);

  const auto sync_buffer = allocGlobalBufferForGridComm(
      getGridSyncBufferSize(out_domain), DataType::Int, true);

  // The thread predicate for GridReduction needs to be set
  // separately from the main predicate. Do not combine them like
  // other expressions.
  const auto& thread_pred =
      GpuLower::current()->threadPredMap().getPredicatedParallelTypes(out_tv);

  auto grid_welford = IrBuilder::create<kir::GridWelford>(
      indexed_wop, out_var_buffer, out_avg_buffer, out_N_buffer, sync_buffer);

  grid_welford->setThreadPredicate(thread_pred);

  const bool block_reduce_separated =
      out_domain->hasBlockReduction() && !indexed_wop->isFused();

  if (indexed_wop->predicate()) {
    if (block_reduce_separated) {
      grid_welford->setPredicate(IrBuilder::create<kir::Predicate>(
          GpuLower::current()->kernel()->trueVal()));
    } else {
      grid_welford->setPredicate(indexed_wop->predicate());
    }
  }

  if (indexed_wop->writePredicate()) {
    grid_welford->setWritePredicate(indexed_wop->writePredicate());
  }

  if (block_reduce_separated) {
    pushBack(indexed_wop);
  }

  pushBack(out_var_buffer);
  pushBack(out_avg_buffer);
  pushBack(out_N_buffer);
  pushBack(sync_buffer);
  pushBack(grid_welford);

  if (indexed_wop->isFused()) {
    // When using the fused reduction, allocate the reduction object at
    // the outer-most scope
    auto fused_reduction_alloc_reduction =
        IrBuilder::create<kir::AllocateFusedReduction>(grid_welford);
    insertAtTopLevel(fused_reduction_alloc_reduction);
  }
}

void IndexLowering::handle(const MmaOp* mma) {
  const auto a = lowerSrcIndex(mma->inA(), mma->out());
  const auto b = lowerSrcIndex(mma->inB(), mma->out());
  const auto out = lowerDstIndex(mma->out());
  auto mma_indexed =
      IrBuilder::create<MmaOp>(out, a, b, mma->init(), mma->options());
  pushBack(mma_indexed);
}

void IndexLowering::handle(const BroadcastOp* bop) {
  TORCH_INTERNAL_ASSERT(ir_utils::isTvOp(bop));

  const auto out_tv = bop->out()->as<TensorView>();

  const auto out = lowerDstIndex(bop->out());
  const auto in = lowerSrcIndex(bop->in(), bop->out());
  auto indexed_expr =
      IrBuilder::create<BroadcastOp>(out, in, bop->getBroadcastDimFlags());

  const ParallelTypeBitmap parallel_bitmap =
      GpuLower::current()->threadPredMap().getParallelBroadcastDomains(out_tv);

  const bool block_x = parallel_bitmap.get(ParallelType::BIDx);
  const bool block_y = parallel_bitmap.get(ParallelType::BIDy);
  const bool block_z = parallel_bitmap.get(ParallelType::BIDz);

  if (bop->predicate()) {
    indexed_expr->setPredicate(bop->predicate());
  }

  const bool grid_broadcast_needed = block_x || block_y || block_z;
  if (!grid_broadcast_needed) {
    pushBack(indexed_expr);
    return;
  }

  // Grid broadcast
  const auto out_domain = out_tv->domain();
  const auto broadcast_buffer = allocGlobalBufferForGridComm(
      getGridCommWorkBufferSize(out_domain), out->dtype(), false);

  const auto sync_buffer = allocGlobalBufferForGridComm(
      getGridSyncBufferSize(out_domain), DataType::Int, true);

  auto grid_broadcast = IrBuilder::create<kir::GridBroadcast>(
      indexed_expr, broadcast_buffer, sync_buffer);

  if (bop->predicate()) {
    grid_broadcast->setPredicate(bop->predicate());
  }

  pushBack(broadcast_buffer);
  pushBack(sync_buffer);
  pushBack(grid_broadcast);
}

void IndexLowering::handle(const kir::Allocate* allocate) {
  // TODO(kir): remove the need for const_cast
  pushBack(const_cast<kir::Allocate*>(allocate)); // NOLINT
}

void IndexLowering::handle(const kir::BlockSync* sync) {
  // TODO(kir): remove the need for const_cast
  pushBack(const_cast<kir::BlockSync*>(sync)); // NOLINT
}

void IndexLowering::handle(const kir::GridSync* sync) {
  // TODO(kir): remove the need for const_cast
  pushBack(const_cast<kir::GridSync*>(sync)); // NOLINT
}

void IndexLowering::generate(const std::vector<Expr*>& exprs) {
  for (auto expr : exprs) {
    OptOutConstDispatch::handle(expr);
  }
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
