#include <torch/csrc/jit/codegen/cuda/arith.h>
#include <torch/csrc/jit/codegen/cuda/index_compute.h>
#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir_builder.h>
#include <torch/csrc/jit/codegen/cuda/lower2device.h>
#include <torch/csrc/jit/codegen/cuda/lower_utils.h>
#include <torch/csrc/jit/codegen/cuda/predicate_compute.h>

#include <torch/csrc/jit/codegen/cuda/lower_index.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

IndexLowering::IndexLowering() : ir_builder_(GpuLower::current()->kernel()) {}

kir::Val* IndexLowering::lowerSrcIndex(kir::Val* src, kir::Val* dst) const {
  if (auto tv = dynamic_cast<kir::TensorView*>(src)) {
    TORCH_INTERNAL_ASSERT(dst->isA<kir::TensorView>());
    return Index::getProducerIndex(
        tv->fuserTv(),
        dst->as<kir::TensorView>()->fuserTv(),
        scope_utils::getLoops(active_scope_expr_));
  } else {
    return src;
  }
}

kir::Val* IndexLowering::lowerDstIndex(kir::Val* dst) const {
  if (auto tv = dynamic_cast<kir::TensorView*>(dst)) {
    return Index::getConsumerIndex(
        tv->fuserTv(), scope_utils::getLoops(active_scope_expr_));
  } else {
    return dst;
  }
}

void IndexLowering::pushBack(kir::Expr* expr) {
  if (active_scope_ == nullptr) {
    lowered_exprs_.push_back(expr);
  } else {
    active_scope_->push_back(expr);
  }
}

void IndexLowering::visit(const kir::IfThenElse* ite) {
  const auto prev_scope_expr = active_scope_expr_;
  const auto prev_scope = active_scope_;

  // TODO(kir): try to avoid recreating new nodes and leaving old ones around
  auto new_ite = ir_builder_.create<kir::IfThenElse>(ite->predicate());
  pushBack(new_ite);

  active_scope_expr_ = new_ite;
  active_scope_ = &new_ite->thenBody();

  for (auto expr : ite->thenBody().exprs()) {
    expr->accept(this);
  }

  active_scope_ = &new_ite->elseBody();

  for (auto expr : ite->elseBody().exprs()) {
    expr->accept(this);
  }

  active_scope_ = prev_scope;
  active_scope_expr_ = prev_scope_expr;
}

void IndexLowering::visit(const kir::ForLoop* for_loop) {
  const auto prev_scope_expr = active_scope_expr_;
  const auto prev_scope = active_scope_;

  auto new_for_loop = ir_builder_.create<kir::ForLoop>(for_loop);
  pushBack(new_for_loop);

  active_scope_expr_ = new_for_loop;
  active_scope_ = &new_for_loop->body();

  for (auto expr : for_loop->body().exprs()) {
    expr->accept(this);
  }

  active_scope_ = prev_scope;
  active_scope_expr_ = prev_scope_expr;
}

void IndexLowering::visit(const kir::UnaryOp* uop) {
  const auto in = lowerSrcIndex(uop->in(), uop->out());
  const auto out = lowerDstIndex(uop->out());
  pushBack(ir_builder_.create<kir::UnaryOp>(uop->operation(), out, in));
}

void IndexLowering::visit(const kir::BinaryOp* bop) {
  const auto lhs = lowerSrcIndex(bop->lhs(), bop->out());
  const auto rhs = lowerSrcIndex(bop->rhs(), bop->out());
  const auto out = lowerDstIndex(bop->out());
  pushBack(ir_builder_.create<kir::BinaryOp>(bop->operation(), out, lhs, rhs));
}

void IndexLowering::visit(const kir::TernaryOp* top) {
  const auto in1 = lowerSrcIndex(top->in1(), top->out());
  const auto in2 = lowerSrcIndex(top->in2(), top->out());
  const auto in3 = lowerSrcIndex(top->in3(), top->out());
  const auto out = lowerDstIndex(top->out());
  pushBack(
      ir_builder_.create<kir::TernaryOp>(top->operation(), out, in1, in2, in3));
}

namespace {

void allocateGridReductionFlag(
    kir::TensorView* out_tv,
    kir::Expr* current_scope_expr) {
  kir::IrBuilder ir_builder(GpuLower::current()->kernel());

  const auto flag_name = kir::GridReduction::getPredicateFlagName(out_tv);
  const auto flag_var = ir_builder.create<kir::Allocate>(
      ir_builder.create<kir::NamedScalar>(flag_name, DataType::Bool),
      MemoryType::Local,
      ir_builder.create<kir::Int>(1));

  // When enclosed by IfThenElse, place the variable outside of the
  // IfThenElse. This IfThenElse is assumed to be the prediate for
  // this grid reduction expression.
  if (current_scope_expr->isA<kir::IfThenElse>()) {
    scope_utils::insertBefore(
        current_scope_expr->parentScope(), current_scope_expr, flag_var);
  } else {
    TORCH_INTERNAL_ASSERT(current_scope_expr->isA<kir::ForLoop>());
    current_scope_expr->as<kir::ForLoop>()->body().push_back(flag_var);
  }
}

// Get the size of the temporary work buffer for a grid
// reduction/welford.
kir::Val* getGridReductionWorkBufferSize(
    kir::IrBuilder& ir_builder,
    const kir::TensorDomain* td) {
  // The buffer size is the number of thread blocks multiplied by the
  // number of threads not used for reduction domains.
  // Note: Previously it was calculated based on the shape of the
  // tensor, but it makes more sense to compute the size based on the
  // shape of the thread block and grid since this buffer is used for
  // communications among them. Both methods should result in the same
  // size if the parallel dimensions are exact, but otherwise, just
  // computing the buffer size based on the tensor shape isn't
  // sufficient since there could be extra threads/blocks.
  kir::Val* buffer_size = ir_builder.create<kir::Int>(1);
  for (auto pt : kParallelTypeThreads) {
    auto pt_dim = GpuLower::current()->parallelDimensionMap().get(pt);
    if (pt_dim == nullptr || pt_dim->isOneInt()) {
      continue;
    }
    if (isParallelTypeThreadDim(pt) &&
        std::any_of(td->domain().begin(), td->domain().end(), [&](auto out_id) {
          return out_id->parallelType() == pt && out_id->isReduction();
        })) {
      continue;
    }
    buffer_size = ir_builder.mulExpr(buffer_size, pt_dim);
  }
  return buffer_size;
}

kir::Val* getGridReductionSyncBufferSize(
    kir::IrBuilder& ir_builder,
    const kir::TensorDomain* td) {
  // See the comment above for getGridReductionWorkBufferSize.
  kir::Val* buffer_size = ir_builder.create<kir::Int>(1);
  for (auto pt : kParallelTypeBIDs) {
    auto pt_dim = GpuLower::current()->parallelDimensionMap().get(pt);
    if (pt_dim == nullptr || pt_dim->isOneInt()) {
      continue;
    }
    if (std::any_of(td->domain().begin(), td->domain().end(), [&](auto out_id) {
          return out_id->parallelType() == pt && out_id->isReduction();
        })) {
      continue;
    }
    buffer_size = ir_builder.mulExpr(buffer_size, pt_dim);
  }
  return buffer_size;
}

// Allocate a buffer for a grid reductin or welford.
kir::Allocate* allocGlobalBufferForGridReduction(
    kir::IrBuilder& ir_builder,
    kir::Val* buffer_size,
    DataType dtype,
    bool zero_init) {
  const std::vector<kir::IterDomain*> new_buffer_ids = {
      ir_builder.create<kir::IterDomain>(ir_builder.zeroVal(), buffer_size)};
  const auto buffer_domain =
      ir_builder.create<kir::TensorDomain>(new_buffer_ids);
  const auto buffer_tv = ir_builder.create<kir::TensorView>(
      dtype, buffer_domain, MemoryType::Global);
  return ir_builder.create<kir::Allocate>(
      buffer_tv, buffer_tv->memoryType(), nullptr, zero_init);
}

} // namespace

void IndexLowering::visit(const kir::ReductionOp* rop) {
  TORCH_INTERNAL_ASSERT(ir_utils::isTVOp(rop));

  const auto out_tv = rop->out()->as<kir::TensorView>();
  const auto out_domain = out_tv->domain();

  const bool is_block_reduce = out_domain->hasBlockReduction();
  const bool is_grid_reduce = out_domain->hasGridReduction();

  // If we do a grid reduction we can't have a reduction axis that is not bound
  // to a grid or block dim ()
  if (is_grid_reduce) {
    TORCH_INTERNAL_ASSERT(
        std::none_of(
            out_domain->domain().begin(),
            out_domain->domain().end(),
            [](kir::IterDomain* id) {
              return !id->isThread() && id->isReduction() &&
                  !id->extent()->isOneInt();
            }),
        "Found a reduction stage that has both a non-parallelized ",
        "reduction and a grid reduction.  This is not supported, ",
        "please use rfactor to do the serialized reduction first, ",
        "then the grid reduction.");
  }

  const auto out = lowerDstIndex(rop->out());
  const auto in = lowerSrcIndex(rop->in(), rop->out());

  kir::ReductionOp* block_reduction_op = nullptr;

  if (is_block_reduce) {
    block_reduction_op = ir_builder_.create<kir::ReductionOp>(
        rop->operation(), rop->init(), out, in);
    if (rop->predicate()) {
      block_reduction_op->setPredicate(rop->predicate());
    }
    if (rop->writePredicate()) {
      block_reduction_op->setWritePredicate(rop->writePredicate());
    }
    pushBack(block_reduction_op);
  }

  if (is_grid_reduce) {
    // First, declare a boolean flag variable storing the return value
    // of the gridReduce() helper
    allocateGridReductionFlag(out_tv, active_scope_expr_);

    const auto reduce_buffer = allocGlobalBufferForGridReduction(
        ir_builder_,
        getGridReductionWorkBufferSize(ir_builder_, out_domain),
        out->dtype(),
        false);

    const auto sync_buffer = allocGlobalBufferForGridReduction(
        ir_builder_,
        getGridReductionSyncBufferSize(ir_builder_, out_domain),
        DataType::Int,
        true);

    const auto grid_reduction_op = (block_reduction_op == nullptr)
        ? ir_builder_.create<kir::ReductionOp>(
              rop->operation(), rop->init(), out, in)
        : block_reduction_op;

    // The thread predicate for GridReduction needs to be set
    // separately from the main predicate. Do not combine them like
    // other expressions.
    const auto& thread_pred =
        GpuLower::current()->threadPredMap().getPredicatedParallelTypes(
            out_tv->fuserTv());
    auto grid_reduction = ir_builder_.create<kir::GridReduction>(
        grid_reduction_op, reduce_buffer, sync_buffer);
    grid_reduction->setThreadPredicate(thread_pred);

    if (rop->predicate()) {
      // If preceded by a blockReduce, all thread blocks should have
      // valid inputs to gridReduce. In fact, using the original
      // predicate does not work when the write predicate of the
      // blockReduce is different from the read predicate.
      if (is_block_reduce) {
        grid_reduction->setPredicate(
            ir_builder_.create<kir::Predicate>(ir_builder_.trueVal()));
      } else {
        grid_reduction->setPredicate(rop->predicate());
      }
    }

    if (rop->writePredicate()) {
      grid_reduction->setWritePredicate(rop->writePredicate());
    }

    pushBack(reduce_buffer);
    pushBack(sync_buffer);
    pushBack(grid_reduction);
  }

  if (!is_block_reduce && !is_grid_reduce) {
    // TODO(kir): this breaks our "SSA" form
    pushBack(ir_builder_.create<kir::BinaryOp>(rop->operation(), out, out, in));
  }
}

void IndexLowering::visit(const kir::WelfordOp* wop) {
  TORCH_INTERNAL_ASSERT(ir_utils::isTVOp(wop));

  const auto out_tv = wop->outAvg()->as<kir::TensorView>();
  const auto out_domain = out_tv->domain();

  const bool is_block_reduce = out_domain->hasBlockReduction();
  const bool is_grid_reduce = out_domain->hasGridReduction();

  // If we do a grid reduction we can't have a reduction axis that is not bound
  // to a grid or block dim ()
  if (is_grid_reduce) {
    TORCH_INTERNAL_ASSERT(
        std::none_of(
            out_domain->domain().begin(),
            out_domain->domain().end(),
            [](kir::IterDomain* id) {
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

  kir::WelfordOp* welford_op = ir_builder_.create<kir::WelfordOp>(
      out_var,
      out_avg,
      out_N,
      wop->initVar(),
      wop->initAvg(),
      wop->initN(),
      in_var,
      in_avg,
      in_N);

  kir::WelfordOp* block_welford_op = nullptr;

  if (is_block_reduce) {
    block_welford_op = welford_op;
    if (wop->predicate()) {
      block_welford_op->setPredicate(wop->predicate());
    }
    if (wop->writePredicate()) {
      block_welford_op->setWritePredicate(wop->writePredicate());
    }
    pushBack(block_welford_op);
  }

  if (is_grid_reduce) {
    // Allocate T_pred
    allocateGridReductionFlag(out_tv, active_scope_expr_);

    // Buffer allocation
    const auto work_buffer_size =
        getGridReductionWorkBufferSize(ir_builder_, out_domain);

    const auto out_var_buffer = allocGlobalBufferForGridReduction(
        ir_builder_, work_buffer_size, out_var->dtype(), false);
    const auto out_avg_buffer = allocGlobalBufferForGridReduction(
        ir_builder_, work_buffer_size, out_avg->dtype(), false);
    const auto out_N_buffer = allocGlobalBufferForGridReduction(
        ir_builder_, work_buffer_size, out_N->dtype(), false);

    const auto sync_buffer = allocGlobalBufferForGridReduction(
        ir_builder_,
        getGridReductionSyncBufferSize(ir_builder_, out_domain),
        DataType::Int,
        true);

    // Grid Welford instantiation
    const auto grid_welford_op =
        (block_welford_op == nullptr) ? welford_op : block_welford_op;

    // The thread predicate for GridReduction needs to be set
    // separately from the main predicate. Do not combine them like
    // other expressions.
    const auto& thread_pred =
        GpuLower::current()->threadPredMap().getPredicatedParallelTypes(
            out_tv->fuserTv());

    auto grid_welford = ir_builder_.create<kir::GridWelford>(
        grid_welford_op,
        out_var_buffer,
        out_avg_buffer,
        out_N_buffer,
        sync_buffer);

    grid_welford->setThreadPredicate(thread_pred);

    if (wop->predicate()) {
      grid_welford->setPredicate(wop->predicate());
    }

    pushBack(out_var_buffer);
    pushBack(out_avg_buffer);
    pushBack(out_N_buffer);
    pushBack(sync_buffer);
    pushBack(grid_welford);
  }

  if (!is_block_reduce && !is_grid_reduce) {
    pushBack(welford_op);
  }
}

void IndexLowering::visit(const kir::BroadcastOp* bop) {
  TORCH_INTERNAL_ASSERT(ir_utils::isTVOp(bop));

  const auto out = lowerDstIndex(bop->out());
  const auto in = lowerSrcIndex(bop->in(), bop->out());
  auto indexed_expr = ir_builder_.create<kir::BroadcastOp>(out, in);

  if (bop->predicate()) {
    indexed_expr->setPredicate(bop->predicate());
  }

  pushBack(indexed_expr);
}

void IndexLowering::visit(const kir::Allocate* allocate) {
  // TODO(kir): remove the need for const_cast
  pushBack(const_cast<kir::Allocate*>(allocate)); // NOLINT
}

void IndexLowering::visit(const kir::Sync* sync) {
  // TODO(kir): remove the need for const_cast
  pushBack(const_cast<kir::Sync*>(sync)); // NOLINT
}

void IndexLowering::generate(const std::vector<kir::Expr*>& exprs) {
  for (auto expr : exprs) {
    expr->accept(this);
  }
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
