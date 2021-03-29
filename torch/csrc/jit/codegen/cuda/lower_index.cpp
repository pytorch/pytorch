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

Val* IndexLowering::lowerOperand(Val* op, Val* out) const {
  if (ir_utils::isTV(op)) {
    return Index::getProducerIndex(
        ir_utils::asTV(op),
        ir_utils::asTV(out),
        scope_utils::getLoops(active_scope_expr));
  } else {
    return GpuLower::lowerValue(op);
  }
}

Val* IndexLowering::lowerOutput(Expr* expr) const {
  TORCH_CHECK(expr->outputs().size() == 1);
  const auto out = expr->output(0);
  if (ir_utils::isTVOp(expr)) {
    return Index::getConsumerIndex(
        ir_utils::asTV(out), scope_utils::getLoops(active_scope_expr));
  } else {
    return GpuLower::lowerValue(out);
  }
}

void IndexLowering::pushBack(Expr* expr) {
  if (active_scope == nullptr) {
    lowered_exprs.push_back(expr);
  } else {
    active_scope->push_back(expr);
  }
}

void IndexLowering::handle(kir::IfThenElse* ite) {
  Expr* prev_scope_expr = active_scope_expr;
  kir::Scope* prev_scope = active_scope;

  auto new_ite =
      ir_builder_.create<kir::IfThenElse>(ite->cond(), prev_scope_expr);
  pushBack(new_ite);
  active_scope_expr = new_ite;
  active_scope = &new_ite->thenBody();

  for (auto expr : ite->thenBody().exprs()) {
    OptInDispatch::handle(expr);
  }

  active_scope = &new_ite->elseBody();

  for (auto expr : ite->elseBody().exprs()) {
    OptInDispatch::handle(expr);
  }

  active_scope = prev_scope;
  active_scope_expr = prev_scope_expr;
}

void IndexLowering::handle(kir::ForLoop* fl) {
  Expr* prev_scope_expr = active_scope_expr;
  kir::Scope* prev_scope = active_scope;

  auto newFl = ir_builder_.create<kir::ForLoop>(
      fl->index(), fl->iter_domain(), prev_scope_expr);
  pushBack(newFl);

  active_scope_expr = newFl;
  active_scope = &newFl->body();

  for (auto expr : fl->body().exprs()) {
    OptInDispatch::handle(expr);
  }

  active_scope = prev_scope;
  active_scope_expr = prev_scope_expr;
}

void IndexLowering::handle(UnaryOp* uop) {
  if (ir_utils::isTVOp(uop)) {
    const auto in = lowerOperand(uop->in(), uop->out());
    const auto out = lowerOutput(uop);
    pushBack(ir_builder_.create<kir::UnaryOp>(uop->getUnaryOpType(), out, in));
  } else {
    // This will automatically lower the expression defining the value
    pushBack(GpuLower::lowerValue(uop->out())->getOrigin());
  }
}

void IndexLowering::handle(BinaryOp* bop) {
  if (ir_utils::isTVOp(bop)) {
    const auto lhs = lowerOperand(bop->lhs(), bop->out());
    const auto rhs = lowerOperand(bop->rhs(), bop->out());
    const auto out = lowerOutput(bop);
    pushBack(ir_builder_.create<kir::BinaryOp>(
        bop->getBinaryOpType(), out, lhs, rhs));
  } else {
    // This will automatically lower the expression defining the value
    pushBack(GpuLower::lowerValue(bop->out())->getOrigin());
  }
}

void IndexLowering::handle(TernaryOp* top) {
  if (ir_utils::isTVOp(top)) {
    const auto in1 = lowerOperand(top->in1(), top->out());
    const auto in2 = lowerOperand(top->in2(), top->out());
    const auto in3 = lowerOperand(top->in3(), top->out());
    const auto out = lowerOutput(top);
    pushBack(ir_builder_.create<kir::TernaryOp>(
        top->getTernaryOpType(), out, in1, in2, in3));
  } else {
    // This will automatically lower the expression defining the value
    pushBack(GpuLower::lowerValue(top->out())->getOrigin());
  }
}

namespace {

void allocateGridReductionFlag(TensorView* out_tv, Expr* current_scope_expr) {
  kir::IrBuilder ir_builder(GpuLower::current()->kernel());
  auto flag_name = kir::GridReduction::getPredicateFlagName(out_tv);
  auto flag_var = ir_builder.create<kir::Allocate>(
      ir_builder.create<kir::NamedScalar>(flag_name, DataType::Bool),
      MemoryType::Local,
      ir_builder.create<kir::Int>(1));
  // When enclosed by IfThenElse, place the variable outside of the
  // IfThenElse. This IfThenElse is assumed to be the prediate for
  // this grid reduction expression.
  if (current_scope_expr->getExprType() == ExprType::IfThenElse) {
    scope_utils::insertBefore(
        scope_utils::getParent(current_scope_expr),
        current_scope_expr,
        flag_var);
  } else {
    scope_utils::pushBack(current_scope_expr, flag_var);
  }
}

} // namespace

void IndexLowering::handle(ReductionOp* rop) {
  TORCH_INTERNAL_ASSERT(
      ir_utils::isTVOp(rop),
      "Cannot have a reduction operation on something other than a tensor view, but received ",
      rop);

  auto out_tv = ir_utils::asTV(rop->out());

  const bool is_block_reduce = out_tv->hasBlockReduction();
  const bool is_grid_reduce = out_tv->hasGridReduction();

  // If we do a grid reduction we can't have a reduction axis that is not bound
  // to a grid or block dim ()
  if (is_grid_reduce) {
    TORCH_INTERNAL_ASSERT(
        std::none_of(
            out_tv->domain()->domain().begin(),
            out_tv->domain()->domain().end(),
            [](IterDomain* id) {
              return !id->isThread() && id->isReduction();
            }),
        "Found a reduction stage that has both a non-parallelized reduction and a grid reduction.",
        " This is not supported, please use rfactor to do the serialized reduction first, then the grid reduction.");
  }
  const auto loops = scope_utils::getLoops(active_scope_expr);

  kir::TensorIndex* out = Index::getConsumerIndex(out_tv, loops);
  kir::TensorIndex* in = Index::getProducerIndex(
      ir_utils::asTV(rop->in()), ir_utils::asTV(rop->out()), loops);

  kir::ReductionOp* block_reduction_op = nullptr;
  if (is_block_reduce) {
    auto pred =
        PredicateCompute::getInlinePredicate(rop, loops, nullptr, false);

    block_reduction_op = ir_builder_.create<kir::ReductionOp>(
        rop->getReductionOpType(),
        GpuLower::lowerValue(rop->init()),
        out,
        in,
        pred);
    pushBack(block_reduction_op);
  }

  if (is_grid_reduce) {
    // First, declare a boolean flag variable storing the return value
    // of gridReduce.
    allocateGridReductionFlag(out_tv, active_scope_expr);

    std::vector<IterDomain*> buffer_ids(out_tv->domain()->domain());
    buffer_ids.erase(
        std::remove_if(
            buffer_ids.begin(),
            buffer_ids.end(),
            [](IterDomain* id) {
              return id->isReduction() & !id->isBlockDim();
            }),
        buffer_ids.end());

    Val* buffer_size =
        buffer_ids.empty() ? new Int(1) : buffer_ids[0]->rawExtent();
    for (size_t i = 1; i < buffer_ids.size(); i++) {
      buffer_size = mul(buffer_size, buffer_ids[i]->rawExtent());
    }

    std::vector<IterDomain*> sync_ids(out_tv->domain()->domain());
    sync_ids.erase(
        std::remove_if(
            sync_ids.begin(),
            sync_ids.end(),
            [](IterDomain* id) {
              return id->isReduction() || !id->isBlockDim();
            }),
        sync_ids.end());

    Val* sync_size = sync_ids.empty() ? new Int(1) : sync_ids[0]->rawExtent();
    for (size_t i = 1; i < sync_ids.size(); i++) {
      sync_size = mul(sync_size, sync_ids[i]->rawExtent());
    }

    IterDomain* buffer_id = new IterDomain(new Int(0), buffer_size);
    TensorView* reduce_buffer_tv = new TensorView(
        new TensorDomain({buffer_id}),
        out->getDataType().value(),
        MemoryType::Global);

    IterDomain* sync_id = new IterDomain(new Int(0), sync_size);
    TensorView* reduce_sync_tv = new TensorView(
        new TensorDomain({sync_id}), DataType::Int, MemoryType::Global);

    const auto reduce_buffer = ir_builder_.create<kir::Allocate>(
        GpuLower::lowerValue(reduce_buffer_tv),
        reduce_sync_tv->getMemoryType());
    const auto sync_buffer = ir_builder_.create<kir::Allocate>(
        GpuLower::lowerValue(reduce_sync_tv),
        reduce_sync_tv->getMemoryType(),
        nullptr,
        true);

    const auto grid_reduction_op = block_reduction_op == nullptr
        ? ir_builder_.create<kir::ReductionOp>(
              rop->getReductionOpType(),
              GpuLower::lowerValue(rop->init()),
              out,
              in)
        : block_reduction_op;
    auto pred =
        PredicateCompute::getInlinePredicate(rop, loops, nullptr, false);
    const auto grid_reduction = ir_builder_.create<kir::GridReduction>(
        grid_reduction_op, reduce_buffer, sync_buffer, pred);

    pushBack(reduce_buffer);
    pushBack(sync_buffer);
    pushBack(grid_reduction);
  }

  if (!is_block_reduce && !is_grid_reduce) {
    pushBack(ir_builder_.create<kir::BinaryOp>(
        rop->getReductionOpType(), out, out, in));
  }
}

void IndexLowering::handle(BroadcastOp* bop) {
  TORCH_INTERNAL_ASSERT(
      ir_utils::isTVOp(bop),
      "Cannot have a broadcast operation on something other than a tensor view, but received ",
      bop);

  auto loops = scope_utils::getLoops(active_scope_expr);

  kir::TensorIndex* out =
      Index::getConsumerIndex(ir_utils::asTV(bop->out()), loops);

  Val* in = bop->in();
  if (ir_utils::isTV(in))
    in = Index::getProducerIndex(
        ir_utils::asTV(in), ir_utils::asTV(bop->out()), loops);
  pushBack(ir_builder_.create<kir::BroadcastOp>(out, in));
}

void IndexLowering::handle(kir::Allocate* allocate) {
  pushBack(allocate);
}

void IndexLowering::handle(kir::Sync* sync) {
  pushBack(sync);
}

void IndexLowering::generate(const std::vector<Expr*>& exprs) {
  // Run through loop nests and further lower the expressions
  for (auto* expr : exprs) {
    OptInDispatch::handle(expr);
  }
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
