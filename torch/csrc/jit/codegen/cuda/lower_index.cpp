#include <torch/csrc/jit/codegen/cuda/arith.h>
#include <torch/csrc/jit/codegen/cuda/index_compute.h>
#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>
#include <torch/csrc/jit/codegen/cuda/lower_utils.h>

#include <torch/csrc/jit/codegen/cuda/lower_index.h>

namespace torch {
namespace jit {
namespace fuser {

void IndexLowering::pushBack(Expr* expr) {
  if (active_scope == nullptr)
    lowered_exprs.push_back(expr);
  else
    active_scope->push_back(expr);
}

void IndexLowering::handle(kir::IfThenElse* ite) {
  Expr* prev_scope_expr = active_scope_expr;
  kir::Scope* prev_scope = active_scope;

  auto new_ite = new kir::IfThenElse(ite->cond(), {}, {}, prev_scope_expr);
  pushBack(new_ite);
  active_scope_expr = new_ite;
  active_scope = &new_ite->body();

  for (auto expr : ite->body().exprs()) {
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

  auto newFl =
      new kir::ForLoop(fl->index(), fl->iter_domain(), {}, prev_scope_expr);
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
  if (!ir_utils::isTVOp(uop)) {
    pushBack(uop);
    return;
  }

  kir::TensorIndex* out = Index::getConsumerIndex(
      ir_utils::asTV(uop->out()), scope_utils::getLoops(active_scope_expr));
  Val* in = uop->in();
  if (ir_utils::isTV(in))
    in = Index::getProducerIndex(
        ir_utils::asTV(in),
        ir_utils::asTV(uop->out()),
        scope_utils::getLoops(active_scope_expr));
  pushBack(new UnaryOp(uop->getUnaryOpType(), out, in));
}

void IndexLowering::handle(BinaryOp* bop) {
  if (!ir_utils::isTVOp(bop)) {
    pushBack(bop);
    return;
  }

  kir::TensorIndex* out = Index::getConsumerIndex(
      ir_utils::asTV(bop->out()), scope_utils::getLoops(active_scope_expr));

  Val* lhs = bop->lhs();
  Val* rhs = bop->rhs();

  if (ir_utils::isTV(lhs))
    lhs = Index::getProducerIndex(
        ir_utils::asTV(lhs),
        ir_utils::asTV(bop->out()),
        scope_utils::getLoops(active_scope_expr));

  if (ir_utils::isTV(rhs))
    rhs = Index::getProducerIndex(
        ir_utils::asTV(rhs),
        ir_utils::asTV(bop->out()),
        scope_utils::getLoops(active_scope_expr));

  pushBack(new BinaryOp(bop->getBinaryOpType(), out, lhs, rhs));
}

void IndexLowering::handle(TernaryOp* top) {
  if (!ir_utils::isTVOp(top)) {
    pushBack(top);
    return;
  }

  kir::TensorIndex* out = Index::getConsumerIndex(
      ir_utils::asTV(top->out()), scope_utils::getLoops(active_scope_expr));
  Val* in1 = top->in1();
  Val* in2 = top->in2();
  Val* in3 = top->in3();

  if (ir_utils::isTV(in1))
    in1 = Index::getProducerIndex(
        ir_utils::asTV(in1),
        ir_utils::asTV(top->out()),
        scope_utils::getLoops(active_scope_expr));

  if (ir_utils::isTV(in2))
    in2 = Index::getProducerIndex(
        ir_utils::asTV(in2),
        ir_utils::asTV(top->out()),
        scope_utils::getLoops(active_scope_expr));

  if (ir_utils::isTV(in3))
    in3 = Index::getProducerIndex(
        ir_utils::asTV(in3),
        ir_utils::asTV(top->out()),
        scope_utils::getLoops(active_scope_expr));

  pushBack(new TernaryOp(top->getTernaryOpType(), out, in1, in2, in3));
}

void IndexLowering::handle(ReductionOp* rop) {
  TORCH_INTERNAL_ASSERT(
      ir_utils::isTVOp(rop),
      "Cannot have a reduction operation on something other than a tensor view, but received ",
      rop);

  auto out_tv = ir_utils::asTV(rop->out());

  bool is_block_reduce = out_tv->hasBlockReduction();

  bool is_grid_reduce = out_tv->hasGridReduction();

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
  auto loops = scope_utils::getLoops(active_scope_expr);

  kir::TensorIndex* out = Index::getConsumerIndex(out_tv, loops);
  Val* in = rop->in();
  in = Index::getProducerIndex(
      ir_utils::asTV(in), ir_utils::asTV(rop->out()), loops);

  ReductionOp* block_reduction = nullptr;
  if (is_block_reduce) {
    block_reduction =
        new ReductionOp(rop->getReductionOpType(), rop->init(), out, in);
    pushBack(block_reduction);
  }

  if (is_grid_reduce) {
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
        new TensorDomain({buffer_id}), out->getDataType().value());

    IterDomain* sync_id = new IterDomain(new Int(0), sync_size);
    TensorView* reduce_sync_tv =
        new TensorView(new TensorDomain({sync_id}), DataType::Int);

    auto reduce_buffer =
        new kir::Allocate(reduce_buffer_tv, MemoryType::Global);
    auto sync_buffer = new kir::Allocate(reduce_sync_tv, MemoryType::Global);

    pushBack(reduce_buffer);
    pushBack(sync_buffer);
    pushBack(new kir::GridReduction(
        block_reduction == nullptr
            ? new ReductionOp(rop->getReductionOpType(), rop->init(), out, in)
            : block_reduction,
        reduce_buffer,
        sync_buffer));
  }

  if (!is_block_reduce && !is_grid_reduce) {
    pushBack(new BinaryOp(rop->getReductionOpType(), out, out, in));
  }
}

void IndexLowering::handle(BroadcastOp* bop) {
  TORCH_INTERNAL_ASSERT(
      ir_utils::isTVOp(bop),
      "Cannot have a broadcast operation on something other than a tensor view, but received ",
      bop);

  kir::TensorIndex* out = Index::getConsumerIndex(
      ir_utils::asTV(bop->out()), scope_utils::getLoops(active_scope_expr));
  Val* in = bop->in();
  if (ir_utils::isTV(in))
    in = Index::getProducerIndex(
        ir_utils::asTV(in),
        ir_utils::asTV(bop->out()),
        scope_utils::getLoops(active_scope_expr));
  pushBack(new BroadcastOp(out, in));
}

void IndexLowering::generate(const std::vector<Expr*>& exprs) {
  // Run through loop nests and further lower the expressions
  for (auto* expr : exprs) {
    OptInDispatch::handle(expr);
  }
}

} // namespace fuser
} // namespace jit
} // namespace torch
