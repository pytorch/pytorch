#include <torch/csrc/jit/codegen/cuda/index_compute.h>
#include <torch/csrc/jit/codegen/cuda/lower_utils.h>

#include <torch/csrc/jit/codegen/cuda/lower_index.h>

namespace torch {
namespace jit {
namespace fuser {

void IndexLowering::pushBack(Expr* expr) {
  if (active_scope == nullptr)
    lowered_exprs.push_back(expr);
  else
    scope_utils::pushBack(active_scope, expr);
}

Statement* IndexLowering::mutate(Expr* expr) {
  Statement* mutated_stmt = OptOutMutator::mutate(expr);
  TORCH_INTERNAL_ASSERT(
      mutated_stmt->isExpr(),
      "Tried to generate a kernel but hit a non expression during lowering: ",
      mutated_stmt);
  return mutated_stmt;
}

Statement* IndexLowering::mutate(IfThenElse* ite) {
  Expr* prev_scope = active_scope;
  active_scope = ite;
  std::vector<Expr*> mutated_exprs;
  bool is_mutated = false;
  for (auto expr : ite->body().exprs()) {
    Statement* mutated_stmt = mutate(expr);
    Expr* mutated_expr = ir_utils::asExpr(mutated_stmt);
    mutated_exprs.push_back(mutated_expr);
    is_mutated = is_mutated | (mutated_expr != expr);
  }

  std::vector<Expr*> mutated_else_exprs;
  for (auto expr : ite->elseBody().exprs()) {
    Statement* mutated_stmt = mutate(expr);
    Expr* mutated_expr = ir_utils::asExpr(mutated_stmt);
    mutated_else_exprs.push_back(mutated_expr);
    is_mutated = is_mutated | (mutated_expr != expr);
  }

  if (is_mutated) {
    ite->body().clear();
    for (auto expr : mutated_exprs)
      ite->body().push_back(expr);
    ite->elseBody().clear();
    for (auto expr : mutated_else_exprs)
      ite->elseBody().push_back(expr);
  }

  active_scope = prev_scope;

  if (is_mutated) {
    auto new_ite = new IfThenElse(
        ite->cond(), mutated_exprs, mutated_else_exprs, ite->parentScope());
    return new_ite;
  }

  return ite;
}

Statement* IndexLowering::mutate(ForLoop* fl) {
  Expr* prev_scope = active_scope;
  active_scope = fl;
  std::vector<Expr*> mutated_exprs;
  bool is_mutated = false;
  for (auto expr : fl->body().exprs()) {
    Statement* mutated_stmt = mutate(expr);
    Expr* mutated_expr = ir_utils::asExpr(mutated_stmt);
    mutated_exprs.push_back(mutated_expr);
    is_mutated = is_mutated | (mutated_expr != expr);
  }

  active_scope = prev_scope;
  if (is_mutated) {
    auto newFL = new ForLoop(
        fl->index(), fl->iter_domain(), mutated_exprs, fl->parentScope());
    return newFL;
  }

  return fl;
}

Statement* IndexLowering::mutate(UnaryOp* uop) {
  if (!ir_utils::isTVOp(uop))
    return OptOutMutator::mutate(uop);

  TensorIndex* out = Index::getConsumerIndex(
      ir_utils::asTV(uop->out()), scope_utils::getLoops(active_scope));
  Val* in = uop->in();
  if (ir_utils::isTV(in))
    in = Index::getProducerIndex(
        ir_utils::asTV(in),
        ir_utils::asTV(uop->out()),
        scope_utils::getLoops(active_scope));
  Expr* new_op = new UnaryOp(uop->getUnaryOpType(), out, in);

  return new_op;
}

Statement* IndexLowering::mutate(BinaryOp* bop) {
  if (!ir_utils::isTVOp(bop))
    return OptOutMutator::mutate(bop);

  TensorIndex* out = Index::getConsumerIndex(
      ir_utils::asTV(bop->out()), scope_utils::getLoops(active_scope));

  Val* lhs = bop->lhs();
  Val* rhs = bop->rhs();

  if (ir_utils::isTV(lhs))
    lhs = Index::getProducerIndex(
        ir_utils::asTV(lhs),
        ir_utils::asTV(bop->out()),
        scope_utils::getLoops(active_scope));

  if (ir_utils::isTV(rhs))
    rhs = Index::getProducerIndex(
        ir_utils::asTV(rhs),
        ir_utils::asTV(bop->out()),
        scope_utils::getLoops(active_scope));

  Expr* new_op = new BinaryOp(bop->getBinaryOpType(), out, lhs, rhs);

  return new_op;
}

Statement* IndexLowering::mutate(TernaryOp* top) {
  if (!ir_utils::isTVOp(top))
    return OptOutMutator::mutate(top);

  TensorIndex* out = Index::getConsumerIndex(
      ir_utils::asTV(top->out()), scope_utils::getLoops(active_scope));
  Val* in1 = top->in1();
  Val* in2 = top->in2();
  Val* in3 = top->in3();

  if (ir_utils::isTV(in1))
    in1 = Index::getProducerIndex(
        ir_utils::asTV(in1),
        ir_utils::asTV(top->out()),
        scope_utils::getLoops(active_scope));

  if (ir_utils::isTV(in2))
    in2 = Index::getProducerIndex(
        ir_utils::asTV(in2),
        ir_utils::asTV(top->out()),
        scope_utils::getLoops(active_scope));

  if (ir_utils::isTV(in3))
    in3 = Index::getProducerIndex(
        ir_utils::asTV(in3),
        ir_utils::asTV(top->out()),
        scope_utils::getLoops(active_scope));

  Expr* new_op = new TernaryOp(top->getTernaryOpType(), out, in1, in2, in3);

  return new_op;
}

Statement* IndexLowering::mutate(ReductionOp* rop) {
  TORCH_INTERNAL_ASSERT(
      ir_utils::isTVOp(rop),
      "Cannot have a reduction operation on something other than a tensor view.");
  auto loops = scope_utils::getLoops(active_scope);

  bool is_private_reduce =
      std::none_of(loops.begin(), loops.end(), [](ForLoop* fl) {
        return fl->iter_domain()->isThread() &&
            fl->iter_domain()->isReduction();
      });

  TensorIndex* out = Index::getConsumerIndex(ir_utils::asTV(rop->out()), loops);

  Val* in = rop->in();
  if (ir_utils::isTV(in))
    in = Index::getProducerIndex(
        ir_utils::asTV(in),
        ir_utils::asTV(rop->out()),
        scope_utils::getLoops(active_scope));

  if (!is_private_reduce)
    return new ReductionOp(rop->getReductionOpType(), rop->init(), out, in);

  Expr* new_op = new BinaryOp(rop->getReductionOpType(), out, out, in);

  return new_op;
}

Statement* IndexLowering::mutate(BroadcastOp* bop) {
  if (!ir_utils::isTVOp(bop))
    return OptOutMutator::mutate(bop);

  TensorIndex* out = Index::getConsumerIndex(
      ir_utils::asTV(bop->out()), scope_utils::getLoops(active_scope));
  Val* in = bop->in();
  if (ir_utils::isTV(in))
    in = Index::getProducerIndex(
        ir_utils::asTV(in),
        ir_utils::asTV(bop->out()),
        scope_utils::getLoops(active_scope));
  Expr* new_op = new BroadcastOp(out, in);

  return new_op;
}

void IndexLowering::generate(const std::vector<Expr*>& exprs) {
  // Run through loop nests and further lower the expressions
  for (auto* expr : exprs) {
    Statement* mutated_stmt = mutate(expr);
    TORCH_INTERNAL_ASSERT(
        mutated_stmt->isExpr(),
        "Tried to generate a kernel but hit a non expression during lowering: ",
        mutated_stmt);
    lowered_exprs.push_back(static_cast<Expr*>(mutated_stmt));
  }
}

} // namespace fuser
} // namespace jit
} // namespace torch
