#include <torch/csrc/jit/codegen/cuda/mutator.h>
#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>

#include <vector>

namespace torch {
namespace jit {
namespace fuser {

void OptOutMutator::mutate(Fusion* fusion) {
  std::vector<Expr*> orig_exprs = fusion->exprs();

  /*
   * We go through all the exprs, in topologically sorted order. We call mutate
   * on them which could insert nodes, removes nodes, or both. These operations
   * modify the dag and the Fusion will keep track of what has/hasn't been
   * changed by the origin dependency tracking that it does. If an operation is
   * added, and its output node is a val which previously was the output of
   * another expresion, that older expresion will be removed as we can only
   * assign a Val once due to our SSA restriction. Therefore we don't need to
   * manually track what expressions stayed constant or were changed.
   */

  for (Statement* stmt : orig_exprs)
    mutate(stmt);
}

// MUTATE FUNCTIONS FOR VALS

Statement* OptOutMutator::mutate(IterDomain* id) {
  Val* s = mutateAsVal(id->start())->asVal();
  Val* e = mutateAsVal(id->extent())->asVal();
  if (s->sameAs(id->start()) && e->sameAs(id->extent()))
    return id;

  Val* mutated_val = new IterDomain(
      s, e, id->parallel_method(), id->isReduction(), id->isRFactorProduct());
  registerMutation(id, mutated_val);
  return mutated_val;
}

Statement* OptOutMutator::mutate(TensorDomain* td) {
  std::vector<IterDomain*> dom;
  bool mutated = false;
  for (decltype(td->nDims()) i = 0; i < td->nDims(); i++) {
    IterDomain* id = static_cast<IterDomain*>(mutateAsVal(td->axis(i)));
    dom.push_back(id);
    if (!id->sameAs(td->axis(i)))
      mutated = true;
  }

  if (mutated) {
    Val* mutated_val = new TensorDomain(dom);
    registerMutation(td, mutated_val);
    return mutated_val;
  }
  return td;
}

Statement* OptOutMutator::mutate(TensorView* tv) {
  TensorDomain* td = static_cast<TensorDomain*>(mutateAsVal(tv->domain()));

  TensorView* computeAtView = nullptr;
  if (tv->hasComputeAt())
    computeAtView =
        static_cast<TensorView*>(mutateAsVal(tv->getComputeAtView()));

  if (!tv->domain()->sameAs(td) ||
      (tv->hasComputeAt() && !tv->getComputeAtView()->sameAs(computeAtView))) {
    TensorView* mutated_tv = new TensorView(td, tv->getDataType().value());
    if (tv->hasComputeAt()) {
      mutated_tv->setComputeAt(
          computeAtView, (int)(tv->getRelativeComputeAtAxis()));
    }
    registerMutation(tv, mutated_tv);
    return mutated_tv;
  }
  return tv;
}

Statement* OptOutMutator::mutate(TensorIndex* ti) {
  std::vector<Statement*> inds;
  for (auto* ind : ti->indices())
    inds.push_back(mutateAsVal(ind));

  bool changed = false;
  for (decltype(inds.size()) i{0}; i < inds.size(); i++) {
    TORCH_INTERNAL_ASSERT(inds[i]->isVal() && inds[i]->asVal()->isAnInt());
    if (!inds[i]->sameAs(ti->index(i)))
      changed = true;
  }

  if (!changed)
    return ti;

  std::vector<Val*> valInds(inds.size(), nullptr);
  for (decltype(inds.size()) i{0}; i < inds.size(); i++)
    valInds[i] = inds[i]->asVal();

  Val* mutated_val = new TensorIndex(ti->view(), valInds);
  registerMutation(ti, mutated_val);
  return mutated_val;
}

Statement* OptOutMutator::mutate(Bool* b) {
  return b;
}

Statement* OptOutMutator::mutate(Float* f) {
  return f;
}

Statement* OptOutMutator::mutate(Half* h) {
  return h;
}

Statement* OptOutMutator::mutate(Int* i) {
  return i;
}

Statement* OptOutMutator::mutate(NamedScalar* ns) {
  return ns;
}

// MUTATE FUNCTIONS FOR EXPRESSIONS.

Statement* OptOutMutator::mutate(Allocate* a) {
  TensorView* tv = static_cast<TensorView*>(mutateAsVal(a->buffer()));
  Val* ext = mutateAsVal(a->extent())->asVal();
  if (ext->sameAs(a->extent()) && tv->sameAs(a->buffer()))
    return a;
  FusionGuard::getCurFusion()->removeExpr(a);
  return new Allocate(tv, ext);
}

Statement* OptOutMutator::mutate(Split* s) {
  IterDomain* ot = static_cast<IterDomain*>(mutateAsVal(s->outer()));
  IterDomain* inr = static_cast<IterDomain*>(mutateAsVal(s->inner()));
  IterDomain* in = static_cast<IterDomain*>(mutateAsVal(s->in()));
  Int* fact = static_cast<Int*>(mutateAsVal(s->factor()));

  if (ot->sameAs(s->outer()) && inr->sameAs(s->inner()) &&
      in->sameAs(s->in()) && fact->sameAs(s->factor()))
    return s;
  FusionGuard::getCurFusion()->removeExpr(s);
  return new Split(ot, inr, in, fact);
}

Statement* OptOutMutator::mutate(Merge* m) {
  IterDomain* ot = static_cast<IterDomain*>(mutateAsVal(m->out()));
  IterDomain* otr = static_cast<IterDomain*>(mutateAsVal(m->outer()));
  IterDomain* in = static_cast<IterDomain*>(mutateAsVal(m->inner()));

  if (ot->sameAs(m->out()) && otr->sameAs(m->outer()) && in->sameAs(m->inner()))
    return m;

  FusionGuard::getCurFusion()->removeExpr(m);
  return new Merge(ot, otr, in);
}

Statement* OptOutMutator::mutate(UnaryOp* uop) {
  Val* out = mutateAsVal(uop->out())->asVal();
  Val* in = mutateAsVal(uop->in())->asVal();

  if (out->sameAs(uop->out()) && in->sameAs(uop->in()))
    return uop;
  FusionGuard::getCurFusion()->removeExpr(uop);
  return new UnaryOp(uop->getUnaryOpType(), out, in);
}

Statement* OptOutMutator::mutate(BinaryOp* bop) {
  Val* out = mutateAsVal(bop->out())->asVal();
  Val* lhs = mutateAsVal(bop->lhs())->asVal();
  Val* rhs = mutateAsVal(bop->rhs())->asVal();
  if (out == bop->out() && lhs == bop->lhs() && rhs == bop->rhs())
    return bop;
  FusionGuard::getCurFusion()->removeExpr(bop);
  return new BinaryOp(bop->getBinaryOpType(), out, lhs, rhs);
}

Statement* OptOutMutator::mutate(TernaryOp* top) {
  Val* out = mutateAsVal(top->out())->asVal();
  Val* in1 = mutateAsVal(top->in1())->asVal();
  Val* in2 = mutateAsVal(top->in2())->asVal();
  Val* in3 = mutateAsVal(top->in3())->asVal();
  if (out == top->out() && in1 == top->in1() && in2 == top->in2() &&
      in3 == top->in3())
    return top;
  FusionGuard::getCurFusion()->removeExpr(top);
  return new TernaryOp(top->getTernaryOpType(), out, in1, in2, in3);
}

Statement* OptOutMutator::mutate(ReductionOp* rop) {
  Val* out = mutateAsVal(rop->out())->asVal();
  Val* in = mutateAsVal(rop->in())->asVal();
  Val* init = rop->init();
  if (out->sameAs(rop->out()) && in->sameAs(rop->in()) &&
      init->sameAs(rop->init()))
    return rop;

  return new ReductionOp(rop->getReductionOpType(), init, out, in);
}

Statement* OptOutMutator::mutate(BroadcastOp* bop) {
  Val* out = mutateAsVal(bop->out())->asVal();
  Val* in = mutateAsVal(bop->in())->asVal();
  if (out->sameAs(bop->out()) && in->sameAs(bop->in()))
    return bop;

  TORCH_INTERNAL_ASSERT(
      out->getValType().value() == ValType::TensorView &&
      in->getValType().value() == ValType::TensorView)
  return new BroadcastOp(
      static_cast<TensorView*>(out), static_cast<TensorView*>(in));
}

Statement* OptOutMutator::mutate(ForLoop* fl) {
  Val* index = mutateAsVal(fl->index())->asVal();
  Val* val_id = mutateAsVal(fl->iter_domain())->asVal();

  TORCH_INTERNAL_ASSERT(val_id->getValType() == ValType::IterDomain);
  IterDomain* id = static_cast<IterDomain*>(val_id);

  bool is_mutated = !index->sameAs(fl->index());
  is_mutated = is_mutated | !id->sameAs(fl->iter_domain());

  std::vector<Expr*> mutated_exprs;
  for (auto expr : fl->body().exprs()) {
    Statement* mutated_stmt = mutate(expr);
    TORCH_INTERNAL_ASSERT(
        mutated_stmt->isExpr(),
        "While mutating a for loop, received a non-expression for a body entry.");
    Expr* mutated_expr = static_cast<Expr*>(mutated_stmt);
    mutated_exprs.push_back(mutated_expr);
    // could use sameAs here, but we'd have to check the output value separately
    is_mutated = is_mutated | (mutated_expr != expr);
  }

  if (is_mutated) {
    auto newFL = new ForLoop(index, id, mutated_exprs, fl->parentScope());
    return newFL;
  }

  return fl;
}

Statement* OptOutMutator::mutate(IfThenElse* ite) {
  Val* val_cond = mutateAsVal(ite->cond())->asVal();
  TORCH_INTERNAL_ASSERT(
      val_cond->getValType().value() == ValType::Scalar &&
      val_cond->getDataType().value() == DataType::Bool);
  Bool* cond = static_cast<Bool*>(val_cond);

  bool is_mutated = !cond->sameAs(ite->cond());

  std::vector<Expr*> mutated_exprs;
  for (auto expr : ite->body().exprs()) {
    Statement* mutated_stmt = mutate(expr);
    TORCH_INTERNAL_ASSERT(
        mutated_stmt->isExpr(),
        "While mutating a for loop, received a non-expression for a body entry.");
    Expr* mutated_expr = static_cast<Expr*>(mutated_stmt);
    mutated_exprs.push_back(mutated_expr);
    // could use sameAs here, but we'd have to check the output value separately
    is_mutated = is_mutated | (mutated_expr != expr);
  }

  std::vector<Expr*> mutated_else_exprs;
  for (auto expr : ite->elseBody().exprs()) {
    Statement* mutated_stmt = mutate(expr);
    TORCH_INTERNAL_ASSERT(
        mutated_stmt->isExpr(),
        "While mutating a for loop, received a non-expression for a body entry.");
    Expr* mutated_expr = static_cast<Expr*>(mutated_stmt);
    mutated_else_exprs.push_back(mutated_expr);
    // could use sameAs here, but we'd have to check the output value separately
    is_mutated = is_mutated | (mutated_expr != expr);
  }

  if (is_mutated) {
    auto newITE = new IfThenElse(
        cond, ite->body().exprs(), ite->elseBody().exprs(), ite->parentScope());
    return newITE;
  }

  return ite;
}

// START REPLACE ALL

void ReplaceAll::replaceInpOut() {
  Fusion* fusion = FusionGuard::getCurFusion();
  for (auto it : mutations) {
    Val* val = it.first;
    if (fusion->hasInput(val)) {
      fusion->replaceInput(it.first, it.second);
    } else if (fusion->hasOutput(val)) {
      fusion->replaceOutput(it.first, it.second);
    }
  }
}

void ReplaceAll::instancesOf(Val* instance, Val* with) {
  std::unordered_map<Val*, Val*> replacement_map;
  replacement_map[instance] = with;
  ReplaceAll::instancesOf(replacement_map);
}

void ReplaceAll::instancesOf(std::unordered_map<Val*, Val*> replacement_map) {
  ReplaceAll ra(std::move(replacement_map));
  // Get a copy because this will be modified in place, we shouldn't auto
  // iterate on it
  std::vector<Expr*> to_mutate;
  for (Expr* expr : FusionGuard::getCurFusion()->unordered_exprs())
    to_mutate.push_back(expr);

  for (Expr* expr : to_mutate)
    ra.mutate(expr);

  ra.replaceInpOut();
}

void ReplaceAll::instancesWithin(Val* instance, Val* with, Expr* within) {
  if (within == nullptr)
    return;
  FusionGuard fg(within->fusion());
  ReplaceAll ra(instance, with);
  ra.mutate(within);
}

void ReplaceAll::instancesWithin(
    std::unordered_map<Val*, Val*> replacement_map,
    Expr* within) {
  if (within == nullptr)
    return;
  FusionGuard fg(within->fusion());
  ReplaceAll ra(std::move(replacement_map));
  ra.mutate(within);
}

} // namespace fuser
} // namespace jit
} // namespace torch
