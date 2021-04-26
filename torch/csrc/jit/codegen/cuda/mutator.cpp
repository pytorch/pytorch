#include <c10/util/irange.h>
#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/mutator.h>

#include <vector>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

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
      s, e, id->getParallelType(), id->getIterType(), id->isRFactorProduct());
  registerMutation(id, mutated_val);
  return mutated_val;
}

Statement* OptOutMutator::mutate(TensorDomain* td) {
  std::vector<IterDomain*> dom;
  bool mutated = false;
  for (const auto i : c10::irange(td->nDims())) {
    IterDomain* id = mutateAsVal(td->axis(i))->as<IterDomain>();
    dom.push_back(id);
    if (!id->sameAs(td->axis(i)))
      mutated = true;
  }

  if (mutated) {
    Val* mutated_val = new TensorDomain(
        td->getRootDomain(), td->getRFactorDomain(), dom, td->contiguity());
    registerMutation(td, mutated_val);
    return mutated_val;
  }
  return td;
}

Statement* OptOutMutator::mutate(TensorView* tv) {
  TensorDomain* td = mutateAsVal(tv->domain())->as<TensorDomain>();

  TensorView* computeAtView = nullptr;
  if (tv->hasComputeAt())
    computeAtView = mutateAsVal(tv->getComputeAtView())->as<TensorView>();

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

Statement* OptOutMutator::mutate(kir::TensorIndex* ti) {
  return ti;
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

Statement* OptOutMutator::mutate(kir::Allocate* a) {
  return a;
}

Statement* OptOutMutator::mutate(kir::Sync* a) {
  return a;
}

Statement* OptOutMutator::mutate(Split* s) {
  IterDomain* ot = mutateAsVal(s->outer())->as<IterDomain>();
  IterDomain* inr = mutateAsVal(s->inner())->as<IterDomain>();
  IterDomain* in = mutateAsVal(s->in())->as<IterDomain>();
  Val* fact = mutateAsVal(s->factor())->as<Val>();

  if (ot->sameAs(s->outer()) && inr->sameAs(s->inner()) &&
      in->sameAs(s->in()) && areEqualScalars(fact, s->factor())) {
    return s;
  }
  FusionGuard::getCurFusion()->removeExpr(s);
  return new Split(ot, inr, in, fact);
}

Statement* OptOutMutator::mutate(Merge* m) {
  IterDomain* ot = mutateAsVal(m->out())->as<IterDomain>();
  IterDomain* otr = mutateAsVal(m->outer())->as<IterDomain>();
  IterDomain* in = mutateAsVal(m->inner())->as<IterDomain>();

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

Statement* OptOutMutator::mutate(kir::GridReduction* gr) {
  return gr;
}

Statement* OptOutMutator::mutate(BroadcastOp* bop) {
  return bop;
}

Statement* OptOutMutator::mutate(kir::ForLoop* fl) {
  return fl;
}

Statement* OptOutMutator::mutate(kir::IfThenElse* ite) {
  return ite;
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
