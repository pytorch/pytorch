#include <c10/util/irange.h>
#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/mutator.h>

#include <vector>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

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

  if (!tv->domain()->sameAs(td)) {
    TensorView* mutated_tv = new TensorView(td, tv->getDataType().value());
    registerMutation(tv, mutated_tv);
    return mutated_tv;
  }
  return tv;
}

Statement* OptOutMutator::mutate(Bool* b) {
  return b;
}

Statement* OptOutMutator::mutate(Double* d) {
  return d;
}

Statement* OptOutMutator::mutate(Int* i) {
  return i;
}

Statement* OptOutMutator::mutate(NamedScalar* ns) {
  return ns;
}

// MUTATE FUNCTIONS FOR EXPRESSIONS.

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
  return new Split(ot, inr, in, fact, s->innerSplit());
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

namespace {
__inline__ bool compareOptional(Val* a, Val* b) {
  if (!a || !b) {
    return (!a && !b);
  }
  return a->sameAs(b);
}

} // namespace

Statement* OptOutMutator::mutate(WelfordOp* wop) {
  Val* out_avg = mutateAsVal(wop->outAvg())->asVal();
  Val* out_var = mutateAsVal(wop->outVar())->asVal();
  Val* out_N = mutateAsVal(wop->outN())->asVal();

  Val* in_avg = mutateAsVal(wop->inAvg())->asVal();
  Val* in_var = wop->inVar() ? mutateAsVal(wop->inVar())->asVal() : nullptr;
  Val* in_N = mutateAsVal(wop->inN())->asVal();

  Val* init_avg =
      wop->initAvg() ? mutateAsVal(wop->initAvg())->asVal() : nullptr;
  Val* init_var =
      wop->initVar() ? mutateAsVal(wop->initVar())->asVal() : nullptr;
  Val* init_N = mutateAsVal(wop->initN())->asVal();

  const bool out_compare = out_avg->sameAs(wop->outAvg()) &&
      out_var->sameAs(wop->outVar()) && out_N->sameAs(wop->outN());
  const bool in_compare = in_avg->sameAs(wop->inAvg()) &&
      compareOptional(in_var, wop->inVar()) && in_N->sameAs(wop->inN());
  const bool init_compare = compareOptional(init_avg, wop->initAvg()) &&
      compareOptional(init_var, wop->initVar()) && init_N->sameAs(wop->initN());

  if (out_compare && init_compare && in_compare) {
    return wop;
  } else {
    return new WelfordOp(
        out_avg,
        out_var,
        out_N,
        init_avg,
        init_var,
        init_N,
        in_avg,
        in_var,
        in_N);
  }
}

Statement* OptOutMutator::mutate(BroadcastOp* bop) {
  return bop;
}

Statement* OptOutMutator::mutate(TransposeOp* top) {
  return top;
}

Statement* OptOutMutator::mutate(ShiftOp* sop) {
  Val* out = mutateAsVal(sop->out())->asVal();
  Val* in = mutateAsVal(sop->in())->asVal();

  if (out->sameAs(sop->out()) && in->sameAs(sop->in()))
    return sop;
  auto offsets = sop->offsets();
  FusionGuard::getCurFusion()->removeExpr(sop);
  return new ShiftOp(out, in, offsets);
}

Statement* OptOutMutator::mutate(GatherOp* op) {
  Val* out = mutateAsVal(op->out())->asVal();
  Val* in = mutateAsVal(op->in())->asVal();

  if (out->sameAs(op->out()) && in->sameAs(op->in()))
    return op;
  auto window_shape = op->windowShape();
  auto pad_width = op->padWidth();
  FusionGuard::getCurFusion()->removeExpr(op);
  return new GatherOp(out, in, window_shape, pad_width);
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
