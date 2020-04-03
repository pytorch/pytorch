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

/*
 * TODO: All of the mutator functions below need to be reviewed and tested.
 */

Statement* OptOutMutator::mutate(IterDomain* id) {
  Int* s = static_cast<Int*>(mutate(id->size()));
  if (!s->sameAs(id->size()))
    return new IterDomain(s, id->parallel_method(), id->isReduction());
  return id;
}

Statement* OptOutMutator::mutate(TensorDomain* td) {
  std::vector<IterDomain*> dom;
  bool mutated = false;
  for (decltype(td->size()) i = 0; i < td->size(); i++) {
    IterDomain* id = static_cast<IterDomain*>(mutate(td->axis(i)));
    if (!id->sameAs(td->axis(i))) {
      mutated = true;
    }
  }

  if (mutated)
    return new TensorDomain(dom);
  return td;
}

Statement* OptOutMutator::mutate(TensorView* tv) {
  TensorDomain* td = static_cast<TensorDomain*>(mutate(tv->domain()));

  if (!(tv->domain()->sameAs(td)))
    return new TensorView(td, tv->getDataType().value());
  return tv;
}

Statement* OptOutMutator::mutate(Float* n) {
  return n;
}
Statement* OptOutMutator::mutate(Int* n) {
  return n;
}

Statement* OptOutMutator::mutate(Split* s) {
  TensorDomain* o = static_cast<TensorDomain*>(mutate(s->out()));
  TensorDomain* i = static_cast<TensorDomain*>(mutate(s->in()));
  Int* fact = static_cast<Int*>(mutate(s->factor()));

  if (!(o->sameAs(s->out()) && i->sameAs(s->in()) && fact->sameAs(s->factor())))
    return new Split(o, i, s->axis(), fact);
  return s;
}

Statement* OptOutMutator::mutate(Merge* m) {
  TensorDomain* o = static_cast<TensorDomain*>(mutate(m->out()));
  TensorDomain* i = static_cast<TensorDomain*>(mutate(m->in()));

  if (!(o->sameAs(m->out()) && i->sameAs(m->in())))
    return new Merge(o, i, m->axis());
  return m;
}

Statement* OptOutMutator::mutate(Reorder* ro) {
  TensorDomain* o = static_cast<TensorDomain*>(mutate(ro->out()));
  TensorDomain* i = static_cast<TensorDomain*>(mutate(ro->in()));

  if (!(o->sameAs(ro->out()) && i->sameAs(ro->in())))
    return new Reorder(o, i, ro->pos2axis());
  return ro;
}

Statement* OptOutMutator::mutate(UnaryOp* uop) {
  Val* out = static_cast<Val*>(mutate(uop->out()));
  Val* in = static_cast<Val*>(mutate(uop->in()));

  if (!(out->sameAs(uop->out()) && in->sameAs(uop->in())))
    return new UnaryOp(uop->getUnaryOpType(), out, in);
  return uop;
}

Statement* OptOutMutator::mutate(BinaryOp* bop) {
  Val* out = static_cast<Val*>(mutate(bop->out()));
  Val* lhs = static_cast<Val*>(mutate(bop->lhs()));
  Val* rhs = static_cast<Val*>(mutate(bop->rhs()));
  if (!(out != bop->out() && lhs != bop->lhs() && rhs != bop->rhs()))
    return new BinaryOp(bop->getBinaryOpType(), out, lhs, rhs);
  return bop;
}

Statement* OptInMutator::mutate(Statement* s) {
  return Statement::mutatorDispatch(this, s);
}
Statement* OptInMutator::mutate(Expr* e) {
  return Expr::mutatorDispatch(this, e);
}
Statement* OptInMutator::mutate(Val* v) {
  return Val::mutatorDispatch(this, v);
}

Statement* ReplaceAll::mutate(Val* val) {
  if (val->sameAs(instance_))
    return with_;
  return val;
}

void ReplaceAll::instancesOf(Val* instance, Val* with) {
  std::set<Expr*> exprs_containing_val;

  Fusion* fusion = FusionGuard::getCurFusion();
  Expr* orig = fusion->origin(instance);
  if (orig != nullptr)
    exprs_containing_val.emplace(orig);

  std::set<Expr*> exprs = fusion->uses(instance);
  for (Expr* expr : exprs)
    exprs_containing_val.emplace(expr);

  ReplaceAll ra(instance, with);

  for (Expr* expr : exprs_containing_val)
    ra.mutate(expr);

  fusion->removeVal(instance);
}

} // namespace fuser
} // namespace jit
} // namespace torch
