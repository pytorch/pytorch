#include <torch/csrc/jit/fuser/common/fusion.h>
#include <torch/csrc/jit/fuser/common/mutator.h>

namespace torch {
namespace jit {
namespace fuser {

const Statement* BaseMutator::mutate(
    const Statement* const statement) {
  return statement->dispatch_mutator(this);
}
const Statement* BaseMutator::mutate(const Val* const val) {
  return val->dispatch_mutator(this);
}

const Statement* BaseMutator::mutate(const Expr* const expr) {
  return expr->dispatch_mutator(this);
}

const Statement* BaseMutator::mutate(const Float* const f) {
  return f;
}

const Statement* BaseMutator::mutate(const Int* const i) {
  return i;
}

const Statement* BaseMutator::mutate(const UnaryOp* const uop) {
  const Val* out = static_cast<const Val*>(uop->out()->dispatch_mutator(this));
  const Val* in = static_cast<const Val*>(uop->in()->dispatch_mutator(this));

  if 
  (
    !(
         out->same_as(uop->out())
      && in->same_as(uop->in())
    )
  )
    return new UnaryOp(uop->type(), out, in);
  return uop;
}

const Statement* BaseMutator::mutate(const BinaryOp* const bop) {
  const Val* out = static_cast<const Val*>(bop->out()->dispatch_mutator(this));
  const Val* lhs = static_cast<const Val*>(bop->lhs()->dispatch_mutator(this));
  const Val* rhs = static_cast<const Val*>(bop->rhs()->dispatch_mutator(this));
  if
  (
    !(
         out != bop->out()
      && lhs != bop->lhs()
      && rhs != bop->rhs()
    )
  )
    return new BinaryOp(bop->type(), out, lhs, rhs);
  return bop;
}

void BaseMutator::mutate(Fusion* fusion) {
  std::vector<const Expr*> orig_exprs = fusion->exprs();

  /*
   * We go through all the exprs, in topologically sorted order. We call mutate on them
   * which could insert nodes, removes nodes, or both. These operations modify the dag
   * and the Fusion will keep track of what has/hasn't been changed by the origin dependency
   * tracking that it does. If an operation is added, and its output node is a val which
   * previously was the output of another expresion, that older expresion will be removed
   * as we can only assign a Val once due to our SSA restriction. Therefore we don't need
   * to manually track what expressions stayed constant or were changed.
   */

  for(const Statement* stmt : orig_exprs)
    stmt->dispatch_mutator(this);

}

/*
 * TODO: Test the below mutator functions
 */

const Statement* BaseMutator::mutate(const TensorDomain* const td) {
  
  std::vector<const IterDomain*> dom;
  bool mutated = false;
  for(decltype(td->size()) i = 0; i<td->size(); i++){
    const IterDomain* id = static_cast<const IterDomain*>(mutate(td->axis(i)));
    if(!id->same_as(td->axis(i))){
      mutated = true;
    }
  }
  
  if(mutated)
    return new TensorDomain(dom);
  return td;
  
}

const Statement* BaseMutator::mutate(const TensorView* const tv) {
  const Tensor* t = static_cast<const Tensor*>(mutate(tv->tensor()));
  const TensorDomain* td = static_cast<const TensorDomain*>( mutate(tv->domain()));

  if(!(  tv->tensor()->same_as(t)
      && tv->domain()->same_as(td)))
      return new TensorView(t, td);
 return tv;
}

const Statement* BaseMutator::mutate(const IterDomain* const id) {
  const Int* s = static_cast<const Int*>(mutate(id->size()));
  if(!s->same_as(id->size()))
    return new IterDomain(s, id->parallel_method(), id->isReduction());
  return id;
}

const Statement* BaseMutator::mutate(const Tensor* const t) {
  return t; //I believe tensor should never be mutated.
}

const Statement* BaseMutator::mutate(const Split* const s) {
  const TensorView* o = static_cast<const TensorView*>(mutate(s->out()));
  const TensorView* i = static_cast<const TensorView*>(mutate(s->in()));
  const Int* fact = static_cast<const Int*>(mutate(s->factor()));

  if(!(
       o->same_as(s->out())
    && i->same_as(s->in())
    && fact->same_as(s->factor())
  ))
    return new Split(o, i, s->axis(), fact);
  return s;
}

const Statement* BaseMutator::mutate(const Merge* const m) {
  const TensorView* o = static_cast<const TensorView*>(mutate(m->out()));
  const TensorView* i = static_cast<const TensorView*>(mutate(m->in()));

  if(!(
       o->same_as(m->out())
    && i->same_as(m->in())
  ))
    return new Merge(o, i, m->axis());
  return m;
}

const Statement* BaseMutator::mutate(const Reorder* const ro) {
  const TensorView* o = static_cast<const TensorView*>(mutate(ro->out()));
  const TensorView* i = static_cast<const TensorView*>(mutate(ro->in()));

  if(!(
       o->same_as(ro->out())
    && i->same_as(ro->in())
  ))
    return new Reorder(o, i, ro->pos2axis());
  return ro;
}

#include <torch/csrc/jit/fuser/common/mutator.h>
const Statement* ReplaceAll::mutate(const Val* const val){
  if(val->same_as(instance_)){
    std::cout<<"replace?"<<std::endl;
    return with_;
  }
  std::cout<<"Don't replace "<<val<<" with "<<with_<<std::endl;
  return val;
}

void ReplaceAll::instancesOf(const Val* const instance, const Val* const with){

  std::set<const Expr*> exprs_containing_val;

  Fusion *fusion = FusionGuard::getCurFusion();
  const Expr* orig = fusion->origin(instance);
  if(orig != nullptr)
    exprs_containing_val.emplace(orig);

  const std::set<const Expr*> exprs = fusion->uses(instance);
  for(const Expr* expr : exprs)
    exprs_containing_val.emplace(expr);

  ReplaceAll ra(instance, with);

  std::cout<<"Exprs to check : "<<exprs_containing_val.size()<<std::endl;
  std::cout<<"Base val dispatch?"<<std::endl;
  for(const Expr* expr : exprs_containing_val)
    expr->dispatch_mutator(static_cast<BaseMutator*>(&ra));

}


} // namespace fuser
} // namespace jit
} // namespace torch
