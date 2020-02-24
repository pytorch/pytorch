#include <torch/csrc/jit/fuser/common/fusion.h>
#include <torch/csrc/jit/fuser/common/mutator.h>

namespace torch {
namespace jit {
namespace fuser {

 Statement* BaseMutator::mutate(
     Statement*  statement) {
  return Statement::dispatch_mutator(this, statement);
}
 Statement* BaseMutator::mutate( Val*  val) {
  return Val::dispatch_mutator(this, val);
}

 Statement* BaseMutator::mutate( Expr*  expr) {
  return Expr::dispatch_mutator(this, expr);
}

 Statement* BaseMutator::mutate( Float*  f) {
  return f;
}

 Statement* BaseMutator::mutate( Int*  i) {
  return i;
}

 Statement* BaseMutator::mutate( UnaryOp*  uop) {
   Val* out = static_cast<Val*>( mutate(uop->out()) );
   Val* in = static_cast<Val*>( mutate(uop->in()) );

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

 Statement* BaseMutator::mutate( BinaryOp*  bop) {
   Val* out = static_cast< Val*>( mutate(bop->out()) );
   Val* lhs = static_cast< Val*>( mutate(bop->lhs()) );
   Val* rhs = static_cast< Val*>( mutate(bop->rhs()) );
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
  std::vector< Expr*> orig_exprs = fusion->exprs();

  /*
   * We go through all the exprs, in topologically sorted order. We call mutate on them
   * which could insert nodes, removes nodes, or both. These operations modify the dag
   * and the Fusion will keep track of what has/hasn't been changed by the origin dependency
   * tracking that it does. If an operation is added, and its output node is a val which
   * previously was the output of another expresion, that older expresion will be removed
   * as we can only assign a Val once due to our SSA restriction. Therefore we don't need
   * to manually track what expressions stayed constant or were changed.
   */

  for( Statement* stmt : orig_exprs)
    mutate(stmt);

}

/*
 * TODO: Test the below mutator functions
 */

 Statement* BaseMutator::mutate( TensorDomain*  td) {
  
  std::vector< IterDomain*> dom;
  bool mutated = false;
  for(decltype(td->size()) i = 0; i<td->size(); i++){
     IterDomain* id = static_cast< IterDomain*>(mutate(td->axis(i)));
    if(!id->same_as(td->axis(i))){
      mutated = true;
    }
  }
  
  if(mutated)
    return new TensorDomain(dom);
  return td;
  
}

 Statement* BaseMutator::mutate( TensorView*  tv) {
   Tensor* t = nullptr;
   if(tv->tensor() != nullptr)
      t = static_cast< Tensor*>(mutate(tv->tensor()));
   TensorDomain* td = static_cast< TensorDomain*>( mutate(tv->domain()));

    bool same_tensor = tv->tensor() == nullptr
                    || t == nullptr 
                     ? tv->tensor()==nullptr && t == nullptr : tv->tensor()->same_as(t);

  if(!(  same_tensor
      && tv->domain()->same_as(td)))
      return new TensorView(t, td);
 return tv;
}

 Statement* BaseMutator::mutate( IterDomain*  id) {
   Int* s = static_cast< Int*>(mutate(id->size()));
  if(!s->same_as(id->size()))
    return new IterDomain(s, id->parallel_method(), id->isReduction());
  return id;
}

 Statement* BaseMutator::mutate( Tensor*  t) {
  return t; //I believe tensor should never be mutated.
}

 Statement* BaseMutator::mutate( Split*  s) {
   TensorDomain* o = static_cast< TensorDomain*>(mutate(s->out()));
   TensorDomain* i = static_cast< TensorDomain*>(mutate(s->in()));
   Int* fact = static_cast< Int*>(mutate(s->factor()));

  if(!(
       o->same_as(s->out())
    && i->same_as(s->in())
    && fact->same_as(s->factor())
  ))
    return new Split(o, i, s->axis(), fact);
  return s;
}

 Statement* BaseMutator::mutate( Merge*  m) {
   TensorDomain* o = static_cast< TensorDomain*>(mutate(m->out()));
   TensorDomain* i = static_cast< TensorDomain*>(mutate(m->in()));

  if(!(
       o->same_as(m->out())
    && i->same_as(m->in())
  ))
    return new Merge(o, i, m->axis());
  return m;
}

 Statement* BaseMutator::mutate( Reorder*  ro) {
   TensorDomain* o = static_cast< TensorDomain*>(mutate(ro->out()));
   TensorDomain* i = static_cast< TensorDomain*>(mutate(ro->in()));

  if(!(
       o->same_as(ro->out())
    && i->same_as(ro->in())
  ))
    return new Reorder(o, i, ro->pos2axis());
  return ro;
}

 Statement* ReplaceAll::mutate( Val*  val){
  if(val->same_as(instance_))
    return with_;
  return val;
}

void ReplaceAll::instancesOf( Val*  instance,  Val*  with){

  std::set< Expr*> exprs_containing_val;

  Fusion *fusion = FusionGuard::getCurFusion();
   Expr* orig = fusion->origin(instance);
  if(orig != nullptr)
    exprs_containing_val.emplace(orig);

   std::set< Expr*> exprs = fusion->uses(instance);
  for( Expr* expr : exprs)
    exprs_containing_val.emplace(expr);

  ReplaceAll ra(instance, with);

  for( Expr* expr : exprs_containing_val)
    static_cast<BaseMutator*>(&ra)->mutate(expr);

  fusion->removeVal(instance);

}


} // namespace fuser
} // namespace jit
} // namespace torch
