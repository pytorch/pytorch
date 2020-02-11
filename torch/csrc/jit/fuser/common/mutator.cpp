#include <torch/csrc/jit/fuser/common/mutator.h>
#include <torch/csrc/jit/fuser/common/fusion.h>

namespace torch {
namespace jit {
namespace fuser {

const Statement* BaseMutator::mutate(const Statement* const statement){
    //throw std::runtime_error("Could not identify statement. Did you update dispatch_mutator in ir.cpp?");
    return statement->dispatch_mutator(this);
}

const Statement* BaseMutator::mutate(const Float* const f){
    if(!f->isSymbolic())
        if(*(f->value()) == 1.0){
            Float* f2 = new Float(0.0);
            return f2;
        }
    return f;
}

const Statement* BaseMutator::mutate(const Int* const i){
    return i;
}

const Statement* BaseMutator::mutate(const UnaryOp* const uop){
    const Val* out = static_cast<const Val*>(uop->out()->dispatch_mutator(this));
    const Val* in  = static_cast<const Val*>(uop->in()->dispatch_mutator(this)); 
    //TODO CHECK IF ADD CHANGED, RETURN NEW ONE.
    if(out!=uop->out()
    || in!=uop->in())
        return new UnaryOp(uop->type(), out, in);
    return uop;
}

const Statement* BaseMutator::mutate(const BinaryOp* const bop){
    const Val* out = static_cast<const Val*>(bop->out()->dispatch_mutator(this));
    const Val* lhs = static_cast<const Val*>(bop->lhs()->dispatch_mutator(this)); 
    const Val* rhs = static_cast<const Val*>(bop->rhs()->dispatch_mutator(this));
    //TODO CHECK IF ADD CHANGED, RETURN NEW ONE.
    if(out!=bop->out()
    || lhs!=bop->lhs()
    || rhs!=bop->rhs())
        return new BinaryOp(bop->type(), out, lhs, rhs);
    return bop;
}

void BaseMutator::mutate(Fusion* fusion){

  std::vector<const Expr*> new_exprs;
  std::vector<const Expr*> orig_exprs(fusion->exprs().begin(), fusion->exprs().end());

  for(std::vector<const Expr*>::size_type i = 0; i < orig_exprs.size(); i++){
      const Statement* new_stmt = orig_exprs[i]->dispatch_mutator(this);
      assert(new_stmt->isExpr());
      new_exprs.push_back(static_cast<const Expr*>(new_stmt));  
  }

  for(std::vector<const Expr*>::size_type i = 0; i < fusion->exprs().size(); i++){
    if(orig_exprs[i] != new_exprs[i]){
        fusion->removeExpr(orig_exprs[i]);
    }
  }

}

}}} // torch::jit::fuser
