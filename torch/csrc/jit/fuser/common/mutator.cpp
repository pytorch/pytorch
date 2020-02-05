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

const Statement* BaseMutator::mutate(const Add* const add){
    const Val* out = static_cast<const Val*>(add->out()->dispatch_mutator(this));
    const Val* lhs = static_cast<const Val*>(add->lhs()->dispatch_mutator(this)); 
    const Val* rhs = static_cast<const Val*>(add->rhs()->dispatch_mutator(this));
    //TODO CHECK IF ADD CHANGED, RETURN NEW ONE.
    if(out!=add->out()
    || lhs!=add->lhs()
    || rhs!=add->rhs())
        return new Add(out, lhs, rhs);
    return add;
}

void BaseMutator::mutate(Fusion* fusion){
  std::vector<const Expr*> new_exprs;
  std::vector<const Expr*> orig_exprs = fusion->exprs();

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
