#include <torch/csrc/jit/fuser/common/fusion.h>
#include <torch/csrc/jit/fuser/common/ir.h>
#include <torch/csrc/jit/fuser/common/tensor.h>
#include <torch/csrc/jit/fuser/common/ir_printer.h>
#include <torch/csrc/jit/fuser/common/type.h>
#include <torch/csrc/jit/fuser/common/iriostream.h>

namespace torch {
namespace jit {
namespace fuser {

void IRPrinter::print(const Fusion* const fusion) {
  irstream_ << "\nPrinting TensorViews...\n";
  for(auto &val : fusion->vals()) {
    if(val->getValType().value() == ValType::TensorView)
      irstream_ << "\t" << cnt_++ << "\n";
  }
  
  cnt_ = 0;   
  irstream_ << "\nPrinting Operator Expressions...\n";
  traverse(fusion, false /*from_outputs_only*/, {ValType::TensorView}, false /*breadth_first*/);

  cnt_ = 0; 
  irstream_ << "\nPrinting Tensor Expressions...\n";
  traverse(fusion, false /*from_outputs_only*/, {ValType::TensorDomain}, false /*breadth_first*/);
  irstream_ << "\n";
}

// Just printing the expressions as of now. 
void IRPrinter::handle(Statement* s) { 
  if(s->isExpr()) Statement::dispatch(this, s); 
}

void IRPrinter::handle(Expr* e) {
  irstream_ << "\t" << cnt_++ << " "; 
  Expr::dispatch(this, e); 
  irstream_ << "\n"; 
}

void IRPrinter::handle(Val* v) { 
  Val::dispatch(this, v); 
}

void IRPrinter::handle(TensorView* tv) {
  Val::dispatch(this, tv->domain()); 
}

void IRPrinter::handle(TensorDomain* tdom) {
  for(std::vector<const IterDomain*>::size_type i = 0; i < tdom->size(); i++){
    Val::dispatch(this, tdom->axis(i));
  }
}
  
void IRPrinter::handle(IterDomain* idom) {
  Val::dispatch(this, idom->size()); 
}

void IRPrinter::handle(Int* val) { 
  if (val->isSymbolic()) {
    irstream_ << "%i" << val->name();
  } else {
    irstream_ << *(val->value()) ;
  }
}
  
void IRPrinter::handle(Float* val) { 
  if (val->isSymbolic()) {
    irstream_ << "%f" << val->name();
  } else {
    irstream_ << *(val->value()) << "f";
  }
}

void IRPrinter::handle(BinaryOp* bop) {
  Val::dispatch(this, bop->out()); 
  Val::dispatch(this, bop->lhs()); 
  Val::dispatch(this, bop->rhs()); 
}

} // namespace fuser
} // namespace jit
} // namespace torch
