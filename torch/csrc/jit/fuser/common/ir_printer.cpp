#include <torch/csrc/jit/fuser/common/fusion.h>
#include <torch/csrc/jit/fuser/common/ir.h>
#include <torch/csrc/jit/fuser/common/tensor.h>
#include <torch/csrc/jit/fuser/common/ir_printer.h>
#include <torch/csrc/jit/fuser/common/type.h>
#include <torch/csrc/jit/fuser/common/iriostream.h>

namespace torch {
namespace jit {
namespace fuser {

void IRMathPrinter::print(const Fusion* const fusion) {
  irstream_ << "\nPrinting TensorViews...\n";
  for(auto &val : fusion->vals()) {
    if(val->getValType().value() == ValType::TensorView)
      irstream_ << "\t" << cnt_++ << "\n";
  }
  
  cnt_ = 0;   
  irstream_ << "\nPrinting Operator Expressions...\n";
  traverse(fusion, false /*from_outputs_only*/, false /*breadth_first*/, {ValType::TensorView});


  cnt_ = 0; 
  irstream_ << "\nPrinting Tensor Expressions...\n";
  //traverse(fusion, false /*from_outputs_only*/, false /*breadth_first*/, {ValType::TensorDomain});
  irstream_ << "\n";
}

// Just printing the expressions as of now. 
void IRMathPrinter::handle(Statement* s) { 
  if(s->isExpr()) 
    IRPrinter::handle(s); 
}

void IRMathPrinter::handle(Expr* e) {
  irstream_ << "\t" << cnt_++ << " ";
  IRPrinter::handle(e); 
  irstream_ << "\n"; 
}

void IRMathPrinter::handle(Tensor* t) {
  irstream_ << "%T" << t->name(); 
  handle(t->domain());
}

void IRMathPrinter::handle(TensorView* tv) {
  irstream_ << "%TV" << tv->name();

  const TensorDomain* td = dynamic_cast<TensorDomain*>(tv->domain());
  irstream_ << "[";
  for(std::vector<const IterDomain*>::size_type i = 0; i < td->size(); i++){
    if(i > 0) irstream_ << ", ";
    if(tv->hasComputeAt()) {
      if(static_cast<std::vector<const IterDomain*>::size_type>(tv->getComputeAtAxis()) == i) {
        irstream_ << "@ ";
      }
    }
    handle(td->axis(i));
  }
  if(tv->hasComputeAt()) {
    if(static_cast<std::vector<const IterDomain*>::size_type>(tv->getComputeAtAxis()) == td->size()) {
      irstream_ << " @ ";
    }
  }
  irstream_ << "]";
}

void IRMathPrinter::handle(TensorDomain* tdom) {
  irstream_ << "[";
  for(std::vector<const IterDomain*>::size_type i = 0; i < tdom->size(); i++){
    if(i > 0) irstream_ << ", ";
    handle(tdom->axis(i));
  }
  irstream_ << "]";
}
  
void IRMathPrinter::handle(IterDomain* idom) {
  const Val* val_id_size = idom->size();
  TORCH_CHECK(val_id_size->getValType().value() == ValType::Scalar);
  TORCH_CHECK(val_id_size->getDataType().value() == DataType::Int);
  
  const Int* id_size = dynamic_cast<Int*>(idom->size());
  if(idom->isReduction()) irstream_ << ">";
  if(idom->parallel_method() == ParallelType::Unroll) irstream_ << "o__" ;
  if(idom->parallel_method() == ParallelType::Vectorize) irstream_ << "{" ;
  if(id_size->isSymbolic()) {
    irstream_ << "%D" << id_size->name();
  } else {
    irstream_ << id_size->value().value();
  }
  if(idom->parallel_method() == ParallelType::Vectorize) irstream_ << "}" ;
  if(idom->isReduction()) irstream_ << "<";
}

void IRMathPrinter::handle(Int* val) { 
  if (val->isSymbolic()) {
    irstream_ << "%i" << val->name();
  } else {
    irstream_ << *(val->value()) ;
  }
}
  
void IRMathPrinter::handle(Float* val) { 
  if (val->isSymbolic()) {
    irstream_ << "%f" << val->name();
  } else {
    irstream_ << *(val->value()) << "f";
  }
}

void IRMathPrinter::handle(BinaryOp* bop) {
  IRPrinter::handle(bop->out());
  irstream_ << " = ";
  if(auto inline_bop = inline_op_str(bop->type())) {
    IRPrinter::handle(bop->lhs());
    irstream_  << " " << inline_bop.value() << " ";
    IRPrinter::handle(bop->rhs());
  } else {
    irstream_ << bop->type() << "("; 
    IRPrinter::handle(bop->lhs()); 
    irstream_ << ", ";
    IRPrinter::handle(bop->rhs());
    irstream_  << ")";
  }
}

void IRMathPrinter::handle(UnaryOp* uop) {
  IRPrinter::handle(uop->out());
  irstream_ << " = ";
  if(auto inline_uop = inline_op_str(uop->type())) {
    irstream_  << inline_uop.value();
    IRPrinter::handle(uop->in());
  } else {
    irstream_ << uop->type() << "("; 
    IRPrinter::handle(uop->in()); 
    irstream_  << ")";
  }
}

} // namespace fuser
} // namespace jit
} // namespace torch
