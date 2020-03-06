#include <torch/csrc/jit/fuser/common/fusion.h>
#include <torch/csrc/jit/fuser/common/ir.h>
#include <torch/csrc/jit/fuser/common/iriostream.h>
#include <torch/csrc/jit/fuser/common/tensor.h>

#include <iostream>

namespace torch {
namespace jit {
namespace fuser {

void IRPrinter::print(Fusion* fusion) {
  for (const Expr* expr : fusion->exprs()) {
    print(expr);
  }
}

void IRPrinter::print(const Statement* const stmt) {
  if (stmt->isVal()){
    print(static_cast<const Val*>(stmt));
    return;
  } else if (stmt->isExpr()) {
    print(static_cast<const Expr*>(stmt));
    return;
  }
  throw std::runtime_error("Unkown statment type found in os << Statement.");
}

void IRPrinter::print(const Val* const val) {
  switch (*(val->getValType())) {
    case ValType::Tensor:
      print(static_cast<const Tensor* const>(val)); return;
    case ValType::TensorDomain:
      print(static_cast<const TensorDomain* const>(val)); return;
    case ValType::TensorView:
      print(static_cast<const TensorView* const>(val)); return;
    case ValType::IterDomain:
      print(static_cast<const IterDomain* const>(val)); return;
    case ValType::TensorIndex:
      print(static_cast<const TensorIndex* const>(val)); return;

    case ValType::Scalar:
      switch (*(val->getDataType())) {
        case DataType::Float:
          print(static_cast<const Float* const>(val)); return;
        case DataType::Int:
          print(static_cast<const Int* const>(val)); return;
        default:
          break;
      }
    default:
      break;
  }
  throw std::runtime_error("Unknown ValType in os << Val.");
}

void IRPrinter::print(const Expr* const expr) {
  switch (*(expr->getExprType())) {
    case ExprType::UnaryOp:
      print(static_cast<const UnaryOp* const>(expr)); return;
    case ExprType::BinaryOp:
      print(static_cast<const BinaryOp* const>(expr)); return;
    case ExprType::ForLoop:
      print(static_cast<const ForLoop* const>(expr)); return;
    case ExprType::IfThenElse:
      print(static_cast<const IfThenElse* const>(expr)); return;
    case ExprType::Split:
      print(static_cast<const Split* const>(expr)); return;
    case ExprType::Merge:
      print(static_cast<const Merge* const>(expr)); return;
    case ExprType::Reorder:
      print(static_cast<const Reorder* const>(expr)); return;
  }
  throw std::runtime_error("Unknown ExprType in os << Expr.");
}

TORCH_API void IRPrinter::print(const TensorDomain* const td) {
  os << "[ ";
  for (std::vector<const IterDomain*>::size_type i = 0; i < td->size(); i++) {
    print(td->axis(i));
    if (i != td->size() - 1)
      os << ", ";
  }
  os << " ]";
}

TORCH_API void IRPrinter::print(const TensorView* const tv) {
  if (tv->tensor() != nullptr){
    os << "T" << tv->tensor()->name();
    print(tv->domain());
  }else{
    os << "TV" << tv->name();
    print(tv->domain());
  }
  if (tv->getComputeAtView() != nullptr) {
    os << " compute_at( ";
    if (tv->getComputeAtView()->tensor() == nullptr)
      os << "TV" << tv->getComputeAtView()->name();
    else
      os << "T" << tv->getComputeAtView()->tensor()->name();
    os << ", " << tv->getComputeAtAxis() << " )";
  }
}

TORCH_API void IRPrinter::print(const IterDomain* const id) {
  if (id->isReduction())
    os << "r";
  else
    os << "i";
  switch (id->parallel_method()) {
    case (ParallelType::Vectorize):
      os << "V";
      break;
    case (ParallelType::Unroll):
      os << "U";
      break;
    case (ParallelType::Serial):
      os << "S";
      break;
    default:
      os << id->parallel_method();
  }
  os << "{";
  print_inline(id->size());
  os << "}";
}

TORCH_API void IRPrinter::print(const TensorIndex* const ti) {
  print_inline(ti->size());
}

void IRPrinter::print(const Tensor* const t) {
  os << "T" << t->name();
  if (t->getDataType().has_value())
    os << " scalar_type: " << *(t->getDataType());
  if (t->domain() != nullptr)
    os << " " << t->domain();
  if (t->hasContiguityInfo())
    os << " " << &t->getContiguityInfo().value();
}

void IRPrinter::print(const TensorContiguity* const t) {
  os << "format_tag: " << t->getContiguityTag();
}

void IRPrinter::print(const Float* const f) {
  if (print_inline_ && FusionGuard::getCurFusion()->origin(f) != nullptr) {
    os << "( ";
    print(FusionGuard::getCurFusion()->origin(f));
    os << " )";
    return;
  }

  if (f->isSymbolic()) {
    os << "f" << f->name();
  } else {
    os << *(f->value()) << "f";
  }
}

void IRPrinter::print(const Int* const i) {
  if (print_inline_ && FusionGuard::getCurFusion()->origin(i) != nullptr) {
    os << "( ";
    print(FusionGuard::getCurFusion()->origin(i));
    os << " )";
    return;
  }

  if (i->isSymbolic()) {
    os << "i" << i->name();
  } else {
    os << *(i->value());
  }
}
namespace{
void check_inlineable(const IRInputOutput* const irio) {
  for (auto inp : irio->inputs())
    TORCH_CHECK(inp->getValType().value() == ValType::Scalar);
  TORCH_CHECK(irio->nOutputs() == 1)
  TORCH_CHECK(irio->output(0)->getValType().value() == ValType::Scalar);
}
}

void IRPrinter::print(const UnaryOp* const uop) {
  if (print_inline_)
    check_inlineable(uop);

  if (!print_inline_)
    os << uop->out() << " = ";
  if (auto inline_uop = inline_op_str(uop->type())) {
    os << inline_uop.value();
    print( uop->in() );
  } else {
    os << uop->type() << "(";
    print( uop->in() ); 
    os << ")";
  }
  
  if(!print_inline_)
    os<<"\n";
}

void IRPrinter::print(const BinaryOp* const bop) {
  if (print_inline_)
    check_inlineable(bop);

  if (!print_inline_)
    os << bop->out() << " = ";
  if (auto inline_bop = inline_op_str(bop->type())) {
    print( bop->lhs() );
    os << " " << inline_bop.value() << " ";
    print( bop->rhs() );
  } else {
    os << bop->type() << "(";
    print( bop->lhs() );
    os  << ", ";
    print( bop->rhs() );
    os << ")";
  }
  
  if(!print_inline_)
    os<<"\n";
}

void IRPrinter::print(const ForLoop* const fl) {
  os << "ForLoop: " << fl->index() << " index " << fl->begin() << " begin " << fl->end() << " end \n";
  for(auto &expr : fl->body()) {
    os << "\t";
    print(expr);	
  }
}
void IRPrinter::print(const IfThenElse* const ite) {
  os << "IfThenElse: if " << ite->cond() << " \n";
  for(auto &expr : ite->if_body()) {
    os << "\t";
    print(expr);	
  }
  if(ite->hasElse()) {
    os << "IfThenElse: else\n";
    for(auto &expr : ite->else_body()) {
      os << "\t";
      print(expr);	
    }
  }
}

void IRPrinter::print(const Split* const s) {
  os << "Split: ";
  print( s->in() );
  os << " axis " << s->axis() << " by factor "
     << s->factor() << " -> ";
  print( s->out() );
  os<< "\n";
}

void IRPrinter::print(const Merge* const m) {
  os << "Merge: " << m->in() << " axis " << m->axis()
     << " with the following -> ";
  print( m->out() );
  os << "\n";
}

void IRPrinter::print(const Reorder* const ro) {
  os << "Reorder: ";
  print( ro->in() );
  os << " -> ";
  print( ro->out() );
  os << "\n";
}

std::ostream& operator<< (std::ostream& os, const Statement* const stmt){
  IRPrinter p(os);
  p.print(stmt);
  return os;
}

std::ostream& operator<< (std::ostream& os, Fusion* f){
  IRPrinter p(os);
  p.print(f);
  return os;
}

std::ostream& operator<< (std::ostream& os, Fusion& f){
  return os << &f;
}

} // namespace fuser
} // namespace jit
} // namespace torch
