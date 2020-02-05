#include <torch/csrc/jit/fuser/common/ir.h>
#include <torch/csrc/jit/fuser/common/visitor.h>

#include <iostream>

namespace torch {
namespace jit {
namespace fuser {

/*
* Simple handler definitions
*/

int SimpleHandler::handle(const Statement* statement){
    throw std::runtime_error("Could not identify statement. Did you update dispatch in ir.cpp?");
}

int SimpleHandler::handle(const Float* f){
  return 0;
}

int SimpleHandler::handle(const Int* i){
  return 0;
}

/*
* IRPrinter definitions
*/

std::ostream& IRPrinter::printValPreamble(std::ostream& out, const Val* const v) {
  return out /*<< "%" << v->name() << ":"*/;
}

int IRPrinter::handle(const Statement* const statement){
  statement->dispatch(this);
  return 0;
}

int IRPrinter::handle(const Float* const f){
  printValPreamble(out_, f) << "f";
  if (f->isSymbolic()) {
    out_ << "?";
  } else {
    out_ << *(f->value());
  }
  return 0;
}

int IRPrinter::handle(const Tensor* const t){
  out_<<"T";
  return 0;
}

int IRPrinter::handle(const Int* const i){
  printValPreamble(out_, i) << "i";
  if (i->isSymbolic()) {
    out_ << "?";
  } else {
    out_ << *(i->value());
  }
  return 0;
}

int IRPrinter::handle(const Add* const add){
  add->out()->dispatch(this);
  out_ << " = ";
  add->lhs()->dispatch(this);
  out_ << " + ";
  add->rhs()->dispatch(this);
  return 0;
}

}}} // torch::jit::fuser
