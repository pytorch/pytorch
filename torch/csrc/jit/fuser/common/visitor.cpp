#include <torch/csrc/jit/fuser/common/visitor.h>

#include <iostream>

namespace torch {
namespace jit {
namespace fuser {

/*
* Simple handler definitions
*/

int SimpleHandler::handle(const Statement* statement){
  return -1;
}

int SimpleHandler::handle(const Float* f){
  return 0;
}

/*
* IRPrinter definitions
*/

std::ostream& IRPrinter::printValPreamble(std::ostream& out, const Val* const v) {
  return out /*<< "%" << v->name() << ":"*/;
}

int IRPrinter::handle(const Statement* const statement){
  out_ << "Unknown statement";
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

int IRPrinter::handle(const Add* const add){
  add->out()->dispatch(this);
  out_ << " = ";
  add->lhs()->dispatch(this);
  out_ << " + ";
  add->rhs()->dispatch(this);
  return 0;
}

}}} // torch::jit::fuser
