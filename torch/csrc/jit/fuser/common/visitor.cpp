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
  return f->value();
}

/*
* IRPrinter definitions
*/

int IRPrinter::handle(const Statement* const statement){
  out_ << "Unknown statement" << std::endl;
  return 0;
}
int IRPrinter::handle(const Float* const f){
  out_ << "%" << f->name() << ":f" << f->value();
  return 0;
}
int IRPrinter::handle(const Add* const add){
  out_ << "( ";
  add->lhs()->dispatch(this);
  out_ << " + ";
  add->rhs()->dispatch(this);
  out_ <<" )"<<std::endl;
  return 0;
}

}}} // torch::jit::fuser
