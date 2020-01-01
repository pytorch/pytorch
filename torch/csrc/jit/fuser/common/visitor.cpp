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

int IRPrinter::handle(const Statement* statement){
  std::cout << "Unknown statement" << std::endl;
  return 0;
}
int IRPrinter::handle(const Float* f){
  std::cout << "f" << f->value();
  return 0;
}
int IRPrinter::handle(const Add* add){
  std::cout << "( ";
  add->lhs()->dispatch(this);
  std::cout << " + ";
  add->rhs()->dispatch(this);
  std::cout<<" )"<<std::endl;
  return 0;
}

}}} // torch::jit::fuser
