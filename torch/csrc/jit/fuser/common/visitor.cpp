#include <torch/csrc/jit/fuser/common/visitor.h>

#include <iostream>

namespace torch {
namespace jit {
namespace fuser {

/*
* Simple handler definitions
*/

int SimpleHandler::handle(Statement* statement) {
  return -1;
}

int SimpleHandler::handle(Float* f) {
  return f->value();
}

/*
* IRPrinter definitions
*/

int IRPrinter::handle(Statement* statement) {
  std::cout << "Unknown statement" << std::endl;
  return 0;
}
int IRPrinter::handle(Float* f) {
  std::cout << "f" << f->value();
  return 0;
}
int IRPrinter::handle(Add* add) {
  std::cout << "Add" << std::endl;
  return 0;
}

}}} // torch::jit::fuser
