#include <torch/csrc/jit/fuser/common/ir.h>

#include <torch/csrc/jit/fuser/common/visitor.h>

#include <iostream>
#include <unordered_map>
#include <string>
#include <stdexcept>

namespace torch {
namespace jit {
namespace fuser {

/*
* Statement member definitions
*/

Statement::~Statement() { }

// Note: when adding a new val or expr a case must be added here
template <typename T>
int Statement::dispatch(T handler) {
  const auto maybe_val_type = getValType();
  if (maybe_val_type) {
    switch (*maybe_val_type) {
      case ValType::Float:
        return handler->handle(static_cast<Float*>(this));
      default:
        throw std::runtime_error("Unknown valtype in dispatch!");
    }
  }

  switch (*getExprType()) {
    case ExprType::Add:
      return handler->handle(static_cast<Add*>(this));
    default:
      throw std::runtime_error("Unknown exprtype in dispatch!");
  }
}

// Handler template instantiations
template int Statement::dispatch(SimpleHandler*);
template int Statement::dispatch(IRPrinter*);

/*
* Val member definitions
*/

Val::~Val() { }

}}} // torch::jit::fuser
