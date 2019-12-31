#include <torch/csrc/jit/fuser/common/ir.h>
#include <torch/csrc/jit/fuser/common/visitor.h>
#include <torch/csrc/jit/fuser/common/fusion.h>

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

Val::Val(
  const ValType _type,
  Fusion& fusion)
  : type_{_type}
  {
    fusion.addVal(this);
  }

Expr::Expr(
    const ExprType _type,
    Fusion& fusion)
  : type_{_type}
  {
    fusion.addExpr(this);
   }

// Note: when adding a new val or expr a case must be added here
template <typename T>
int Statement::dispatch(T* handler) const{
  const auto maybe_val_type = getValType();
  if (maybe_val_type) {
    switch (*maybe_val_type) {
      case ValType::Float:
        return handler->handle(static_cast<const Float*>(this));
      default:
        throw std::runtime_error("Unknown valtype in dispatch!");
    }
  }

  switch (*getExprType()) {
    case ExprType::Add:
      return handler->handle(static_cast<const Add*>(this));
    default:
      throw std::runtime_error("Unknown exprtype in dispatch!");
  }
}

// Handler template instantiations
template int Statement::dispatch(SimpleHandler*) const;
template int Statement::dispatch(IRPrinter*) const;

/*
* Val member definitions
*/

Val::~Val() { }

}}} // torch::jit::fuser
