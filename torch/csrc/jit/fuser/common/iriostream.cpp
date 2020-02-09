#include <torch/csrc/jit/fuser/common/fusion.h>
#include <torch/csrc/jit/fuser/common/ir.h>
#include <torch/csrc/jit/fuser/common/iriostream.h>

#include <iostream>

namespace torch {
namespace jit {
namespace fuser {

std::ostream& operator<<(std::ostream& os, const Fusion* const fusion) {
  std::cout<<"Fusion has "<< fusion->exprs().size() << "exprs" << std::endl;
  for (const Expr* expr : fusion->exprs()){
    os << expr << std::endl;
  }
  return os;
}

std::ostream& operator<<(std::ostream& os, const Statement* const stmt) {
  if (stmt->isVal())
    return os << static_cast<const Val*>(stmt);
  else if (stmt->isExpr())
    return os << static_cast<const Expr*>(stmt);
  throw std::runtime_error("Unkown statment type found in os << Statement.");
}

std::ostream& operator<<(std::ostream& os, const Val* const val) {
  switch (*(val->getValType())) {
    case ValType::Tensor:
      return os << static_cast<const Tensor*>(val);
    case ValType::Float:
      return os << static_cast<const Float*>(val);
    case ValType::Int:
      return os << static_cast<const Int*>(val);
  }
  throw std::runtime_error("Unknown ValType in os << Val.");
}

std::ostream& operator<<(std::ostream& os, const Expr* const expr) {
  switch (*(expr->getExprType())) {
    case ExprType::Add:
      return os << static_cast<const Add*>(expr);
  }
  throw std::runtime_error("Unknown ExprType in os << Expr.");
}

std::ostream& operator<<(std::ostream& os, const Tensor* const tensor) {
  return os << "%T" << tensor->name();
}

std::ostream& operator<<(std::ostream& os, const Float* const f) {
  os << "%f";
  if (f->isSymbolic()) {
    return os << f->name();
  } else {
    return os << f->name() << "{" << *(f->value()) << "}";
  }
}

std::ostream& operator<<(std::ostream& os, const Int* const i) {
  os << "%i";
  if (i->isSymbolic()) {
    return os << i->name();
  } else {
    return os << i->name() << "{" << *(i->value()) << "}";
  }
}

std::ostream& operator<<(std::ostream& os, const Add* const add) {
  return os << add->out() << " = " << add->lhs() << " + " << add->rhs();
}

std::ostream& operator<<(std::ostream& os, const Fusion& f) {
  return os << &f;
}

} // namespace fuser
} // namespace jit
} // namespace torch