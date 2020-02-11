#include <torch/csrc/jit/fuser/common/fusion.h>
#include <torch/csrc/jit/fuser/common/ir.h>
#include <torch/csrc/jit/fuser/common/tensor.h>
#include <torch/csrc/jit/fuser/common/iriostream.h>

#include <iostream>

namespace torch {
namespace jit {
namespace fuser {

namespace {

template<typename T>
std::ostream& operator<<(std::ostream& os, const c10::optional<std::vector<T>>& data) {
  os << "(";
  if (data.has_value()) {
    for (auto i = data.value().begin(); i != data.value().end(); i++) {
      os << (*i);
      os << " ";
    }
  } else {
    os << "?";
  }
  return os << ")";
}

}

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
      return os << static_cast<const Tensor* const>(val);

    case ValType::Scalar:
      switch (*(val->getDataType())){
        case DataType::Float:
          return os << static_cast<const Float* const>(val);
        case DataType::Int:
          return os << static_cast<const Int* const>(val);
        default:
          break;
      }

    default:
      break;
  }
  throw std::runtime_error("Unknown ValType in os << Val.");
}

std::ostream& operator<<(std::ostream& os, const Expr* const expr) {
  switch (*(expr->getExprType())) {
    case ExprType::UnaryOp:
      return os << static_cast<const UnaryOp*>(expr);
    case ExprType::BinaryOp:
      return os << static_cast<const BinaryOp*>(expr);
  }
  throw std::runtime_error("Unknown ExprType in os << Expr.");
}

/*
std::ostream& operator<<(std::ostream& os, const Tensor* const tensor) {
  return os << "%T" << tensor->name() <<
      " type: " << tensor->scalarType().value() <<
      ", sizes: " << tensor->sizes() <<
      ", strides: " << tensor->strides();
}
*/

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

std::ostream& operator<<(std::ostream& os, const UnaryOp* const uop) {
  return os << uop->out() << " = " << uop->type() << "(" << uop->in() << ")";
}

std::ostream& operator<<(std::ostream& os, const BinaryOp* const bop) {
  return os << bop->out() << " = " << bop->type() << "(" << bop->lhs() << ", " << bop->rhs() << ")";
}

std::ostream& operator<<(std::ostream& os, const Fusion& f) {
  return os << &f;
}

} // namespace fuser
} // namespace jit
} // namespace torch
