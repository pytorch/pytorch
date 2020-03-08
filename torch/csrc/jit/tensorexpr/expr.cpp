#include "torch/csrc/jit/tensorexpr/expr.h"

#include "torch/csrc/jit/tensorexpr/ir.h"

namespace torch {
namespace jit {
namespace tensorexpr {

Expr Expr::operator+(const Expr& other) const {
  return Add::make(*this, other);
}

Expr Expr::operator-(const Expr& other) const {
  return Sub::make(*this, other);
}

Expr Expr::operator*(const Expr& other) const {
  return Mul::make(*this, other);
}

Expr Expr::operator/(const Expr& other) const {
  return Div::make(*this, other);
}

Expr Expr::operator==(const Expr& other) const {
  return CompareSelect::make(*this, other, CompareSelectOperation::kEQ);
}

Expr Expr::operator!=(const Expr& other) const {
  return CompareSelect::make(*this, other, CompareSelectOperation::kNE);
}

Expr Expr::operator>(const Expr& other) const {
  return CompareSelect::make(*this, other, CompareSelectOperation::kGT);
}

Expr Expr::operator>=(const Expr& other) const {
  return CompareSelect::make(*this, other, CompareSelectOperation::kGE);
}

Expr Expr::operator<(const Expr& other) const {
  return CompareSelect::make(*this, other, CompareSelectOperation::kLT);
}

Expr Expr::operator<=(const Expr& other) const {
  return CompareSelect::make(*this, other, CompareSelectOperation::kLE);
}

Expr::Expr(int v) : Expr(IntImm::make(v)) {}

Expr::Expr(float v) : Expr(FloatImm::make(v)) {}

Expr ifThenElse(const Expr& c, const Expr& t, const Expr& f) {
  return IfThenElse::make(c, t, f);
}

} // namespace tensorexpr
} // namespace jit
} // namespace torch
