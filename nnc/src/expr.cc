#include "expr.h"

#include "ir.h"

namespace nnc {

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

} // namespace nnc
