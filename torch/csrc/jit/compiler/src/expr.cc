#include "expr.h"

#include "ir.h"

namespace torch {
namespace jit {
namespace compiler {

Expr Expr::operator+(const Expr& other) const { return Add::make(*this, other); }

Expr Expr::operator-(const Expr& other) const { return Sub::make(*this, other); }

Expr Expr::operator*(const Expr& other) const { return Mul::make(*this, other); }

Expr Expr::operator/(const Expr& other) const { return Div::make(*this, other); }

Expr::Expr(int v) : Expr(std::move(IntImm::make(v))) {}

Expr::Expr(float v) : Expr(std::move(FloatImm::make(v))) {}

} // namespace compiler
} // namespace jit
} // namespace torch
