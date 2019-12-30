#include <torch/csrc/jit/fuser/common/expr.h>
#include <torch/csrc/jit/fuser/common/ir.h>

namespace torch {
namespace jit {
namespace fuser {

Expr Expr::operator+(const Expr& other) const { return Add::make(*this, other); }

Expr Expr::operator-(const Expr& other) const { return Sub::make(*this, other); }

Expr Expr::operator*(const Expr& other) const { return Mul::make(*this, other); }

Expr Expr::operator/(const Expr& other) const { return Div::make(*this, other); }

Expr::Expr(int v) : Expr(IntImm::make(v)) {}

Expr::Expr(float v) : Expr(FloatImm::make(v)) {}

} // namespace fuser
} // namespace jit
} // namespace torch
