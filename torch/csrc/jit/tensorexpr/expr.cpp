#include "torch/csrc/jit/tensorexpr/expr.h"

#include "torch/csrc/jit/tensorexpr/ir.h"

namespace torch {
namespace jit {
namespace tensorexpr {

ExprHandle ExprHandle::operator+(const ExprHandle& other) const {
  return Add::make(*this, other);
}

ExprHandle ExprHandle::operator-(const ExprHandle& other) const {
  return Sub::make(*this, other);
}

ExprHandle ExprHandle::operator*(const ExprHandle& other) const {
  return Mul::make(*this, other);
}

ExprHandle ExprHandle::operator/(const ExprHandle& other) const {
  return Div::make(*this, other);
}

ExprHandle ExprHandle::operator%(const ExprHandle& other) const {
  return Mod::make(*this, other);
}

ExprHandle ExprHandle::operator==(const ExprHandle& other) const {
  return CompareSelect::make(*this, other, CompareSelectOperation::kEQ);
}

ExprHandle ExprHandle::operator!=(const ExprHandle& other) const {
  return CompareSelect::make(*this, other, CompareSelectOperation::kNE);
}

ExprHandle ExprHandle::operator>(const ExprHandle& other) const {
  return CompareSelect::make(*this, other, CompareSelectOperation::kGT);
}

ExprHandle ExprHandle::operator>=(const ExprHandle& other) const {
  return CompareSelect::make(*this, other, CompareSelectOperation::kGE);
}

ExprHandle ExprHandle::operator<(const ExprHandle& other) const {
  return CompareSelect::make(*this, other, CompareSelectOperation::kLT);
}

ExprHandle ExprHandle::operator<=(const ExprHandle& other) const {
  return CompareSelect::make(*this, other, CompareSelectOperation::kLE);
}

ExprHandle ExprHandle::operator&(const ExprHandle& other) const {
  return And::make(*this, other);
}

ExprHandle ExprHandle::operator|(const ExprHandle& other) const {
  return Or::make(*this, other);
}

ExprHandle ExprHandle::operator^(const ExprHandle& other) const {
  return Xor::make(*this, other);
}

ExprHandle ExprHandle::operator<<(const ExprHandle& other) const {
  return Lshift::make(*this, other);
}

ExprHandle ExprHandle::operator>>(const ExprHandle& other) const {
  return Rshift::make(*this, other);
}

// NOLINTNEXTLINE
#define IMM_EXPR_DECLARE(Type, Name) \
  ExprHandle::ExprHandle(Type v) : ExprHandle(Name##Imm::make(v)) {}
AT_FORALL_SCALAR_TYPES_AND2(Bool, Half, IMM_EXPR_DECLARE);
#undef IMM_EXPR_DECLARE

ExprHandle sin(const ExprHandle& v) {
  return Intrinsics::make(kSin, v);
}

ExprHandle cos(const ExprHandle& v) {
  return Intrinsics::make(kCos, v);
}

ExprHandle tan(const ExprHandle& v) {
  return Intrinsics::make(kTan, v);
}

ExprHandle asin(const ExprHandle& v) {
  return Intrinsics::make(kAsin, v);
}

ExprHandle acos(const ExprHandle& v) {
  return Intrinsics::make(kAcos, v);
}

ExprHandle atan(const ExprHandle& v) {
  return Intrinsics::make(kAtan, v);
}

ExprHandle sinh(const ExprHandle& v) {
  return Intrinsics::make(kSinh, v);
}

ExprHandle cosh(const ExprHandle& v) {
  return Intrinsics::make(kCosh, v);
}

ExprHandle tanh(const ExprHandle& v) {
  return Intrinsics::make(kTanh, v);
}

ExprHandle exp(const ExprHandle& v) {
  return Intrinsics::make(kExp, v);
}

ExprHandle expm1(const ExprHandle& v) {
  return Intrinsics::make(kExpm1, v);
}

ExprHandle fabs(const ExprHandle& v) {
  return Intrinsics::make(kFabs, v);
}

ExprHandle log(const ExprHandle& v) {
  return Intrinsics::make(kLog, v);
}

ExprHandle log2(const ExprHandle& v) {
  return Intrinsics::make(kLog2, v);
}

ExprHandle log10(const ExprHandle& v) {
  return Intrinsics::make(kLog10, v);
}

ExprHandle log1p(const ExprHandle& v) {
  return Intrinsics::make(kLog1p, v);
}

ExprHandle erf(const ExprHandle& v) {
  return Intrinsics::make(kErf, v);
}

ExprHandle erfc(const ExprHandle& v) {
  return Intrinsics::make(kErfc, v);
}

ExprHandle sqrt(const ExprHandle& v) {
  return Intrinsics::make(kSqrt, v);
}

ExprHandle rsqrt(const ExprHandle& v) {
  return Intrinsics::make(kRsqrt, v);
}

ExprHandle ceil(const ExprHandle& v) {
  return Intrinsics::make(kCeil, v);
}

ExprHandle floor(const ExprHandle& v) {
  return Intrinsics::make(kFloor, v);
}

ExprHandle round(const ExprHandle& v) {
  return Intrinsics::make(kRound, v);
}

ExprHandle trunc(const ExprHandle& v) {
  return Intrinsics::make(kTrunc, v);
}

ExprHandle frac(const ExprHandle& v) {
  return Intrinsics::make(kFrac, v);
}

ExprHandle lgamma(const ExprHandle& v) {
  return Intrinsics::make(kLgamma, v);
}

ExprHandle atan2(const ExprHandle& v1, const ExprHandle& v2) {
  return Intrinsics::make(kAtan2, v1, v2);
}

ExprHandle pow(const ExprHandle& v1, const ExprHandle& v2) {
  return Intrinsics::make(kPow, v1, v2);
}

ExprHandle fmod(const ExprHandle& v1, const ExprHandle& v2) {
  return Intrinsics::make(kFmod, v1, v2);
}

ExprHandle remainder(const ExprHandle& v1, const ExprHandle& v2) {
  return Intrinsics::make(kRemainder, v1, v2);
}

ExprHandle ifThenElse(
    const ExprHandle& c,
    const ExprHandle& t,
    const ExprHandle& f) {
  return IfThenElse::make(c, t, f);
}

} // namespace tensorexpr
} // namespace jit
} // namespace torch
