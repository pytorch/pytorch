#include <torch/csrc/jit/tensorexpr/expr.h>

#include <torch/csrc/jit/tensorexpr/ir.h>

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

ExprHandle sigmoid(const ExprHandle& v) {
  return Intrinsics::make(kSigmoid, v);
}

ExprHandle exp(const ExprHandle& v) {
  return Intrinsics::make(kExp, v);
}

ExprHandle expm1(const ExprHandle& v) {
  return Intrinsics::make(kExpm1, v);
}

ExprHandle abs(const ExprHandle& v) {
  return Intrinsics::make(kAbs, v);
}

ExprHandle fast_log(const ExprHandle& v) {
  // this implementation is taken from sleef:
  // https://github.com/shibatch/sleef/blob/master/src/libm/sleefsp.c#L1131
  // to generate coefficients, this tool is provided
  // https://github.com/shibatch/sleef/blob/master/src/gencoef/gencoef.txt
  auto ilogb2kf = [](ExprHandle x) {
    auto y = (bitcast<int32_t>(x) >> IntImm::make(23)) & IntImm::make(0xff);
    return y - IntImm::make(0x7f);
  };

  auto ldexp3kf = [](ExprHandle x, ExprHandle e) {
    return bitcast<float>(bitcast<int32_t>(x) + (e << IntImm::make(23)));
  };
  auto e = ilogb2kf(v * FloatImm::make(1.0 / 0.75));
  auto m = ldexp3kf(v, IntImm::make(-1) * e);
  auto one = FloatImm::make(1.0f);
  auto x = (m - one) / (m + one);
  auto x2 = x * x;

  auto mlaf = [](ExprHandle x, ExprHandle y, float z) {
    return x * y + FloatImm::make(z);
  };

  auto t = FloatImm::make(0.2392828464508056640625);
  t = mlaf(t, x2, 0.28518211841583251953125);
  t = mlaf(t, x2, 0.400005877017974853515625);
  t = mlaf(t, x2, 0.666666686534881591796875);
  t = mlaf(t, x2, 2.0);
  x = x * t + FloatImm::make(0.693147180559945286226764) * e;
  x = IfThenElse::make(
      v < FloatImm::make(0),
      FloatImm::make(std::numeric_limits<float>::quiet_NaN()),
      x);
  x = IfThenElse::make(
      v == FloatImm::make(0),
      FloatImm::make(-std::numeric_limits<float>::infinity()),
      x);
  return x;
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

ExprHandle isnan(const ExprHandle& v1) {
  return Intrinsics::make(kIsNan, v1);
}

ExprHandle ifThenElse(
    const ExprHandle& c,
    const ExprHandle& t,
    const ExprHandle& f) {
  return IfThenElse::make(c, t, f);
}

ExprHandle Buf::make(
    const std::string& name_hint,
    const std::vector<ExprHandle>& dims,
    Dtype dtype) {
  return ExprHandle(
      new Buf(name_hint, ExprHandleVectorToExprVector(dims), dtype));
}

ExprHandle Buf::make(const std::vector<ExprHandle>& dims, Dtype dtype) {
  return Buf::make("", dims, dtype);
}

ExprHandle expr_to_vec(ExprHandle v, int lanes) {
  if (lanes == 1) {
    return v;
  } else {
    return Broadcast::make(v, lanes);
  }
}

} // namespace tensorexpr
} // namespace jit
} // namespace torch
