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

// The default tanh is quite slow, use the Eigen version from here:
// https://bitbucket.org/eigen/eigen/src/94875feeeeb9abe5509b314197da1991ba2070f5/Eigen/src/Core/MathFunctionsImpl.h#lines-26
ExprHandle fast_tanh(const ExprHandle& v) {
  Dtype dtype = v.dtype();
  // TODO: use a dedicated bind-var to make sure v is not evalualted multiple
  // times. Clamp the input expression to [-9, 9]
  ExprHandle plus_9 = FloatImm::make(9.0f);
  ExprHandle minus_9 = FloatImm::make(-9.0f);
  ExprHandle v1 = Min::make(v, plus_9, false);
  v1 = Max::make(v1, minus_9, false);

  // The coefficients for the numerator
  ExprHandle alpha_1 = FloatImm::make(4.89352455891786e-03f);
  ExprHandle alpha_3 = FloatImm::make(6.37261928875436e-04f);
  ExprHandle alpha_5 = FloatImm::make(1.48572235717979e-05f);
  ExprHandle alpha_7 = FloatImm::make(5.12229709037114e-08f);
  ExprHandle alpha_9 = FloatImm::make(-8.60467152213735e-11f);
  ExprHandle alpha_11 = FloatImm::make(2.00018790482477e-13f);
  ExprHandle alpha_13 = FloatImm::make(-2.76076847742355e-16f);

  // The coeffecients for the denominator
  ExprHandle beta_0 = FloatImm::make(4.89352518554385e-03f);
  ExprHandle beta_2 = FloatImm::make(2.26843463243900e-03f);
  ExprHandle beta_4 = FloatImm::make(1.18534705686654e-04f);
  ExprHandle beta_6 = FloatImm::make(1.19825839466702e-06f);

  // numerator
  ExprHandle v2 = v1 * v1;
  ExprHandle p = v2 * alpha_13 + alpha_11;
  p = v2 * p + alpha_9;
  p = v2 * p + alpha_7;
  p = v2 * p + alpha_5;
  p = v2 * p + alpha_3;
  p = v2 * p + alpha_1;
  p = v1 * p;

  // denominator
  ExprHandle q = v2 * beta_6 + beta_4;
  q = v2 * q + beta_2;
  q = v2 * q + beta_0;

  ExprHandle result = p / q;
  return result;
}

ExprHandle fast_sigmoid(const ExprHandle& x) {
  // sigmoid(x) = (tanh(x / 2) + 1) / 2
  ExprHandle one_v = FloatImm::make(1.f);
  ExprHandle half_v = FloatImm::make(0.5f);
  ExprHandle x2 = x * half_v;
  ExprHandle y{fast_tanh(x2)};
  ExprHandle z = (y + one_v) * half_v;
  return z;
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

  auto zero = FloatImm::make(0);
  auto nan = FloatImm::make(std::numeric_limits<float>::quiet_NaN());
  auto neg_inf = FloatImm::make(-std::numeric_limits<float>::infinity());
  x = CompareSelect::make(v, zero, nan, x, kLT);
  x = CompareSelect::make(v, zero, neg_inf, x, kEQ);
  return x;
}

ExprHandle log_vml(const ExprHandle& v) {
  auto mlaf = [](ExprHandle x, ExprHandle y, float z) {
    return x * y + FloatImm::make(z);
  };

  auto in = bitcast<int32_t>(v);
  auto a = in - IntImm::make(0x3f2aaaab);
  auto e = cast<float>(a >> IntImm::make(23));

  auto x = (a & IntImm::make(0x7fffff)) + IntImm::make(0x3f2aaaab);
  x = bitcast<float>(x) - 1.0f;

  auto t = FloatImm::make(-0.12891686f);
  t = mlaf(x, t, 0.139844373f);
  t = mlaf(x, t, -0.121842608f);
  t = mlaf(x, t, 0.140058696f);
  t = mlaf(x, t, -0.16680488f);
  t = mlaf(x, t, 0.200104058f);
  t = mlaf(x, t, -0.249997973f);
  t = mlaf(x, t, 0.333332151f);
  t = mlaf(x, t, -0.5f);
  t = x * t;
  t = x * t + x;

  auto z = e * FloatImm::make(1.42860677e-06f) + t;
  z = e * FloatImm::make(0.693145752f) + z;

  return CompareSelect::make(
      IntImm::make(0x1000000),
      in + IntImm::make(0x800000),
      log(v),
      z,
      kGT,
      kUnlikely);
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
