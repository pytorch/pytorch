#include <c10/metal/indexing.h>
#include <c10/metal/utils.h>
#include <metal_stdlib>
using namespace metal;

// Both functors mirror the CPU kernels: the products are evaluated at opmath
// precision and in the same order, i.e. `value` is folded into tensor1 first.
// `fma` is deliberately not used, as CPU does not use it either.
struct addcmul_functor {
  template <typename T, typename A>
  inline T operator()(const T a, const T b, const T c, const A alpha) {
    using op_t = c10::metal::opmath_t<T>;
    const auto scaled = c10::metal::mul(static_cast<op_t>(alpha), op_t(b));
    return static_cast<T>(op_t(a) + c10::metal::mul(scaled, op_t(c)));
  }
};

struct addcdiv_functor {
  template <typename T, typename A>
  inline T operator()(const T a, const T b, const T c, const A alpha) {
    using op_t = c10::metal::opmath_t<T>;
    const auto scaled = c10::metal::mul(static_cast<op_t>(alpha), op_t(b));
    return static_cast<T>(op_t(a) + c10::metal::div(scaled, op_t(c)));
  }
};

REGISTER_OPMATH_TERNARY_ALPHA_OP(addcmul, float, float, float);
REGISTER_OPMATH_TERNARY_ALPHA_OP(addcmul, half, float, half);
REGISTER_OPMATH_TERNARY_ALPHA_OP(addcmul, bfloat, float, bfloat);
REGISTER_OPMATH_TERNARY_ALPHA_OP(addcmul, long, long, long);
REGISTER_OPMATH_TERNARY_ALPHA_OP(addcmul, int, int, int);
REGISTER_OPMATH_TERNARY_ALPHA_OP(addcmul, short, short, short);
REGISTER_OPMATH_TERNARY_ALPHA_OP(addcmul, char, char, char);
REGISTER_OPMATH_TERNARY_ALPHA_OP(addcmul, uchar, uchar, uchar);
REGISTER_OPMATH_TERNARY_ALPHA_OP(addcmul, float2, float2, float2);
REGISTER_OPMATH_TERNARY_ALPHA_OP(addcmul, half2, float2, half2);

// addcdiv rejects integral tensor1/tensor2 in its meta function, so its common
// dtype is always floating point or complex.
REGISTER_OPMATH_TERNARY_ALPHA_OP(addcdiv, float, float, float);
REGISTER_OPMATH_TERNARY_ALPHA_OP(addcdiv, half, float, half);
REGISTER_OPMATH_TERNARY_ALPHA_OP(addcdiv, bfloat, float, bfloat);
REGISTER_OPMATH_TERNARY_ALPHA_OP(addcdiv, float2, float2, float2);
REGISTER_OPMATH_TERNARY_ALPHA_OP(addcdiv, half2, float2, half2);
