#include <ATen/native/mps/kernels/TensorCompare.h>
#include <c10/metal/indexing.h>
#include <c10/metal/special_math.h>
#include <c10/metal/utils.h>
#include <metal_stdlib>
using namespace metal;

struct clamp_functor {
  template <typename T>
  inline T operator()(const T a, const T b_min, const T c_max) {
    return c10::metal::min(c10::metal::max(a, b_min), c_max);
  }
};

// The condition reaches the functor converted to T by the ternary cast
// machinery (bool/byte sources become exact 0/1), so truthiness is a
// zero-compare; complex needs the component form (vector != is not scalar).
struct where_functor {
  template <typename T>
  inline T operator()(const T cond, const T a, const T b) {
    return cond != T(0) ? a : b;
  }
  inline float2 operator()(const float2 cond, const float2 a, const float2 b) {
    return (cond.x != 0.0f || cond.y != 0.0f) ? a : b;
  }
  inline half2 operator()(const half2 cond, const half2 a, const half2 b) {
    return (cond.x != 0.0h || cond.y != 0.0h) ? a : b;
  }
};

struct clamp_scalar_functor {
  template <typename T>
  inline T operator()(const T a, const ClampScalarParams<T> params) {
    return c10::metal::min(c10::metal::max(a, params.min), params.max);
  }
};

struct clamp_min_scalar_functor {
  template <typename T>
  inline T operator()(const T a, const T b_min) {
    return c10::metal::max(a, b_min);
  }
};

struct clamp_max_scalar_functor {
  template <typename T>
  inline T operator()(const T a, const T b_max) {
    return c10::metal::min(a, b_max);
  }
};

#define REGISTER_CLAMP_SCALAR_OP(T)                   \
  typedef ClampScalarParams<T> ClampScalarParams_##T; \
  REGISTER_UNARY_ALPHA_OP(clamp_scalar, T, ClampScalarParams_##T, T);

#define REGISTER_ALL_CLAMP_OPS(T)                     \
  REGISTER_TERNARY_OP(clamp, T, T);                   \
  REGISTER_CLAMP_SCALAR_OP(T);                        \
  REGISTER_UNARY_ALPHA_OP(clamp_min_scalar, T, T, T); \
  REGISTER_UNARY_ALPHA_OP(clamp_max_scalar, T, T, T);

REGISTER_ALL_CLAMP_OPS(long);
REGISTER_ALL_CLAMP_OPS(int);
REGISTER_ALL_CLAMP_OPS(short);
REGISTER_ALL_CLAMP_OPS(uchar);
REGISTER_ALL_CLAMP_OPS(char);
REGISTER_ALL_CLAMP_OPS(bool);

REGISTER_ALL_CLAMP_OPS(float);
REGISTER_ALL_CLAMP_OPS(half);
REGISTER_ALL_CLAMP_OPS(bfloat);

REGISTER_TERNARY_OP(where, float, float);
REGISTER_TERNARY_OP(where, half, half);
REGISTER_TERNARY_OP(where, bfloat, bfloat);
REGISTER_TERNARY_OP(where, long, long);
REGISTER_TERNARY_OP(where, int, int);
REGISTER_TERNARY_OP(where, short, short);
REGISTER_TERNARY_OP(where, char, char);
REGISTER_TERNARY_OP(where, uchar, uchar);
REGISTER_TERNARY_OP(where, ushort, ushort);
REGISTER_TERNARY_OP(where, uint, uint);
REGISTER_TERNARY_OP(where, ulong, ulong);
REGISTER_TERNARY_OP(where, bool, bool);
REGISTER_TERNARY_OP(where, float2, float2);
REGISTER_TERNARY_OP(where, half2, half2);
