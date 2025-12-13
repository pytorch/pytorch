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
