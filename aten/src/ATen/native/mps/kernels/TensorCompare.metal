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

struct where_functor {
  template <typename T>
  inline T operator()(const T cond, const T a, const T b) {
    return c10::metal::cast_to<bool>(cond) ? a : b;
  }
};

#define REGISTER_WHERE_OP(T)                                      \
  template [[host_name("where_dense_" #T "_bool")]] kernel void   \
  c10::metal::ternary_dense<T, where_functor, T, bool>(           \
      device T*, constant bool*, constant T*, constant T*, uint); \
  template [[host_name("where_strided_" #T "_bool")]] kernel void \
  c10::metal::ternary_strided<T, where_functor, T, bool>(         \
      device void*,                                               \
      constant void*,                                             \
      constant void*,                                             \
      constant void*,                                             \
      constant long*,                                             \
      constant long*,                                             \
      constant long*,                                             \
      constant long*,                                             \
      constant long*,                                             \
      constant uint&,                                             \
      uint3)

REGISTER_WHERE_OP(bool);
REGISTER_WHERE_OP(uchar);
REGISTER_WHERE_OP(char);
REGISTER_WHERE_OP(short);
REGISTER_WHERE_OP(ushort);
REGISTER_WHERE_OP(int);
REGISTER_WHERE_OP(uint);
REGISTER_WHERE_OP(long);
REGISTER_WHERE_OP(ulong);
REGISTER_WHERE_OP(half);
REGISTER_WHERE_OP(bfloat);
REGISTER_WHERE_OP(float);
REGISTER_WHERE_OP(half2);
REGISTER_WHERE_OP(float2);

struct isposinf_functor {
  template <typename T>
  inline bool operator()(const T x) {
    return x == ::metal::numeric_limits<T>::infinity();
  }
};

struct isneginf_functor {
  template <typename T>
  inline bool operator()(const T x) {
    return x == -::metal::numeric_limits<T>::infinity();
  }
};

#define REGISTER_ISINF_OPS(T)           \
  REGISTER_UNARY_OP(isposinf, T, bool); \
  REGISTER_UNARY_OP(isneginf, T, bool);

REGISTER_ISINF_OPS(float);
REGISTER_ISINF_OPS(half);
REGISTER_ISINF_OPS(bfloat);
