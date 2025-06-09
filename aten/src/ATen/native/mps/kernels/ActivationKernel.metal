#include <c10/metal/indexing.h>
#include <c10/metal/special_math.h>
#include <metal_stdlib>
using namespace metal;
using namespace c10::metal;

struct hardshrink_functor {
  template <typename T>
  inline T operator()(const T x, const T lambda) {
    return (x >= -lambda && x <= lambda) ? T(0) : x;
  }
};

struct hardshrink_backward_functor {
  template <typename T>
  inline T operator()(const T grad_output, const T x, const T lambda) {
    return (x >= -lambda && x <= lambda) ? T(0) : grad_output;
  }
};

REGISTER_UNARY_ALPHA_OP(hardshrink, float, float, float);
REGISTER_UNARY_ALPHA_OP(hardshrink, half, half, half);
#if __METAL_VERSION__ >= 310
REGISTER_UNARY_ALPHA_OP(hardshrink, bfloat, bfloat, bfloat);
#endif

REGISTER_BINARY_ALPHA_OP(hardshrink_backward, float, float, float);
REGISTER_BINARY_ALPHA_OP(hardshrink_backward, half, half, half);
#if __METAL_VERSION__ >= 310
REGISTER_BINARY_ALPHA_OP(hardshrink_backward, bfloat, bfloat, bfloat);
#endif
