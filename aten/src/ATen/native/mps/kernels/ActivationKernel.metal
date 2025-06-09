#include <c10/metal/indexing.h>
#include <c10/metal/special_math.h>
#include <metal_stdlib>
using namespace metal;
using namespace c10::metal;

struct hardshrink_functor {
  template <typename T>
  inline T operator()(const T x, const float lambda) {
    return (x > lambda || x < -lambda) ? x : T(0);
  }
};

struct hardshrink_backward_functor {
  template <typename T>
  inline T operator()(const T grad_output, const T x, const float lambda) {
    return (x > lambda || x < -lambda) ? grad_output : T(0);
  }
};

REGISTER_UNARY_ALPHA_OP(hardshrink, float, float, float);
REGISTER_UNARY_ALPHA_OP(hardshrink, half, float, half);
#if __METAL_VERSION__ >= 310
REGISTER_UNARY_ALPHA_OP(hardshrink, bfloat, float, bfloat);
#endif

REGISTER_BINARY_ALPHA_OP(hardshrink_backward, float, float, float);
REGISTER_BINARY_ALPHA_OP(hardshrink_backward, half, float, half);
#if __METAL_VERSION__ >= 310
REGISTER_BINARY_ALPHA_OP(hardshrink_backward, bfloat, float, bfloat);

struct hardsigmoid_functor {
  template <typename T>
  inline T operator()(const T x) {
    T zero(0);
    T three(3);
    T six(6);
    T result = min(max(x + three, zero), six) / six;
    return result;
  }
};

struct hardsigmoid_backward_functor {
  template <typename T>
  inline T operator()(const T grad_output, const T self) {
    T zero(0);
    T one_sixth(1.0 / 6.0);
    T neg_three(-3);
    T three(3);

    if (self < neg_three || self > three) {
      return zero;
    } else {
      return grad_output * one_sixth;
    }
  }
};

REGISTER_UNARY_OP(hardsigmoid, float, float);
REGISTER_UNARY_OP(hardsigmoid, half, half);
#if __METAL_VERSION__ >= 310
REGISTER_UNARY_OP(hardsigmoid, bfloat, bfloat);
#endif

REGISTER_BINARY_OP(hardsigmoid_backward, float, float, float);
REGISTER_BINARY_OP(hardsigmoid_backward, half, half, half);
#if __METAL_VERSION__ >= 310
REGISTER_BINARY_OP(hardsigmoid_backward, bfloat, bfloat, bfloat);
#endif
