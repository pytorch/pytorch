#include <c10/metal/indexing.h>
#include <c10/metal/special_math.h>
#include <metal_stdlib>
using namespace metal;
using namespace c10::metal;

struct hardshrink_functor {
  template <typename T>
  inline T operator()(const T x, const T lambda) {
    return abs(float(x)) <= float(lambda) ? T(0) : x;
  }
};

struct softshrink_functor {
  template <typename T>
  inline T operator()(const T x, const T lambda) {
    if (x > lambda) {
      return x - lambda;
    } else if (x < -lambda) {
      return x + lambda;
    } else {
      return T(0);
    }
  }
};

struct shrink_backward_functor {
  template <typename T>
  inline T operator()(const T grad_output, const T x, const T lambda) {
    return abs(float(x)) <= float(lambda) ? T(0) : grad_output;
  }
};

REGISTER_UNARY_ALPHA_OP(hardshrink, float, float, float);
REGISTER_UNARY_ALPHA_OP(hardshrink, half, half, half);
#if __METAL_VERSION__ >= 310
REGISTER_UNARY_ALPHA_OP(hardshrink, bfloat, bfloat, bfloat);
#endif

REGISTER_UNARY_ALPHA_OP(softshrink, float, float, float);
REGISTER_UNARY_ALPHA_OP(softshrink, half, half, half);
#if __METAL_VERSION__ >= 310
REGISTER_UNARY_ALPHA_OP(softshrink, bfloat, bfloat, bfloat);
#endif

REGISTER_BINARY_ALPHA_OP(shrink_backward, float, float, float);
REGISTER_BINARY_ALPHA_OP(shrink_backward, half, half, half);
#if __METAL_VERSION__ >= 310
REGISTER_BINARY_ALPHA_OP(shrink_backward, bfloat, bfloat, bfloat);
#endif

struct hardsigmoid_functor {
  template <typename T>
  inline T operator()(const T x) {
    return static_cast<T>(min(max(x + 3.0f, .0f), 6.f) / 6.f);
  }
};

struct hardsigmoid_backward_functor {
  template <typename T>
  inline T operator()(const T grad_output, const T self) {
    constexpr auto one_sixth = 1.0f / 6.0f;
    return static_cast<T>(
        abs(float(self)) < 3.0f ? float(grad_output) * one_sixth : 0.0f);
  }
};

REGISTER_UNARY_OP(hardsigmoid, float, float);
REGISTER_UNARY_OP(hardsigmoid, half, half);
#if __METAL_VERSION__ >= 310
REGISTER_UNARY_OP(hardsigmoid, bfloat, bfloat);
#endif

REGISTER_BINARY_OP(hardsigmoid_backward, float, float);
REGISTER_BINARY_OP(hardsigmoid_backward, half, half);
#if __METAL_VERSION__ >= 310
REGISTER_BINARY_OP(hardsigmoid_backward, bfloat, bfloat);
#endif

struct hardswish_functor {
  template <typename T>
  inline T operator()(const T x) {
    return static_cast<T>(float(x) * min(max(float(x) + 3.0f, .0f), 6.f) / 6.f);
  }
};

struct hardswish_backward_functor {
  template <typename T>
  inline T operator()(const T grad_output, const T self) {
    constexpr T zero(0);
    constexpr T three(3);
    constexpr T neg_three(-3);

    if (self <= neg_three) {
      return zero;
    } else if (self >= three) {
      return grad_output;
    } else {
      return static_cast<T>(float(grad_output) * (float(self) / 3.0f + 0.5f));
    }
  }
};

REGISTER_UNARY_OP(hardswish, float, float);
REGISTER_UNARY_OP(hardswish, half, half);
#if __METAL_VERSION__ >= 310
REGISTER_UNARY_OP(hardswish, bfloat, bfloat);
#endif

REGISTER_BINARY_OP(hardswish_backward, float, float);
REGISTER_BINARY_OP(hardswish_backward, half, half);
#if __METAL_VERSION__ >= 310
REGISTER_BINARY_OP(hardswish_backward, bfloat, bfloat);
#endif

struct leaky_relu_functor {
  template <typename T>
  inline T operator()(const T x, const T negative_slope) {
    return float(x) > 0.0f ? x
                           : static_cast<T>(float(x) * float(negative_slope));
  }
};

struct leaky_relu_backward_functor {
  template <typename T>
  inline T operator()(
      const T self,
      const T grad_output,
      const T negative_slope) {
    return float(self) > 0.0f
        ? grad_output
        : static_cast<T>(float(grad_output) * float(negative_slope));
  }
};

REGISTER_UNARY_ALPHA_OP(leaky_relu, float, float, float);
REGISTER_UNARY_ALPHA_OP(leaky_relu, half, half, half);
#if __METAL_VERSION__ >= 310
REGISTER_UNARY_ALPHA_OP(leaky_relu, bfloat, bfloat, bfloat);
#endif

REGISTER_BINARY_ALPHA_OP(leaky_relu_backward, float, float, float);
REGISTER_BINARY_ALPHA_OP(leaky_relu_backward, half, half, half);
#if __METAL_VERSION__ >= 310
REGISTER_BINARY_ALPHA_OP(leaky_relu_backward, bfloat, bfloat, bfloat);
#endif
