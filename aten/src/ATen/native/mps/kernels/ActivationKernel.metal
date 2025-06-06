#include <c10/metal/indexing.h>
#include <c10/metal/special_math.h>
#include <metal_stdlib>
using namespace metal;
using namespace c10::metal;

struct hardshrink_functor {
  template <typename T>
  inline T operator()(const T x, float lambda) {
    return (x > lambda || x < -lambda) ? x : T(0);
  }
};

REGISTER_UNARY_ALPHA_OP(hardshrink, float, float, float);
REGISTER_UNARY_ALPHA_OP(hardshrink, half, float, half);
#if __METAL_VERSION__ >= 310
REGISTER_UNARY_ALPHA_OP(hardshrink, bfloat, float, bfloat);
#endif
