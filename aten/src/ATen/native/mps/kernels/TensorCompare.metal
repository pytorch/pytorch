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

#define REGISTER_ALL_CLAMP_OPS(T) REGISTER_TERNARY_OP(clamp, T, T);

REGISTER_ALL_CLAMP_OPS(long);
REGISTER_ALL_CLAMP_OPS(int);
REGISTER_ALL_CLAMP_OPS(short);
REGISTER_ALL_CLAMP_OPS(uchar);
REGISTER_ALL_CLAMP_OPS(char);
REGISTER_ALL_CLAMP_OPS(bool);

REGISTER_ALL_CLAMP_OPS(float);
REGISTER_ALL_CLAMP_OPS(half);
REGISTER_ALL_CLAMP_OPS(bfloat);
