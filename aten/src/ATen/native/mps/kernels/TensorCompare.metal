#include <c10/metal/indexing.h>
#include <c10/metal/special_math.h>
#include <c10/metal/utils.h>
#include <metal_stdlib>
using namespace metal;

struct clamp_functor {
  template <typename T>
  inline T operator()(const T a, const T b_min, const T c_max) {
    return min(max(a, b_min), c_max);
  }
};

#define REGISTER_CLAMP_OP(DTYPE) REGISTER_TERNARY_OP(clamp, DTYPE, DTYPE);

REGISTER_CLAMP_OP(long);
REGISTER_CLAMP_OP(int);
REGISTER_CLAMP_OP(short);
REGISTER_CLAMP_OP(uchar);
REGISTER_CLAMP_OP(char);
REGISTER_CLAMP_OP(bool);

REGISTER_CLAMP_OP(float);
REGISTER_CLAMP_OP(half);
REGISTER_CLAMP_OP(bfloat);
