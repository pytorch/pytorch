#include <c10/metal/indexing.h>
#include <c10/metal/utils.h>
#include <metal_stdlib>
using namespace metal;
using namespace c10::metal;

// Templated on output dtype Out. Input dtype is runtime-dispatched on load via
// val_at_offs<Out>(ptr, offs, ScalarType), which performs the cross-dtype cast
// (including real<->complex packing) before the functor sees the value. One
// kernel per Out covers every input dtype the runtime switch recognises.

struct copy_identity_functor {
  template <typename T>
  inline T operator()(const T x) {
    return x;
  }
};

struct copy_conj_functor {
  template <typename T, enable_if_t<!is_complex_v<T>, bool> = true>
  inline T operator()(const T x) {
    return x;
  }
  template <typename T, enable_if_t<is_complex_v<T>, bool> = true>
  inline T operator()(const T x) {
    return T(x.x, -x.y);
  }
};

struct copy_neg_functor {
  template <typename T>
  inline T operator()(const T x) {
    return T(-1 * x);
  }
};

#define REGISTER_COPY_CAST(DTYPE)               \
  REGISTER_UNARY_OP_CAST(copy_identity, DTYPE); \
  REGISTER_UNARY_OP_CAST(copy_conj, DTYPE);     \
  REGISTER_UNARY_OP_CAST(copy_neg, DTYPE)

REGISTER_COPY_CAST(bool);
REGISTER_COPY_CAST(uchar);
REGISTER_COPY_CAST(char);
REGISTER_COPY_CAST(short);
REGISTER_COPY_CAST(int);
REGISTER_COPY_CAST(long);
REGISTER_COPY_CAST(half);
REGISTER_COPY_CAST(bfloat);
REGISTER_COPY_CAST(float);
REGISTER_COPY_CAST(float2);
REGISTER_COPY_CAST(half2);
