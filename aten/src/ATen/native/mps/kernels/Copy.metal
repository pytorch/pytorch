#include <c10/metal/indexing.h>
#include <c10/metal/utils.h>
#include <metal_stdlib>
using namespace metal;
using namespace c10::metal;

// Castout: input is loaded at compile-time Tin (the registered input dtype) and
// the result is cast to the user-supplied output dtype on store (runtime
// ScalarType switch in store_at_offs handles all dtype combinations, including
// real<->complex packing). One kernel per input dtype covers every output
// dtype.

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

#define REGISTER_COPY_CASTOUT(DTYPE)               \
  REGISTER_UNARY_OP_CASTOUT(copy_identity, DTYPE); \
  REGISTER_UNARY_OP_CASTOUT(copy_conj, DTYPE);     \
  REGISTER_UNARY_OP_CASTOUT(copy_neg, DTYPE)

REGISTER_COPY_CASTOUT(bool);
REGISTER_COPY_CASTOUT(uchar);
REGISTER_COPY_CASTOUT(char);
REGISTER_COPY_CASTOUT(short);
REGISTER_COPY_CASTOUT(int);
REGISTER_COPY_CASTOUT(long);
REGISTER_COPY_CASTOUT(half);
REGISTER_COPY_CASTOUT(bfloat);
REGISTER_COPY_CASTOUT(float);
REGISTER_COPY_CASTOUT(float2);
REGISTER_COPY_CASTOUT(half2);
