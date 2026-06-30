#include <c10/metal/indexing.h>
#include <c10/metal/utils.h>
#include <metal_stdlib>
using namespace metal;
using namespace c10::metal;

// Castout: input is loaded at compile-time Tin (the registered input dtype) and
// the result is cast to the user-supplied output dtype on store (runtime
// ScalarType switch in store_at_offs handles all dtype combinations, including
// real<->complex packing). REGISTER_UNARY_OP(NAME, DTYPE, DTYPE) registers both
// the direct same-dtype kernel and the castout variant keyed on the input
// dtype; exec_unary_kernel auto-falls back to castout when the direct
// per-(out,in) kernel isn't registered.

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

// Fused conj+neg: complex only. On real types conj is identity, so
// conj+neg degenerates to plain neg and the caller routes there instead.
struct copy_conj_neg_functor {
  template <typename T, enable_if_t<is_complex_v<T>, bool> = true>
  inline T operator()(const T x) {
    return T(-x.x, x.y);
  }
};

#define REGISTER_COPY_CASTOUT(DTYPE)              \
  REGISTER_UNARY_OP(copy_identity, DTYPE, DTYPE); \
  REGISTER_UNARY_OP(copy_conj, DTYPE, DTYPE);     \
  REGISTER_UNARY_OP(copy_neg, DTYPE, DTYPE)

REGISTER_COPY_CASTOUT(bool);
REGISTER_COPY_CASTOUT(uchar);
REGISTER_COPY_CASTOUT(char);
REGISTER_COPY_CASTOUT(short);
REGISTER_COPY_CASTOUT(int);
REGISTER_COPY_CASTOUT(long);
// Unsigned integer views (e.g. complex64 viewed as uint64) reach the copy path.
REGISTER_COPY_CASTOUT(ushort);
REGISTER_COPY_CASTOUT(uint);
REGISTER_COPY_CASTOUT(ulong);
REGISTER_COPY_CASTOUT(half);
REGISTER_COPY_CASTOUT(bfloat);
REGISTER_COPY_CASTOUT(float);
REGISTER_COPY_CASTOUT(float2);
REGISTER_COPY_CASTOUT(half2);

REGISTER_UNARY_OP(copy_conj_neg, float2, float2);
REGISTER_UNARY_OP(copy_conj_neg, half2, half2);
