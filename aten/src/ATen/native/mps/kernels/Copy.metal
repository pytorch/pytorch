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

// Byte-erased identity copy of an inner-contiguous view: a same-dtype copy has
// no functor, so move the contiguous inner run as raw bytes with the widest
// aligned vector (uint4 -> ushort -> byte ladder) regardless of element dtype.
// grid.x indexes 16-byte chunks of the inner run.
kernel void inner_contiguous_copy(
    device uchar* output [[buffer(0)]],
    constant uchar* input [[buffer(1)]],
    constant long* outer_sizes [[buffer(2)]],
    constant long* input_outer_strides [[buffer(3)]],
    constant long* output_outer_strides [[buffer(4)]],
    constant uint2& ndim_outer_inner_bytes [[buffer(5)]],
    uint2 thread_pos [[thread_position_in_grid]]) {
  const uint ndim_outer = ndim_outer_inner_bytes.x;
  const uint inner_bytes = ndim_outer_inner_bytes.y;
  uint pos = thread_pos.x * 16;
  if (pos >= inner_bytes) {
    return;
  }
  int opos[max_ndim];
  pos_from_thread_index(int(thread_pos.y), opos, outer_sizes, ndim_outer);
  const auto in_base = offset_from_coord(opos, input_outer_strides, ndim_outer);
  const auto out_base =
      offset_from_coord(opos, output_outer_strides, ndim_outer);
  device uchar* o = output + out_base + pos;
  constant uchar* in = input + in_base + pos;
  uint n = min(16u, inner_bytes - pos);
  if (n < 16) {
    for (uint k = 0; k < n; ++k) {
      o[k] = in[k];
    }
    return;
  }
  uint align = (reinterpret_cast<ulong>(o) | reinterpret_cast<ulong>(in)) & 15;
  if (align == 0) {
    *reinterpret_cast<device uint4*>(o) =
        *reinterpret_cast<constant uint4*>(in);
  } else if ((align & 7) == 0) {
    for (uint k = 0; k < 2; ++k) {
      reinterpret_cast<device uint2*>(o)[k] =
          reinterpret_cast<constant uint2*>(in)[k];
    }
  } else if ((align & 3) == 0) {
    for (uint k = 0; k < 4; ++k) {
      reinterpret_cast<device uint*>(o)[k] =
          reinterpret_cast<constant uint*>(in)[k];
    }
  } else if ((align & 1) == 0) {
    for (uint k = 0; k < 8; ++k) {
      reinterpret_cast<device ushort*>(o)[k] =
          reinterpret_cast<constant ushort*>(in)[k];
    }
  } else {
    for (uint k = 0; k < 16; ++k) {
      o[k] = in[k];
    }
  }
}
