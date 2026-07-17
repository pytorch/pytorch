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
// no functor, so move the contiguous inner run as raw bytes via
// copy_bytes_aligned regardless of element dtype. grid.x indexes 16-byte chunks
// of the inner run.
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
  const uint pos = thread_pos.x * 16;
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
  copy_bytes_aligned(o, in, min(16u, inner_bytes - pos));
}

// Fully contiguous same-dtype copy: a flat byte run, so no outer offsets to
// compute. The host issues one dispatch per <=2GB chunk (base is the chunk's
// byte offset, chunk_bytes its size), matching the blit path's chunking.
kernel void contiguous_byte_copy(
    device uchar* out [[buffer(0)]],
    constant uchar* in [[buffer(1)]],
    constant uint& chunk_bytes [[buffer(2)]],
    constant ulong& base [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  const uint pos = tid * 16;
  if (pos >= chunk_bytes) {
    return;
  }
  copy_bytes_aligned(
      out + base + pos, in + base + pos, min(16u, chunk_bytes - pos));
}
