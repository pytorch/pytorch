#include <ATen/native/mps/kernels/Copy.h>
#include <c10/metal/indexing.h>
#include <c10/metal/utils.h>
#include <metal_stdlib>
using namespace metal;
using namespace c10::metal;

// Each thread copies a 16-byte chunk of a CONTIGUOUS input into the output,
// using the widest aligned vector store both endpoints allow, with a per-byte
// fallback for runs that cross a slice boundary (or the dispatch tail). The
// output is regularly-strided blocks; see StridedBlockParams /
// inner_contiguous_scatter_mps.
template <typename I>
kernel void inner_contiguous_scatter(
    constant uchar* input [[buffer(0)]],
    device uchar* output [[buffer(1)]],
    constant StridedBlockParams<I>& params [[buffer(2)]],
    uint tid [[thread_position_in_grid]]) {
  uint pos = tid * 16;
  if (pos >= params.nbytes) {
    return;
  }
  I g = params.chunk_base + pos;
  I o = g / params.slice_bytes;
  I j = g - o * params.slice_bytes;
  constant uchar* inp = input + g;

  // A run that crosses a slice boundary (or the dispatch tail) maps to
  // discontiguous output, so copy it byte by byte, recomputing the destination.
  if (params.slice_bytes - j < 16 || pos + 16 > params.nbytes) {
    uint stop = min(pos + 16u, params.nbytes) - pos;
    for (uint i = 0; i < stop; i++) {
      I gg = g + i;
      I oo = gg / params.slice_bytes;
      I jj = gg - oo * params.slice_bytes;
      output[oo * params.out_stride_bytes + params.off_bytes + jj] = inp[i];
    }
    return;
  }

  // Whole 16-byte run into a contiguous slice of the output.
  device uchar* out =
      output + o * params.out_stride_bytes + params.off_bytes + j;
  copy_bytes_aligned(out, inp, 16);
}

#define REGISTER_INNER_CONTIGUOUS_SCATTER_OP(I, SUFFIX)       \
  template [[host_name("inner_contiguous_scatter_" #SUFFIX)]] \
  kernel void inner_contiguous_scatter<I>(                    \
      constant uchar * input [[buffer(0)]],                   \
      device uchar * output [[buffer(1)]],                    \
      constant StridedBlockParams<I> & params [[buffer(2)]],  \
      uint tid [[thread_position_in_grid]]);

REGISTER_INNER_CONTIGUOUS_SCATTER_OP(uint, u32);
REGISTER_INNER_CONTIGUOUS_SCATTER_OP(ulong, u64);

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
