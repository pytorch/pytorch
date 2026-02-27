#include <c10/metal/indexing.h>
#include <c10/metal/utils.h>
#include <metal_stdlib>
using namespace metal;
using namespace c10::metal;

template <typename T>
inline void cross_product(
    device T* out,
    long os,
    constant T* x,
    long xs,
    constant T* y,
    long ys) {
  out[0] = mul(x[xs], y[2 * ys]) - mul(x[2 * xs], y[ys]);
  out[os] = mul(x[2 * xs], y[0]) - mul(x[0], y[2 * ys]);
  out[2 * os] = mul(x[0], y[ys]) - mul(x[xs], y[0]);
}

// Handles any contiguous layout regardless of which dim is the cross dim.
// For cross-dim element stride B, thread tid covers the triple whose base is
//   base = tid + 2*(tid/B)*B
// i.e. B=numThreads (outermost) → base=tid; B=1 (innermost) → base=3*tid.
template <typename T>
kernel void cross_dense(
    device T* out [[buffer(0)]],
    constant T* input [[buffer(1)]],
    constant T* other [[buffer(2)]],
    constant uint& dim_stride [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  uint base = tid + 2 * (tid / dim_stride) * dim_stride;
  cross_product(
      out + base,
      long(dim_stride),
      input + base,
      long(dim_stride),
      other + base,
      long(dim_stride));
}

template <typename T>
kernel void cross_strided(
    device T* out [[buffer(0)]],
    constant T* input [[buffer(1)]],
    constant T* other [[buffer(2)]],
    constant long* squashed_sizes [[buffer(3)]],
    constant long* out_strides [[buffer(4)]],
    constant long* input_strides [[buffer(5)]],
    constant long* other_strides [[buffer(6)]],
    constant uint& ndim [[buffer(7)]],
    constant uint& dim [[buffer(8)]],
    uint tid [[thread_position_in_grid]]) {
  int pos[max_ndim];
  pos_from_thread_index(int(tid), pos, squashed_sizes, ndim - 1);
  long out_offs = 0, in_offs = 0, oth_offs = 0;
  uint p = 0;
  for (uint d = 0; d < ndim; d++) {
    if (d == dim)
      continue;
    out_offs += pos[p] * out_strides[d];
    in_offs += pos[p] * input_strides[d];
    oth_offs += pos[p] * other_strides[d];
    p++;
  }
  cross_product(
      out + out_offs,
      out_strides[dim],
      input + in_offs,
      input_strides[dim],
      other + oth_offs,
      other_strides[dim]);
}

#define REGISTER_CROSS_OP(DTYPE)                              \
  template [[host_name("cross_dense_" #DTYPE)]] kernel void   \
  cross_dense<DTYPE>(                                         \
      device DTYPE * out [[buffer(0)]],                       \
      constant DTYPE * input [[buffer(1)]],                   \
      constant DTYPE * other [[buffer(2)]],                   \
      constant uint & dim_stride [[buffer(3)]],               \
      uint tid [[thread_position_in_grid]]);                  \
  template [[host_name("cross_strided_" #DTYPE)]] kernel void \
  cross_strided<DTYPE>(                                       \
      device DTYPE * out [[buffer(0)]],                       \
      constant DTYPE * input [[buffer(1)]],                   \
      constant DTYPE * other [[buffer(2)]],                   \
      constant long* squashed_sizes [[buffer(3)]],            \
      constant long* out_strides [[buffer(4)]],               \
      constant long* input_strides [[buffer(5)]],             \
      constant long* other_strides [[buffer(6)]],             \
      constant uint& ndim [[buffer(7)]],                      \
      constant uint& dim [[buffer(8)]],                       \
      uint tid [[thread_position_in_grid]])

REGISTER_CROSS_OP(float);
REGISTER_CROSS_OP(half);
REGISTER_CROSS_OP(bfloat);
REGISTER_CROSS_OP(float2);
REGISTER_CROSS_OP(half2);
REGISTER_CROSS_OP(long);
REGISTER_CROSS_OP(int);
REGISTER_CROSS_OP(short);
REGISTER_CROSS_OP(char);
REGISTER_CROSS_OP(uchar);
