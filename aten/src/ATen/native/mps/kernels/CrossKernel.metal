#include <c10/metal/indexing.h>
#include <c10/metal/utils.h>
#include <metal_stdlib>
using namespace metal;
using namespace c10::metal;

template <typename T>
inline void cross_product(
    device T* out,
    long os,
    device const T* x,
    long xs,
    device const T* y,
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
    device const T* input [[buffer(1)]],
    device const T* other [[buffer(2)]],
    constant uint& dim_stride [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  const long base = tid + 2 * long(tid / dim_stride) * dim_stride;
  cross_product(
      out + base,
      long(dim_stride),
      input + base,
      long(dim_stride),
      other + base,
      long(dim_stride));
}

// Iterates a TensorIterator whose cross dim is squashed to size 1: sizes and
// byte strides locate each triplet's first element, dim_strides holds the
// element strides along the cross dim of out, input and other.
template <typename T>
kernel void cross_strided(
    device void* out [[buffer(0)]],
    device const char* input [[buffer(1)]],
    device const char* other [[buffer(2)]],
    constant DimDivider* sizes [[buffer(3)]],
    constant long* out_strides [[buffer(4)]],
    constant long* input_strides [[buffer(5)]],
    constant long* other_strides [[buffer(6)]],
    constant uint& ndim [[buffer(7)]],
    constant long* dim_strides [[buffer(8)]],
    uint tid [[thread_position_in_grid]]) {
  const auto offs = offsets_from_thread_index(
      tid, sizes, ndim, out_strides, input_strides, other_strides);
  cross_product(
      &ref_at_offs<T>(out, offs[0]),
      dim_strides[0],
      reinterpret_cast<device const T*>(input + offs[1]),
      dim_strides[1],
      reinterpret_cast<device const T*>(other + offs[2]),
      dim_strides[2]);
}

#define REGISTER_CROSS_OP(DTYPE)                              \
  template [[host_name("cross_dense_" #DTYPE)]] kernel void   \
  cross_dense<DTYPE>(                                         \
      device DTYPE * out [[buffer(0)]],                       \
      device const DTYPE* input [[buffer(1)]],                \
      device const DTYPE* other [[buffer(2)]],                \
      constant uint& dim_stride [[buffer(3)]],                \
      uint tid [[thread_position_in_grid]]);                  \
  template [[host_name("cross_strided_" #DTYPE)]] kernel void \
  cross_strided<DTYPE>(                                       \
      device void* out [[buffer(0)]],                         \
      device const char* input [[buffer(1)]],                 \
      device const char* other [[buffer(2)]],                 \
      constant DimDivider* sizes [[buffer(3)]],               \
      constant long* out_strides [[buffer(4)]],               \
      constant long* input_strides [[buffer(5)]],             \
      constant long* other_strides [[buffer(6)]],             \
      constant uint& ndim [[buffer(7)]],                      \
      constant long* dim_strides [[buffer(8)]],               \
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
