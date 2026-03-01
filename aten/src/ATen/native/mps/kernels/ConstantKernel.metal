#include <metal_stdlib>

using namespace metal;

template <typename T>
kernel void fill_scalar_dense(
    device T* out [[buffer(0)]],
    constant T& fill_val [[buffer(1)]],
    uint index [[thread_position_in_grid]]) {
  out[index] = fill_val;
}

#define REGISTER_FILL_OP(T)                                       \
  template [[host_name("fill_scalar_dense_" #T)]] kernel void     \
      fill_scalar_dense<T>(device T*, constant T&, uint)

REGISTER_FILL_OP(float);
REGISTER_FILL_OP(half);
REGISTER_FILL_OP(bfloat);
REGISTER_FILL_OP(int);
REGISTER_FILL_OP(long);
REGISTER_FILL_OP(short);
REGISTER_FILL_OP(char);
REGISTER_FILL_OP(uchar);
REGISTER_FILL_OP(bool);
REGISTER_FILL_OP(float2);
REGISTER_FILL_OP(half2);

// 2D dispatch: tid.y = dim-0 index (no division), tid.x = linear index for dims 1..ndim-1.
// For an N-dim tensor this requires N-1 divisions instead of N, and consecutive threads
// in x access consecutive addresses in the innermost dimension (coalesced writes).
template <typename T>
kernel void fill_scalar_strided(
    device T* out [[buffer(0)]],
    constant T& fill_val [[buffer(1)]],
    constant long* sizes [[buffer(2)]],
    constant long* strides [[buffer(3)]],
    constant uint& ndim [[buffer(4)]],
    uint2 tid [[thread_position_in_grid]]) {
  long offset = long(tid.y) * strides[0];
  uint inner = tid.x;
  for (uint i = 1; i < ndim; i++) {
    offset += long(inner % uint(sizes[i])) * strides[i];
    inner /= uint(sizes[i]);
  }
  out[offset] = fill_val;
}

#define REGISTER_FILL_STRIDED_OP(T)                                        \
  template [[host_name("fill_scalar_strided_" #T)]] kernel void            \
      fill_scalar_strided<T>(device T*, constant T&, constant long*,       \
                             constant long*, constant uint&, uint2)

REGISTER_FILL_STRIDED_OP(float);
REGISTER_FILL_STRIDED_OP(half);
REGISTER_FILL_STRIDED_OP(bfloat);
REGISTER_FILL_STRIDED_OP(int);
REGISTER_FILL_STRIDED_OP(long);
REGISTER_FILL_STRIDED_OP(short);
REGISTER_FILL_STRIDED_OP(char);
REGISTER_FILL_STRIDED_OP(uchar);
REGISTER_FILL_STRIDED_OP(bool);
REGISTER_FILL_STRIDED_OP(float2);
REGISTER_FILL_STRIDED_OP(half2);
