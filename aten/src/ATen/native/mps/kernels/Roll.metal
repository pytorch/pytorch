#include <metal_stdlib>
using namespace metal;

// Computes output[tid] = input[scatter(tid)] where scatter applies the
// per-dimension shifts. Strides are in element units, not bytes. shifts[d]
// is the effective shift for dim d, normalised into [0, sizes[d]).
template <typename T>
kernel void roll(
    constant T* input [[buffer(0)]],
    device T* output [[buffer(1)]],
    constant long* sizes [[buffer(2)]],
    constant long* in_strides [[buffer(3)]],
    constant long* out_strides [[buffer(4)]],
    constant long* shifts [[buffer(5)]],
    constant uint& ndim [[buffer(6)]],
    constant ulong& numel [[buffer(7)]],
    uint tid [[thread_position_in_grid]]) {
  if (ulong(tid) >= numel) {
    return;
  }
  long in_offset = 0;
  long out_lin = long(tid);
  // Walk dims outermost-first using contiguous output strides. After each
  // step `out_lin` holds the residual within the current dim, so the next
  // dim's coord is just (residual / stride) without an additional modulus.
  for (uint d = 0; d < ndim; ++d) {
    long os = out_strides[d];
    long out_coord = out_lin / os;
    out_lin -= out_coord * os;
    long in_coord = out_coord - shifts[d];
    if (in_coord < 0) {
      in_coord += sizes[d];
    }
    in_offset += in_coord * in_strides[d];
  }
  output[tid] = input[in_offset];
}

// 2D-specialised path. Avoids the per-dim div/sub loop of the general kernel
// by addressing the output via uint2 thread positions. For contiguous 2D
// tensors this is purely memory-bound. Wired in by the host when ndim == 2
// and both shifts/strides are aligned with the row-major output layout.
template <typename T>
kernel void roll_2d(
    constant T* input [[buffer(0)]],
    device T* output [[buffer(1)]],
    constant long2& sizes [[buffer(2)]],
    constant long2& shifts [[buffer(3)]],
    uint2 tid [[thread_position_in_grid]]) {
  long s0 = sizes.x;
  long s1 = sizes.y;
  long y = long(tid.y);
  long x = long(tid.x);
  if (y >= s0 || x >= s1) {
    return;
  }
  long iy = y - shifts.x;
  if (iy < 0) {
    iy += s0;
  }
  long ix = x - shifts.y;
  if (ix < 0) {
    ix += s1;
  }
  output[y * s1 + x] = input[iy * s1 + ix];
}

#define REGISTER_ROLL_OP(DTYPE)                                         \
  template [[host_name("roll_" #DTYPE)]] kernel void roll<DTYPE>(       \
      constant DTYPE * input [[buffer(0)]],                             \
      device DTYPE * output [[buffer(1)]],                              \
      constant long* sizes [[buffer(2)]],                               \
      constant long* in_strides [[buffer(3)]],                          \
      constant long* out_strides [[buffer(4)]],                         \
      constant long* shifts [[buffer(5)]],                              \
      constant uint& ndim [[buffer(6)]],                                \
      constant ulong& numel [[buffer(7)]],                              \
      uint tid [[thread_position_in_grid]]);                            \
  template [[host_name("roll_2d_" #DTYPE)]] kernel void roll_2d<DTYPE>( \
      constant DTYPE * input [[buffer(0)]],                             \
      device DTYPE * output [[buffer(1)]],                              \
      constant long2 & sizes [[buffer(2)]],                             \
      constant long2 & shifts [[buffer(3)]],                            \
      uint2 tid [[thread_position_in_grid]]);

REGISTER_ROLL_OP(float);
REGISTER_ROLL_OP(half);
REGISTER_ROLL_OP(bfloat);
REGISTER_ROLL_OP(float2);
REGISTER_ROLL_OP(half2);
REGISTER_ROLL_OP(long);
REGISTER_ROLL_OP(int);
REGISTER_ROLL_OP(short);
REGISTER_ROLL_OP(char);
REGISTER_ROLL_OP(uchar);
REGISTER_ROLL_OP(bool);
