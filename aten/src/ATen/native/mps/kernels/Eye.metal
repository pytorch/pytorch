#include <metal_stdlib>
using namespace metal;

// For complex types (float2/half2), "1" is (1, 0), not (1, 1)
template <typename T>
constexpr T eye_one() {
  return static_cast<T>(1);
}
template <>
constexpr float2 eye_one<float2>() {
  return float2(1.0, 0.0);
}
template <>
constexpr half2 eye_one<half2>() {
  return half2(half(1.0), half(0.0));
}

// Single-pass: writes both 0s and 1s in one dispatch (better for small tensors)
template <typename T>
kernel void eye(
    device T* output [[buffer(0)]],
    constant long& stride0 [[buffer(1)]],
    constant long& stride1 [[buffer(2)]],
    uint2 pos [[thread_position_in_grid]]) {
  output[pos.y * stride0 + pos.x * stride1] =
      (pos.x == pos.y) ? eye_one<T>() : static_cast<T>(0);
}

// Diagonal-only: writes 1s to pre-zeroed tensor (better for large tensors)
template <typename T>
kernel void eye_diag(
    device T* output [[buffer(0)]],
    constant long& diag_stride [[buffer(1)]],
    uint index [[thread_position_in_grid]]) {
  output[index * diag_stride] = eye_one<T>();
}

#define REGISTER_EYE_OP(DTYPE)                                            \
  template [[host_name("eye_" #DTYPE)]] kernel void eye<DTYPE>(           \
      device DTYPE * output [[buffer(0)]],                                \
      constant long& stride0 [[buffer(1)]],                               \
      constant long& stride1 [[buffer(2)]],                               \
      uint2 pos [[thread_position_in_grid]]);                             \
  template [[host_name("eye_diag_" #DTYPE)]] kernel void eye_diag<DTYPE>( \
      device DTYPE * output [[buffer(0)]],                                \
      constant long& diag_stride [[buffer(1)]],                           \
      uint index [[thread_position_in_grid]]);

REGISTER_EYE_OP(float);
REGISTER_EYE_OP(half);
REGISTER_EYE_OP(bfloat);
REGISTER_EYE_OP(float2);
REGISTER_EYE_OP(half2);
REGISTER_EYE_OP(long);
REGISTER_EYE_OP(int);
REGISTER_EYE_OP(short);
REGISTER_EYE_OP(char);
REGISTER_EYE_OP(uchar);
REGISTER_EYE_OP(bool);
