#include <metal_array>

#include <metal_stdlib>
using namespace metal;

#define REGISTER_CROSS_FUNC(DTYPE)                       \
  static inline DTYPE##3 cross(DTYPE##3 x, DTYPE##3 y) { \
    DTYPE##3 out;                                        \
    out.x = x.y * y.z - x.z * y.y;                       \
    out.y = x.z * y.x - x.x * y.z;                       \
    out.z = x.x * y.y - x.y * y.x;                       \
    return out;                                          \
  }

// Metal only supports half and float for native cross implementation.
// For all the other data types, implement cross manually.
REGISTER_CROSS_FUNC(int);
REGISTER_CROSS_FUNC(long);
REGISTER_CROSS_FUNC(short);
REGISTER_CROSS_FUNC(char);
REGISTER_CROSS_FUNC(uchar);
REGISTER_CROSS_FUNC(bool);
REGISTER_CROSS_FUNC(bfloat);

template <typename T, typename U>
kernel void cross(
    constant void* input_ [[buffer(0)]],
    constant void* other_ [[buffer(1)]],
    device void* out_ [[buffer(2)]],
    constant uint3* offsets [[buffer(3)]],
    constant int64_t& outStride [[buffer(4)]],
    constant int64_t& inputStride [[buffer(5)]],
    constant int64_t& otherStride [[buffer(6)]],
    uint tid [[thread_position_in_grid]]) {
  device T* out = (device T*)((device uint8_t*)out_ + offsets[tid].x);
  constant T* input = (constant T*)((constant uint8_t*)input_ + offsets[tid].y);
  constant T* other = (constant T*)((constant uint8_t*)other_ + offsets[tid].z);

  const U x = {
      input[0 * inputStride], input[1 * inputStride], input[2 * inputStride]};
  const U y = {
      other[0 * otherStride], other[1 * otherStride], other[2 * otherStride]};
  const U res = cross(x, y);

  out[0 * outStride] = res.x;
  out[1 * outStride] = res.y;
  out[2 * outStride] = res.z;
}

#define REGISTER_CROSS_OP(DTYPE)                                              \
  template [[host_name("cross_" #DTYPE)]] kernel void cross<DTYPE, DTYPE##3>( \
      constant void* input_ [[buffer(0)]],                                    \
      constant void* other_ [[buffer(1)]],                                    \
      device void* out_ [[buffer(2)]],                                        \
      constant uint3* offsets [[buffer(3)]],                                  \
      constant int64_t& outStride [[buffer(4)]],                              \
      constant int64_t& inputStride [[buffer(5)]],                            \
      constant int64_t& otherStride [[buffer(6)]],                            \
      uint tid [[thread_position_in_grid]]);

REGISTER_CROSS_OP(float);
REGISTER_CROSS_OP(half);
REGISTER_CROSS_OP(int);
REGISTER_CROSS_OP(long);
REGISTER_CROSS_OP(short);
REGISTER_CROSS_OP(char);
REGISTER_CROSS_OP(uchar);
REGISTER_CROSS_OP(bool);
REGISTER_CROSS_OP(bfloat);
