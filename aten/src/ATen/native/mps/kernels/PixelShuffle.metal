#include <metal_stdlib>
using namespace metal;

template <typename T>
kernel void pixel_shuffle(
    device const T* input [[buffer(0)]],
    device T* output [[buffer(1)]],
    constant long* in_strides [[buffer(2)]],
    constant long* out_strides [[buffer(3)]],
    constant uint2& factor_zdim [[buffer(4)]],
    constant bool& channel_major [[buffer(5)]],
    uint3 pos [[thread_position_in_grid]]) {
  const uint factor = factor_zdim.x, z_dim = factor_zdim.y;
  // grid is (wd, hd, n * C + c), or (c, wd, n * Hd + hd) when channel_major
  const uint n = pos.z / z_dim, z = pos.z - n * z_dim;
  const uint channel = channel_major ? pos.x : z,
             hd = channel_major ? z : pos.y, wd = channel_major ? pos.y : pos.x;
  const long in_base = n * in_strides[0] + channel * in_strides[1] +
      hd * in_strides[2] + wd * in_strides[3];
  const long out_base = n * out_strides[0] + channel * out_strides[1] +
      hd * out_strides[2] + wd * out_strides[3];
  for (uint i = 0; i < factor; ++i) {
    long in = in_base + i * in_strides[4], out = out_base + i * out_strides[4];
    for (uint j = 0; j < factor;
         ++j, in += in_strides[5], out += out_strides[5]) {
      output[out] = input[in];
    }
  }
}

#define REGISTER_PIXEL_SHUFFLE(WIDTH, DTYPE)                  \
  template [[host_name("pixel_shuffle_" #WIDTH)]] kernel void \
  pixel_shuffle<DTYPE>(                                       \
      device const DTYPE* input,                              \
      device DTYPE* output,                                   \
      constant long* in_strides,                              \
      constant long* out_strides,                             \
      constant uint2& factor_zdim,                            \
      constant bool& channel_major,                           \
      uint3 pos);

REGISTER_PIXEL_SHUFFLE(1, uchar);
REGISTER_PIXEL_SHUFFLE(2, ushort);
REGISTER_PIXEL_SHUFFLE(4, uint);
REGISTER_PIXEL_SHUFFLE(8, ulong);
