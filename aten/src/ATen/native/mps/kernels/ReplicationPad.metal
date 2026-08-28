#include <c10/metal/utils.h>
#include <metal_stdlib>

using namespace metal;
using namespace c10::metal;

template <typename T>
kernel void replication_pad1d_forward(
    constant T* input [[buffer(0)]],
    device T* output [[buffer(1)]],
    constant int4& sizes_pad [[buffer(2)]], // (input_W, output_W, padL, padR)
    uint3 tid [[thread_position_in_grid]],
    uint3 grid [[threads_per_grid]]) {
  const int input_W = sizes_pad.x;
  const int output_W = sizes_pad.y;
  const int padL = sizes_pad.z;

  const int w_out = static_cast<int>(tid.x);
  const uint c = tid.y;
  const uint n = tid.z;
  const uint nplane = grid.y;

  const int iStart = max(0, -padL);
  const int oStart = max(0, padL);
  const int w_in = min(max(padL, w_out), input_W + padL - 1) - oStart + iStart;

  const ulong in_base =
      (static_cast<ulong>(n) * nplane + c) * static_cast<ulong>(input_W);
  const ulong out_base =
      (static_cast<ulong>(n) * nplane + c) * static_cast<ulong>(output_W);
  output[out_base + static_cast<ulong>(w_out)] =
      input[in_base + static_cast<ulong>(w_in)];
}

template <typename T>
kernel void replication_pad1d_backward(
    constant T* grad_output [[buffer(0)]],
    device T* grad_input [[buffer(1)]],
    constant int4& sizes_pad [[buffer(2)]], // (input_W, output_W, padL, padR)
    uint3 tid [[thread_position_in_grid]],
    uint3 grid [[threads_per_grid]]) {
  const int input_W = sizes_pad.x;
  const int output_W = sizes_pad.y;
  const int padL = sizes_pad.z;

  const int w_in = static_cast<int>(tid.x);
  const uint c = tid.y;
  const uint n = tid.z;
  const uint nplane = grid.y;

  int wo_lo = 0;
  int wo_hi = -1;
  if (input_W == 1) {
    wo_lo = 0;
    wo_hi = output_W - 1;
  } else if (w_in == 0) {
    wo_lo = 0;
    wo_hi = min(padL, output_W - 1);
  } else if (w_in == input_W - 1) {
    wo_lo = max(0, input_W + padL - 1);
    wo_hi = output_W - 1;
  } else {
    const int wo = w_in + padL;
    if (wo >= 0 && wo < output_W) {
      wo_lo = wo;
      wo_hi = wo;
    }
  }

  const ulong in_base =
      (static_cast<ulong>(n) * nplane + c) * static_cast<ulong>(input_W);
  const ulong out_base =
      (static_cast<ulong>(n) * nplane + c) * static_cast<ulong>(output_W);

  opmath_t<T> sum = 0;
  for (int wo = wo_lo; wo <= wo_hi; ++wo) {
    sum += static_cast<opmath_t<T>>(
        grad_output[out_base + static_cast<ulong>(wo)]);
  }
  grad_input[in_base + static_cast<ulong>(w_in)] = static_cast<T>(sum);
}

#define INSTANTIATE_REPLICATION_PAD1D_FWD(DTYPE)              \
  template [[host_name("replication_pad1d_forward_" #DTYPE)]] \
  kernel void replication_pad1d_forward<DTYPE>(               \
      constant DTYPE * input [[buffer(0)]],                   \
      device DTYPE * output [[buffer(1)]],                    \
      constant int4 & sizes_pad [[buffer(2)]],                \
      uint3 tid [[thread_position_in_grid]],                  \
      uint3 grid [[threads_per_grid]])

#define INSTANTIATE_REPLICATION_PAD1D_BWD(DTYPE)               \
  template [[host_name("replication_pad1d_backward_" #DTYPE)]] \
  kernel void replication_pad1d_backward<DTYPE>(               \
      constant DTYPE * grad_output [[buffer(0)]],              \
      device DTYPE * grad_input [[buffer(1)]],                 \
      constant int4 & sizes_pad [[buffer(2)]],                 \
      uint3 tid [[thread_position_in_grid]],                   \
      uint3 grid [[threads_per_grid]])

INSTANTIATE_REPLICATION_PAD1D_FWD(float);
INSTANTIATE_REPLICATION_PAD1D_FWD(half);
INSTANTIATE_REPLICATION_PAD1D_FWD(bfloat);
INSTANTIATE_REPLICATION_PAD1D_FWD(float2);
INSTANTIATE_REPLICATION_PAD1D_FWD(half2);
INSTANTIATE_REPLICATION_PAD1D_FWD(long);
INSTANTIATE_REPLICATION_PAD1D_FWD(int);
INSTANTIATE_REPLICATION_PAD1D_FWD(short);
INSTANTIATE_REPLICATION_PAD1D_FWD(char);
INSTANTIATE_REPLICATION_PAD1D_FWD(uchar);
INSTANTIATE_REPLICATION_PAD1D_FWD(bool);

INSTANTIATE_REPLICATION_PAD1D_BWD(float);
INSTANTIATE_REPLICATION_PAD1D_BWD(half);
INSTANTIATE_REPLICATION_PAD1D_BWD(bfloat);
INSTANTIATE_REPLICATION_PAD1D_BWD(float2);
INSTANTIATE_REPLICATION_PAD1D_BWD(half2);
