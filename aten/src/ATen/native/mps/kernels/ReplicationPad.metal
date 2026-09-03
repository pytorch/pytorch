#include <ATen/native/mps/kernels/Pad.h>
#include <c10/metal/utils.h>
#include <metal_stdlib>

using namespace metal;
using namespace c10::metal;

template <typename T>
kernel void constant_pad_nd_dense(
    constant T* input [[buffer(0)]],
    device T* output [[buffer(1)]],
    constant ConstantPadDenseParams& params [[buffer(2)]],
    constant T& fill_value [[buffer(3)]],
    uint3 tid [[thread_position_in_grid]],
    uint3 grid [[threads_per_grid]]) {
  const uint input_w = params.input_sizes[0];
  const uint input_h = params.input_sizes[1];
  const uint input_d = params.input_sizes[2];
  const uint output_w = params.output_sizes[0];
  const uint output_h = params.output_sizes[1];
  const uint output_d = params.output_sizes[2];
  const uint left_w = params.left_pad[0];
  const uint left_h = params.left_pad[1];
  const uint left_d = params.left_pad[2];
  const uint output_d_idx = tid.z % output_d;
  const uint outer_idx = tid.z / output_d;
  const bool h_in_bounds = tid.y >= left_h && tid.y - left_h < input_h;
  const bool d_in_bounds =
      output_d_idx >= left_d && output_d_idx - left_d < input_d;
  const bool outer_in_bounds = h_in_bounds && d_in_bounds;
  const uint input_h_idx = h_in_bounds ? tid.y - left_h : 0;
  const uint input_d_idx = d_in_bounds ? output_d_idx - left_d : 0;
  const ulong output_base = (ulong(tid.z) * output_h + tid.y) * output_w;
  const ulong input_base =
      ((ulong(outer_idx) * input_d + input_d_idx) * input_h + input_h_idx) *
      input_w;

  for (uint i = 0; i < ILP_PER_THREAD; ++i) {
    const uint output_w_idx = tid.x + i * grid.x;
    if (output_w_idx >= output_w) {
      break;
    }
    const bool w_in_bounds =
        output_w_idx >= left_w && output_w_idx - left_w < input_w;
    const uint input_w_idx = w_in_bounds ? output_w_idx - left_w : 0;
    output[output_base + output_w_idx] = outer_in_bounds && w_in_bounds
        ? input[input_base + input_w_idx]
        : fill_value;
  }
}

template <typename T, typename idx_t>
kernel void constant_pad_nd(
    constant T* input [[buffer(0)]],
    device T* output [[buffer(1)]],
    constant ConstantPadNdParams<idx_t>& params [[buffer(2)]],
    constant T& fill_value [[buffer(3)]],
    uint2 tid [[thread_position_in_grid]],
    uint2 grid [[threads_per_grid]]) {
  idx_t input_offset = 0;
  idx_t output_offset = 0;
  idx_t outer_idx = tid.y;
  bool outer_in_bounds = true;
  for (uint dim = 1; dim < params.ndim; ++dim) {
    const idx_t output_idx = outer_idx % params.output_sizes[dim];
    outer_idx /= params.output_sizes[dim];
    const bool dim_in_bounds = output_idx >= params.left_pad[dim] &&
        output_idx - params.left_pad[dim] < params.input_sizes[dim];
    const idx_t input_idx =
        dim_in_bounds ? output_idx - params.left_pad[dim] : 0;
    output_offset += output_idx * params.output_strides[dim];
    input_offset += input_idx * params.input_strides[dim];
    outer_in_bounds = outer_in_bounds && dim_in_bounds;
  }

  for (uint i = 0; i < ILP_PER_THREAD; ++i) {
    const auto output_idx = idx_t(tid.x) + idx_t(i) * idx_t(grid.x);
    if (output_idx >= params.output_sizes[0]) {
      break;
    }
    const bool dim_in_bounds = output_idx >= params.left_pad[0] &&
        output_idx - params.left_pad[0] < params.input_sizes[0];
    const idx_t input_idx = dim_in_bounds ? output_idx - params.left_pad[0] : 0;
    output[output_offset + output_idx * params.output_strides[0]] =
        outer_in_bounds && dim_in_bounds
        ? input[input_offset + input_idx * params.input_strides[0]]
        : fill_value;
  }
}

#define INSTANTIATE_CONSTANT_PAD_ND_IDX(DTYPE, IDX, SUFFIX) \
  template [[host_name("constant_pad_nd_" #DTYPE #SUFFIX)]] \
  kernel void constant_pad_nd<DTYPE, IDX>(                  \
      constant DTYPE*,                                      \
      device DTYPE*,                                        \
      constant ConstantPadNdParams<IDX>&,                   \
      constant DTYPE&,                                      \
      uint2,                                                \
      uint2)

#define INSTANTIATE_CONSTANT_PAD_ND(DTYPE)                \
  template [[host_name("constant_pad_nd_dense_" #DTYPE)]] \
  kernel void constant_pad_nd_dense<DTYPE>(               \
      constant DTYPE*,                                    \
      device DTYPE*,                                      \
      constant ConstantPadDenseParams&,                   \
      constant DTYPE&,                                    \
      uint3,                                              \
      uint3);                                             \
  INSTANTIATE_CONSTANT_PAD_ND_IDX(DTYPE, uint, _u32);     \
  INSTANTIATE_CONSTANT_PAD_ND_IDX(DTYPE, ulong, _u64)

INSTANTIATE_CONSTANT_PAD_ND(float);
INSTANTIATE_CONSTANT_PAD_ND(half);
INSTANTIATE_CONSTANT_PAD_ND(bfloat);
INSTANTIATE_CONSTANT_PAD_ND(long);
INSTANTIATE_CONSTANT_PAD_ND(ulong);
INSTANTIATE_CONSTANT_PAD_ND(int);
INSTANTIATE_CONSTANT_PAD_ND(uint);
INSTANTIATE_CONSTANT_PAD_ND(short);
INSTANTIATE_CONSTANT_PAD_ND(ushort);
INSTANTIATE_CONSTANT_PAD_ND(char);
INSTANTIATE_CONSTANT_PAD_ND(uchar);
INSTANTIATE_CONSTANT_PAD_ND(bool);
INSTANTIATE_CONSTANT_PAD_ND(float2);
INSTANTIATE_CONSTANT_PAD_ND(half2);

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
