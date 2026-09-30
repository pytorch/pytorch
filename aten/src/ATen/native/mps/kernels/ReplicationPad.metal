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
INSTANTIATE_CONSTANT_PAD_ND(float8_e4m3fn);
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

template <typename idx_t>
inline idx_t reflect_index(idx_t pos, idx_t size) {
  const idx_t last = size - 1;
  pos = pos < 0 ? -pos : pos;
  const idx_t past_end = pos - last;
  return last - (past_end < 0 ? -past_end : past_end);
}

// Output positions along one dim whose gradient flows into input position x.
template <bool reflection, typename idx_t>
struct PadBackwardRange;

template <typename idx_t>
struct PadBackwardRange<true, idx_t> {
  idx_t count = 0;
  idx_t indices[3];

  PadBackwardRange(
      idx_t x,
      idx_t input_size,
      idx_t output_size,
      idx_t pad_left) {
    const idx_t center = x + pad_left;
    const idx_t left = pad_left - x;
    const idx_t right = pad_left + 2 * (input_size - 1) - x;
    if (center >= 0 && center < output_size) {
      indices[count++] = center;
    }
    if (x > 0 && left >= 0 && left < output_size) {
      indices[count++] = left;
    }
    if (x < input_size - 1 && right >= 0 && right < output_size) {
      indices[count++] = right;
    }
  }

  idx_t operator[](idx_t index) const {
    return indices[index];
  }
};

template <typename idx_t>
struct PadBackwardRange<false, idx_t> {
  idx_t first = 0;
  idx_t count = 0;

  PadBackwardRange(
      idx_t x,
      idx_t input_size,
      idx_t output_size,
      idx_t pad_left) {
    const idx_t center = x + pad_left;
    first = x == 0 ? 0 : max(center, idx_t(0));
    const idx_t end =
        x == input_size - 1 ? output_size : min(center + 1, output_size);
    count = max(end - first, idx_t(0));
  }

  idx_t operator[](idx_t index) const {
    return first + index;
  }
};

// Where this thread sits in the tensor it writes (result) and the base offset
// of the matching channel/batch in the tensor it reads (source).
template <typename idx_t>
struct PadPosition {
  idx_t y;
  idx_t z;
  idx_t source_base;
  idx_t result_base;
};

template <typename idx_t>
inline PadPosition<idx_t> pad_position(
    uint3 tid,
    uint depth,
    uint channels,
    constant ::c10::metal::array<idx_t, 5>& source_strides,
    constant ::c10::metal::array<idx_t, 5>& result_strides) {
  const idx_t y = tid.y;
  const idx_t z = tid.z % depth;
  const uint outer = tid.z / depth;
  const idx_t channel = safe_mod(outer, channels);
  const idx_t batch = outer / channels;
  return {
      y,
      z,
      channel * source_strides[3] + batch * source_strides[4],
      y * result_strides[1] + z * result_strides[2] +
          channel * result_strides[3] + batch * result_strides[4]};
}

template <typename T, typename idx_t, uint ndim, bool reflection>
kernel void pad_forward(
    device const T* input,
    device T* output,
    constant PadParams<idx_t>& params,
    uint3 tid [[thread_position_in_grid]],
    uint3 grid [[threads_per_grid]]) {
  const uint depth = ndim == 3 ? uint(params.output_sizes[2]) : 1;
  const auto position = pad_position<idx_t>(
      tid,
      depth,
      uint(params.channels),
      params.input_strides,
      params.output_strides);
  for (uint i = 0; i < ILP_PER_THREAD; ++i) {
    const idx_t x = idx_t(tid.x) + idx_t(i) * idx_t(grid.x);
    if (x >= params.output_sizes[0]) {
      break;
    }
    const idx_t pos[3] = {x, position.y, position.z};
    idx_t input_offset = position.source_base;
    for (uint dim = 0; dim < ndim; ++dim) {
      idx_t input_pos = pos[dim] - params.left_pad[dim];
      if IF_CONSTEXPR (reflection) {
        input_pos = reflect_index(input_pos, params.input_sizes[dim]);
      } else {
        input_pos = clamp(input_pos, idx_t(0), params.input_sizes[dim] - 1);
      }
      input_offset += input_pos * params.input_strides[dim];
    }
    output[position.result_base + x * params.output_strides[0]] =
        input[input_offset];
  }
}

template <typename T, typename idx_t, uint ndim, bool reflection>
kernel void pad_backward(
    device const T* grad_output,
    device T* grad_input,
    constant PadParams<idx_t>& params,
    uint3 tid [[thread_position_in_grid]],
    uint3 grid [[threads_per_grid]]) {
  const uint depth = ndim == 3 ? uint(params.input_sizes[2]) : 1;
  const auto position = pad_position<idx_t>(
      tid,
      depth,
      uint(params.channels),
      params.output_strides,
      params.input_strides);
  const PadBackwardRange<reflection, idx_t> ys(
      ndim >= 2 ? position.y : 0,
      ndim >= 2 ? params.input_sizes[1] : 1,
      ndim >= 2 ? params.output_sizes[1] : 1,
      ndim >= 2 ? params.left_pad[1] : 0);
  const PadBackwardRange<reflection, idx_t> zs(
      position.z,
      ndim == 3 ? params.input_sizes[2] : 1,
      ndim == 3 ? params.output_sizes[2] : 1,
      ndim == 3 ? params.left_pad[2] : 0);
  for (uint i = 0; i < ILP_PER_THREAD; ++i) {
    const idx_t x = idx_t(tid.x) + idx_t(i) * idx_t(grid.x);
    if (x >= params.input_sizes[0]) {
      break;
    }
    const PadBackwardRange<reflection, idx_t> xs(
        x, params.input_sizes[0], params.output_sizes[0], params.left_pad[0]);
    // Each input element owns its gradient; accumulate in float without
    // atomics.
    opmath_t<T> sum = 0;
    for (idx_t iz = 0; iz < zs.count; ++iz) {
      const idx_t z_offset =
          position.source_base + zs[iz] * params.output_strides[2];
      for (idx_t iy = 0; iy < ys.count; ++iy) {
        const idx_t y_offset = z_offset + ys[iy] * params.output_strides[1];
        for (idx_t ix = 0; ix < xs.count; ++ix) {
          sum += opmath_t<T>(
              grad_output[y_offset + xs[ix] * params.output_strides[0]]);
        }
      }
    }
    grad_input[position.result_base + x * params.input_strides[0]] = T(sum);
  }
}

#define INSTANTIATE_PAD(DTYPE, IDX, SUFFIX, NDIM, REFLECT, MODE, PASS)     \
  template [[host_name(#MODE "_pad" #NDIM "d_" #PASS "_" #DTYPE #SUFFIX)]] \
  kernel void pad_##PASS<DTYPE, IDX, NDIM, REFLECT>(                       \
      device const DTYPE*,                                                 \
      device DTYPE*,                                                       \
      constant PadParams<IDX>&,                                            \
      uint3,                                                               \
      uint3)

#define INSTANTIATE_PAD_DIM(DTYPE, IDX, SUFFIX, NDIM, PASS)          \
  INSTANTIATE_PAD(DTYPE, IDX, SUFFIX, NDIM, true, reflection, PASS); \
  INSTANTIATE_PAD(DTYPE, IDX, SUFFIX, NDIM, false, replication, PASS)

#define INSTANTIATE_PAD_IDX(DTYPE, IDX, SUFFIX, PASS)             \
  INSTANTIATE_PAD(DTYPE, IDX, SUFFIX, 1, true, reflection, PASS); \
  INSTANTIATE_PAD_DIM(DTYPE, IDX, SUFFIX, 2, PASS);               \
  INSTANTIATE_PAD_DIM(DTYPE, IDX, SUFFIX, 3, PASS)

#define INSTANTIATE_PAD_DTYPE(DTYPE, PASS)     \
  INSTANTIATE_PAD_IDX(DTYPE, int, _i32, PASS); \
  INSTANTIATE_PAD_IDX(DTYPE, long, _i64, PASS)

INSTANTIATE_PAD_DTYPE(float, forward);
INSTANTIATE_PAD_DTYPE(half, forward);
INSTANTIATE_PAD_DTYPE(bfloat, forward);
INSTANTIATE_PAD_DTYPE(float2, forward);
INSTANTIATE_PAD_DTYPE(half2, forward);
INSTANTIATE_PAD_DTYPE(long, forward);
INSTANTIATE_PAD_DTYPE(int, forward);
INSTANTIATE_PAD_DTYPE(short, forward);
INSTANTIATE_PAD_DTYPE(char, forward);
INSTANTIATE_PAD_DTYPE(uchar, forward);
INSTANTIATE_PAD_DTYPE(bool, forward);

INSTANTIATE_PAD_DTYPE(float, backward);
INSTANTIATE_PAD_DTYPE(half, backward);
INSTANTIATE_PAD_DTYPE(bfloat, backward);
INSTANTIATE_PAD_DTYPE(float2, backward);
INSTANTIATE_PAD_DTYPE(half2, backward);

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
