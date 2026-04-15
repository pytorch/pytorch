#include <ATen/native/mps/kernels/GridSampler.h>
#include <ATen/native/mps/kernels/SamplingHelpers.h>
#include <c10/metal/atomic.h>
#include <c10/metal/utils.h>
#include <metal_array>
#include <metal_stdlib>

using namespace metal;
using namespace c10::metal;

struct GridSamplerOffsets {
  int32_t output;
  int32_t input;
  int32_t grid;

  GridSamplerOffsets() : output(0), input(0), grid(0) {}
};

// Find offsets into the tensors that this thread will operate on,
// based on the thread ID.
static GridSamplerOffsets find_grid_sampler_offsets(
    constant int32_t* output_sizes,
    constant int32_t* output_strides,
    constant int32_t* input_strides,
    constant int32_t* grid_strides,
    int32_t sampler_dims,
    uint tid) {
  auto dims = sampler_dims + 2;
  auto output_idx = static_cast<int32_t>(tid);
  GridSamplerOffsets offsets;

  for (auto dim = dims - 1; dim >= 0; dim--) {
    auto dim_idx = output_idx % output_sizes[dim];
    output_idx = output_idx / output_sizes[dim];

    // Select the output element that this thread will calculate.
    // output shape:
    //   2 sampler dims: (N, C, Hout, Wout)
    //   3 sampler dims: (N, C, Dout, Hout, Wout)
    offsets.output += output_strides[dim] * dim_idx;

    // Select the batch and channel for the input.
    // input shape:
    //   2 sampler dims: (N, C, Hin, Win)
    //   3 sampler dims: (N, C, Din, Hin, Win)
    if (dim < 2) {
      offsets.input += input_strides[dim] * dim_idx;
    }

    // Select the grid coordinates for the output element.
    // grid shape:
    //   2 sampler dims: (N, Hout, Wout, 2)
    //   3 sampler dims: (N, Dout, Hout, Wout, 3)
    if (dim == 0) {
      offsets.grid += grid_strides[dim] * dim_idx;
    } else if (dim >= 2) {
      offsets.grid += grid_strides[dim - 1] * dim_idx;
    }
  }

  return offsets;
}

// Mod function which gives positive output when `a` is negative
static int32_t mod(int32_t a, int32_t b) {
  auto r = a % b;
  return r + (r < 0 ? b : 0);
}

// Sentinel index value to indicate zero padding
constant int32_t IDX_ZERO = -1;

// Unnormalize grid coordinate from [-1, 1] to pixel space
static float grid_sampler_unnormalize(
    float coord,
    int32_t size,
    bool align_corners) {
  if (align_corners) {
    return ((coord + 1) / 2) * (size - 1);
  } else {
    return ((coord + 1) * size - 1) / 2;
  }
}

// Clip coordinates to [0, max_val]
static float clip_coordinates(float in, float max_val) {
  return ::metal::clamp(in, 0.0f, max_val);
}

// Reflect coordinates for reflection padding
template <typename T>
static T reflect_coordinates(T in, T low, T high) {
  if (low == high) {
    return 0;
  }
  auto span = high - low;
  in = fabs(in - low);
  auto extra = fmod(in, span);
  int32_t flips = static_cast<int32_t>(floor(in / span));
  return (flips % 2 == 0) ? (extra + low) : (span - extra + low);
}

// Padding functors: each encapsulates the padding logic for integer indices
// (pad) and float source coordinates (compute_source).
struct PadZeros {
  static constant constexpr bool checks_bounds = true;

  static int32_t pad(int32_t idx, int32_t input_size, bool) {
    return (idx < 0 || idx >= input_size) ? IDX_ZERO : idx;
  }

  static float apply_padding(float coord, int32_t, bool) {
    return coord;
  }

  static float compute_source(float coord, int32_t size, bool align_corners) {
    return grid_sampler_unnormalize(coord, size, align_corners);
  }
};

struct PadBorder {
  static constant constexpr bool checks_bounds = false;

  static int32_t pad(int32_t idx, int32_t input_size, bool) {
    return clamp(idx, 0, input_size - 1);
  }

  static float apply_padding(float coord, int32_t size, bool) {
    return clip_coordinates(coord, size - 1.0f);
  }

  static float compute_source(float coord, int32_t size, bool align_corners) {
    return apply_padding(
        grid_sampler_unnormalize(coord, size, align_corners),
        size,
        align_corners);
  }
};

struct PadReflection {
  static constant constexpr bool checks_bounds = false;

  static int32_t pad(int32_t idx, int32_t input_size, bool align_corners) {
    auto scale_length = align_corners ? (input_size - 1) : input_size;
    auto idx_mod = mod(idx, scale_length);
    auto idx_mod_reverse = (input_size - 1) - idx_mod;
    bool is_reverse = (abs(idx - idx_mod) / scale_length) % 2 == 1;
    return is_reverse ? idx_mod_reverse : idx_mod;
  }

  static float apply_padding(float coord, int32_t size, bool align_corners) {
    if (align_corners) {
      coord = reflect_coordinates(coord, 0.0f, static_cast<float>(size - 1));
    } else {
      coord = reflect_coordinates(coord, -0.5f, size - 0.5f);
    }
    return clip_coordinates(coord, size - 1.0f);
  }

  static float compute_source(float coord, int32_t size, bool align_corners) {
    return apply_padding(
        grid_sampler_unnormalize(coord, size, align_corners),
        size,
        align_corners);
  }
};

// 2D Bilinear interpolation
template <typename Pad, typename T>
static T interpolate_bilinear_2d(
    constant T* input,
    float ix,
    float iy,
    int32_t inp_H,
    int32_t inp_W,
    int32_t inp_sH,
    int32_t inp_sW,
    bool align_corners) {
  ix = grid_sampler_unnormalize(ix, inp_W, align_corners);
  iy = grid_sampler_unnormalize(iy, inp_H, align_corners);

  int32_t ix_nw = static_cast<int32_t>(floor(ix));
  int32_t iy_nw = static_cast<int32_t>(floor(iy));
  int32_t ix_ne = ix_nw + 1;
  int32_t iy_ne = iy_nw;
  int32_t ix_sw = ix_nw;
  int32_t iy_sw = iy_nw + 1;
  int32_t ix_se = ix_nw + 1;
  int32_t iy_se = iy_nw + 1;

  const auto nw = (ix_se - ix) * (iy_se - iy);
  const auto ne = (ix - ix_sw) * (iy_sw - iy);
  const auto sw = (ix_ne - ix) * (iy - iy_ne);
  const auto se = (ix - ix_nw) * (iy - iy_nw);

  int32_t iy_nw_p = Pad::pad(iy_nw, inp_H, align_corners);
  int32_t ix_nw_p = Pad::pad(ix_nw, inp_W, align_corners);
  int32_t iy_ne_p = Pad::pad(iy_ne, inp_H, align_corners);
  int32_t ix_ne_p = Pad::pad(ix_ne, inp_W, align_corners);
  int32_t iy_sw_p = Pad::pad(iy_sw, inp_H, align_corners);
  int32_t ix_sw_p = Pad::pad(ix_sw, inp_W, align_corners);
  int32_t iy_se_p = Pad::pad(iy_se, inp_H, align_corners);
  int32_t ix_se_p = Pad::pad(ix_se, inp_W, align_corners);

  opmath_t<T> out_acc = 0;
  if (iy_nw_p != IDX_ZERO && ix_nw_p != IDX_ZERO) {
    out_acc += input[iy_nw_p * inp_sH + ix_nw_p * inp_sW] * nw;
  }
  if (iy_ne_p != IDX_ZERO && ix_ne_p != IDX_ZERO) {
    out_acc += input[iy_ne_p * inp_sH + ix_ne_p * inp_sW] * ne;
  }
  if (iy_sw_p != IDX_ZERO && ix_sw_p != IDX_ZERO) {
    out_acc += input[iy_sw_p * inp_sH + ix_sw_p * inp_sW] * sw;
  }
  if (iy_se_p != IDX_ZERO && ix_se_p != IDX_ZERO) {
    out_acc += input[iy_se_p * inp_sH + ix_se_p * inp_sW] * se;
  }

  return static_cast<T>(out_acc);
}

// 2D Nearest neighbor interpolation
template <typename Pad, typename T>
static T interpolate_nearest_2d(
    constant T* input,
    opmath_t<T> ix,
    opmath_t<T> iy,
    int32_t inp_H,
    int32_t inp_W,
    int32_t inp_sH,
    int32_t inp_sW,
    bool align_corners) {
  ix = Pad::compute_source(ix, inp_W, align_corners);
  iy = Pad::compute_source(iy, inp_H, align_corners);

  int32_t ix_nearest = static_cast<int32_t>(rint(ix));
  int32_t iy_nearest = static_cast<int32_t>(rint(iy));

  if (Pad::checks_bounds) {
    if (ix_nearest < 0 || ix_nearest >= inp_W || iy_nearest < 0 ||
        iy_nearest >= inp_H) {
      return static_cast<T>(0);
    }
  }

  return input[iy_nearest * inp_sH + ix_nearest * inp_sW];
}

// Helper to get bounded value for bicubic interpolation
template <typename Pad, typename T>
static opmath_t<T> get_bicubic_value(
    constant T* input,
    int32_t y,
    int32_t x,
    int32_t inp_H,
    int32_t inp_W,
    int32_t inp_sH,
    int32_t inp_sW,
    bool align_corners) {
  int32_t y_p = Pad::pad(y, inp_H, align_corners);
  int32_t x_p = Pad::pad(x, inp_W, align_corners);
  if (y_p == IDX_ZERO || x_p == IDX_ZERO) {
    return 0;
  }
  return input[y_p * inp_sH + x_p * inp_sW];
}

// 2D Bicubic interpolation
template <typename Pad, typename T>
static T interpolate_bicubic_2d(
    constant T* input,
    float ix,
    float iy,
    int32_t inp_H,
    int32_t inp_W,
    int32_t inp_sH,
    int32_t inp_sW,
    bool align_corners) {
  ix = grid_sampler_unnormalize(ix, inp_W, align_corners);
  iy = grid_sampler_unnormalize(iy, inp_H, align_corners);

  auto ix_nw = floor(ix);
  auto iy_nw = floor(iy);
  auto tx = ix - ix_nw;
  auto ty = iy - iy_nw;

  opmath_t<T> coefficients[4];
  int32_t ix_nw_i = static_cast<int32_t>(ix_nw);
  int32_t iy_nw_i = static_cast<int32_t>(iy_nw);

  for (int32_t i = 0; i < 4; ++i) {
    coefficients[i] = cubic_interp1d(
        get_bicubic_value<Pad, T>(
            input,
            iy_nw_i - 1 + i,
            ix_nw_i - 1,
            inp_H,
            inp_W,
            inp_sH,
            inp_sW,
            align_corners),
        get_bicubic_value<Pad, T>(
            input,
            iy_nw_i - 1 + i,
            ix_nw_i + 0,
            inp_H,
            inp_W,
            inp_sH,
            inp_sW,
            align_corners),
        get_bicubic_value<Pad, T>(
            input,
            iy_nw_i - 1 + i,
            ix_nw_i + 1,
            inp_H,
            inp_W,
            inp_sH,
            inp_sW,
            align_corners),
        get_bicubic_value<Pad, T>(
            input,
            iy_nw_i - 1 + i,
            ix_nw_i + 2,
            inp_H,
            inp_W,
            inp_sH,
            inp_sW,
            align_corners),
        tx);
  }

  return static_cast<T>(cubic_interp1d(
      coefficients[0], coefficients[1], coefficients[2], coefficients[3], ty));
}

// Interpolation functors for the 2D kernel.
// Each wraps the respective interpolation function with the padding already
// baked in via the Pad template parameter.
template <typename Pad>
struct Bilinear2D {
  template <typename T>
  static T interpolate(
      constant T* input,
      opmath_t<T> ix,
      opmath_t<T> iy,
      int32_t inp_H,
      int32_t inp_W,
      int32_t inp_sH,
      int32_t inp_sW,
      bool align_corners) {
    return interpolate_bilinear_2d<Pad, T>(
        input, ix, iy, inp_H, inp_W, inp_sH, inp_sW, align_corners);
  }
};

template <typename Pad>
struct Nearest2D {
  template <typename T>
  static T interpolate(
      constant T* input,
      opmath_t<T> ix,
      opmath_t<T> iy,
      int32_t inp_H,
      int32_t inp_W,
      int32_t inp_sH,
      int32_t inp_sW,
      bool align_corners) {
    return interpolate_nearest_2d<Pad, T>(
        input, ix, iy, inp_H, inp_W, inp_sH, inp_sW, align_corners);
  }
};

template <typename Pad>
struct Bicubic2D {
  template <typename T>
  static T interpolate(
      constant T* input,
      float ix,
      float iy,
      int32_t inp_H,
      int32_t inp_W,
      int32_t inp_sH,
      int32_t inp_sW,
      bool align_corners) {
    return interpolate_bicubic_2d<Pad, T>(
        input, ix, iy, inp_H, inp_W, inp_sH, inp_sW, align_corners);
  }
};

// 2D grid sampler kernel
template <typename Interp, typename T>
kernel void grid_sampler_2d(
    device T* output [[buffer(0)]],
    constant T* input [[buffer(1)]],
    constant T* grid [[buffer(2)]],
    constant GridSamplerParams<4>& params [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  auto C = params.output_sizes[1];
  auto out_H = params.output_sizes[2];
  auto out_W = params.output_sizes[3];
  auto inp_H = params.input_sizes[2];
  auto inp_W = params.input_sizes[3];

  auto out_sN = params.output_strides[0];
  auto out_sC = params.output_strides[1];
  auto out_sH = params.output_strides[2];
  auto out_sW = params.output_strides[3];
  auto inp_sN = params.input_strides[0];
  auto inp_sC = params.input_strides[1];
  auto inp_sH = params.input_strides[2];
  auto inp_sW = params.input_strides[3];
  auto grid_sN = params.grid_strides[0];
  auto grid_sH = params.grid_strides[1];
  auto grid_sW = params.grid_strides[2];
  auto grid_sCoor = params.grid_strides[3];

  auto align_corners = params.align_corners;

  int32_t w = tid % out_W;
  int32_t h = (tid / out_W) % out_H;
  int32_t n = tid / (out_H * out_W);

  auto grid_ptr = grid + n * grid_sN + h * grid_sH + w * grid_sW;
  opmath_t<T> ix = static_cast<opmath_t<T>>(grid_ptr[0]);
  opmath_t<T> iy = static_cast<opmath_t<T>>(grid_ptr[grid_sCoor]);

  auto inp_ptr_N = input + n * inp_sN;
  auto out_ptr_NCHW = output + n * out_sN + h * out_sH + w * out_sW;

  for (int32_t c = 0; c < C; ++c) {
    auto result = Interp::template interpolate<T>(
        inp_ptr_N, ix, iy, inp_H, inp_W, inp_sH, inp_sW, align_corners);
    out_ptr_NCHW[c * out_sC] = result;
    inp_ptr_N += inp_sC;
  }
}

template <int32_t dims, typename T>
T get_tensor_val(
    constant T* input,
    constant int32_t* input_strides,
    int32_t indices[dims]) {
  bool found_idx_zero = false;
  int32_t offset = 0;

  for (auto dim = 0; dim < dims; dim++) {
    auto idx = indices[dim];
    found_idx_zero = found_idx_zero || (idx == IDX_ZERO);
    offset += (found_idx_zero ? 0 : idx) * input_strides[dim];
  }

  return found_idx_zero ? 0 : input[offset];
}

// This function performs 3D linear interpolation for one value. One way to
// think of how this works is to imagine a unit cube where each corner of the
// cube has one scalar value associated with it. Inside the cube, the values
// change linearly, so the gradient is constant. The values associated with each
// corner are given by the `input`, indexed at all eight different combinations
// of the `left_indices` and `right_indices`. Given a 3D coordinate anywhere
// within the cube, specified by the `scales` argument, we must calculate the
// value associated with that position.
template <typename T>
T interpolate_linear_3d(
    constant T* input,
    constant int32_t* input_strides,
    int32_t left_indices[3],
    int32_t right_indices[3],
    opmath_t<T> scales[3]) {
  int32_t a_idx[3] = {left_indices[0], left_indices[1], left_indices[2]};
  int32_t b_idx[3] = {left_indices[0], left_indices[1], right_indices[2]};
  int32_t c_idx[3] = {left_indices[0], right_indices[1], left_indices[2]};
  int32_t d_idx[3] = {left_indices[0], right_indices[1], right_indices[2]};
  int32_t e_idx[3] = {right_indices[0], left_indices[1], left_indices[2]};
  int32_t f_idx[3] = {right_indices[0], left_indices[1], right_indices[2]};
  int32_t g_idx[3] = {right_indices[0], right_indices[1], left_indices[2]};
  int32_t h_idx[3] = {right_indices[0], right_indices[1], right_indices[2]};
  auto a =
      static_cast<opmath_t<T>>(get_tensor_val<3>(input, input_strides, a_idx));
  auto b =
      static_cast<opmath_t<T>>(get_tensor_val<3>(input, input_strides, b_idx));
  auto c =
      static_cast<opmath_t<T>>(get_tensor_val<3>(input, input_strides, c_idx));
  auto d =
      static_cast<opmath_t<T>>(get_tensor_val<3>(input, input_strides, d_idx));
  auto e =
      static_cast<opmath_t<T>>(get_tensor_val<3>(input, input_strides, e_idx));
  auto f =
      static_cast<opmath_t<T>>(get_tensor_val<3>(input, input_strides, f_idx));
  auto g =
      static_cast<opmath_t<T>>(get_tensor_val<3>(input, input_strides, g_idx));
  auto h =
      static_cast<opmath_t<T>>(get_tensor_val<3>(input, input_strides, h_idx));

  auto scale0_right = scales[0];
  auto scale1_right = scales[1];
  auto scale2_right = scales[2];
  auto scale0_left = 1 - scale0_right;
  auto scale1_left = 1 - scale1_right;
  auto scale2_left = 1 - scale2_right;

  return static_cast<T>(
      scale0_left * scale1_left * scale2_left * a +
      scale0_left * scale1_left * scale2_right * b +
      scale0_left * scale1_right * scale2_left * c +
      scale0_left * scale1_right * scale2_right * d +
      scale0_right * scale1_left * scale2_left * e +
      scale0_right * scale1_left * scale2_right * f +
      scale0_right * scale1_right * scale2_left * g +
      scale0_right * scale1_right * scale2_right * h);
}

// 3D bilinear sampling for a single output element.
template <typename Pad, typename T>
void grid_sampler_3d_single_element(
    device T* output,
    constant T* input,
    constant T* coords,
    int32_t dims,
    constant int32_t* input_sizes,
    constant int32_t* input_strides,
    int32_t coord_stride,
    bool align_corners) {
  int32_t left_indices[3];
  int32_t right_indices[3];
  opmath_t<T> scales[3];

  for (auto coord_dim = 0; coord_dim < dims; coord_dim++) {
    auto input_dim = dims - coord_dim - 1;
    auto input_size = input_sizes[input_dim];
    auto coord = static_cast<opmath_t<T>>(coords[coord_dim * coord_stride]);

    if (!align_corners) {
      auto corner_alignment_factor = static_cast<opmath_t<T>>(input_size) /
          static_cast<opmath_t<T>>(input_size - 1);
      coord = coord * corner_alignment_factor;
    }

    coord = (coord + 1) * (static_cast<opmath_t<T>>(input_size - 1) / 2);

    auto left_idx = static_cast<int32_t>(floor(coord));
    auto right_idx = static_cast<int32_t>(ceil(coord));
    left_indices[input_dim] = Pad::pad(left_idx, input_size, align_corners);
    right_indices[input_dim] = Pad::pad(right_idx, input_size, align_corners);
    scales[input_dim] = coord - left_idx;
  }

  *output = interpolate_linear_3d(
      input, input_strides, left_indices, right_indices, scales);
}

template <typename Pad, typename T>
kernel void grid_sampler_3d(
    device T* output [[buffer(0)]],
    constant T* input [[buffer(1)]],
    constant T* grid [[buffer(2)]],
    constant GridSamplerParams<5>& params [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  auto output_sizes = params.output_sizes.data();
  auto output_strides = params.output_strides.data();
  auto input_sizes = params.input_sizes.data();
  auto input_strides = params.input_strides.data();
  auto grid_strides = params.grid_strides.data();
  auto sampler_dims = params.sampler_dims;

  auto offsets = find_grid_sampler_offsets(
      output_sizes,
      output_strides,
      input_strides,
      grid_strides,
      sampler_dims,
      tid);

  output += offsets.output;
  input += offsets.input;
  auto coords = grid + offsets.grid;

  input_sizes += 2;
  input_strides += 2;
  auto coord_stride = grid_strides[sampler_dims + 1];

  grid_sampler_3d_single_element<Pad>(
      output,
      input,
      coords,
      sampler_dims,
      input_sizes,
      input_strides,
      coord_stride,
      params.align_corners);
}

#define REGISTER_GRID_SAMPLER_2D(DTYPE, INTERP, INAME, PAD, PNAME)      \
  template [[host_name("grid_sampler_2d_" INAME "_" PNAME "_" #DTYPE)]] \
  kernel void grid_sampler_2d<INTERP<PAD>, DTYPE>(                      \
      device DTYPE * output [[buffer(0)]],                              \
      constant DTYPE * input [[buffer(1)]],                             \
      constant DTYPE * grid [[buffer(2)]],                              \
      constant GridSamplerParams<4> & params [[buffer(3)]],             \
      uint tid [[thread_position_in_grid]]);

#define REGISTER_GRID_SAMPLER_2D_INTERP(DTYPE, INTERP, INAME)         \
  REGISTER_GRID_SAMPLER_2D(DTYPE, INTERP, INAME, PadZeros, "zeros")   \
  REGISTER_GRID_SAMPLER_2D(DTYPE, INTERP, INAME, PadBorder, "border") \
  REGISTER_GRID_SAMPLER_2D(DTYPE, INTERP, INAME, PadReflection, "reflection")

#define REGISTER_GRID_SAMPLER_3D(DTYPE, PAD, PNAME)           \
  template [[host_name("grid_sampler_3d_" PNAME "_" #DTYPE)]] \
  kernel void grid_sampler_3d<PAD, DTYPE>(                    \
      device DTYPE * output [[buffer(0)]],                    \
      constant DTYPE * input [[buffer(1)]],                   \
      constant DTYPE * grid [[buffer(2)]],                    \
      constant GridSamplerParams<5> & params [[buffer(3)]],   \
      uint tid [[thread_position_in_grid]]);

#define REGISTER_GRID_SAMPLER_OPS(DTYPE)                         \
  REGISTER_GRID_SAMPLER_2D_INTERP(DTYPE, Bilinear2D, "bilinear") \
  REGISTER_GRID_SAMPLER_2D_INTERP(DTYPE, Nearest2D, "nearest")   \
  REGISTER_GRID_SAMPLER_2D_INTERP(DTYPE, Bicubic2D, "bicubic")   \
  REGISTER_GRID_SAMPLER_3D(DTYPE, PadZeros, "zeros")             \
  REGISTER_GRID_SAMPLER_3D(DTYPE, PadBorder, "border")           \
  REGISTER_GRID_SAMPLER_3D(DTYPE, PadReflection, "reflection")

REGISTER_GRID_SAMPLER_OPS(float);
REGISTER_GRID_SAMPLER_OPS(half);
REGISTER_GRID_SAMPLER_OPS(bfloat);

// ========== Backward kernels ==========

// Each _set_grad function returns float2{coord, grad} where grad is
// d(output_coord)/d(input_coord), used to chain-rule through the
// coordinate transform in the backward pass.

static float2 grid_sampler_unnormalize_set_grad(
    float coord,
    int32_t size,
    bool align_corners) {
  float grad = align_corners ? (size - 1) / 2.0f : size / 2.0f;
  return {grid_sampler_unnormalize(coord, size, align_corners), grad};
}

static float2 clip_coordinates_set_grad(float in, float max_val) {
  if (in <= 0.0f) {
    return {0.0f, 0.0f};
  }
  if (in >= max_val) {
    return {max_val, 0.0f};
  }
  return {in, 1.0f};
}

static float2 reflect_coordinates_set_grad(float in, float low, float high) {
  if (low == high) {
    return {0.0f, 0.0f};
  }
  int grad_in_mult = 1;
  float span = high - low;
  in = in - low;
  if (in < 0.0f) {
    grad_in_mult = -1;
    in = -in;
  }
  float extra = fmod(in, span);
  int flips = static_cast<int>(floor(in / span));
  if (flips % 2 == 0) {
    return {extra + low, static_cast<float>(grad_in_mult)};
  }
  return {span - extra + low, static_cast<float>(-grad_in_mult)};
}

// Combines unnormalize + padding, returns {source_index, grad_multiplier}.
template <typename Pad>
static float2 compute_source_index_set_grad(
    float coord,
    int32_t size,
    bool align_corners);

template <>
float2 compute_source_index_set_grad<PadZeros>(
    float coord,
    int32_t size,
    bool align_corners) {
  return grid_sampler_unnormalize_set_grad(coord, size, align_corners);
}

template <>
float2 compute_source_index_set_grad<PadBorder>(
    float coord,
    int32_t size,
    bool align_corners) {
  float2 unnorm = grid_sampler_unnormalize_set_grad(coord, size, align_corners);
  float2 clip = clip_coordinates_set_grad(unnorm.x, size - 1.0f);
  return {clip.x, unnorm.y * clip.y};
}

template <>
float2 compute_source_index_set_grad<PadReflection>(
    float coord,
    int32_t size,
    bool align_corners) {
  float2 unnorm = grid_sampler_unnormalize_set_grad(coord, size, align_corners);
  float2 refl;
  if (align_corners) {
    refl = reflect_coordinates_set_grad(
        unnorm.x, 0.0f, static_cast<float>(size - 1));
  } else {
    refl = reflect_coordinates_set_grad(unnorm.x, -0.5f, size - 0.5f);
  }
  float2 clip = clip_coordinates_set_grad(refl.x, size - 1.0f);
  return {clip.x, unnorm.y * refl.y * clip.y};
}

// Runtime-dispatch versions for kernels where Pad is not in the hot loop.
static float2 compute_source_index_set_grad(
    float coord,
    int32_t size,
    bool align_corners,
    int32_t padding_mode) {
  switch (padding_mode) {
    case 1:
      return compute_source_index_set_grad<PadBorder>(
          coord, size, align_corners);
    case 2:
      return compute_source_index_set_grad<PadReflection>(
          coord, size, align_corners);
    default:
      return compute_source_index_set_grad<PadZeros>(
          coord, size, align_corners);
  }
}

static float compute_source(
    float coord,
    int32_t size,
    bool align_corners,
    int32_t padding_mode) {
  switch (padding_mode) {
    case 1:
      return PadBorder::compute_source(coord, size, align_corners);
    case 2:
      return PadReflection::compute_source(coord, size, align_corners);
    default:
      return PadZeros::compute_source(coord, size, align_corners);
  }
}

static bool within_bounds_2d(int2 pos, int2 size) {
  return pos.x >= 0 && pos.x < size.x && pos.y >= 0 && pos.y < size.y;
}

// Atomic safe add for grad_input
template <typename T>
static void safe_add_2d_atomic(
    device AtomicType_t<T>* data,
    int2 pos,
    int2 stride,
    int2 size,
    opmath_t<T> delta,
    long NC_offset) {
  if (within_bounds_2d(pos, size)) {
    AtomicType<T>::atomic_add(
        data,
        NC_offset + pos.y * stride.y + pos.x * stride.x,
        static_cast<T>(delta));
  }
}

// Apply padding and convert to bounded int2 position (for bicubic backward
// where coordinates are already in pixel space).
template <typename Pad>
static int2 apply_padding_2d(float x, float y, int2 size, bool align_corners) {
  return {
      static_cast<int32_t>(Pad::apply_padding(x, size.x, align_corners)),
      static_cast<int32_t>(Pad::apply_padding(y, size.y, align_corners))};
}

// Get bounded value for bicubic backward
template <typename Pad, typename T>
static opmath_t<T> get_value_bounded_backward(
    constant T* data,
    float x,
    float y,
    int2 size,
    int2 stride,
    bool align_corners) {
  int2 pos = apply_padding_2d<Pad>(x, y, size, align_corners);
  if (within_bounds_2d(pos, size)) {
    return static_cast<opmath_t<T>>(data[pos.y * stride.y + pos.x * stride.x]);
  }
  return 0;
}

// Add value at bounded coordinates for bicubic backward grad_input
template <typename Pad, typename T>
static void add_value_bounded_backward(
    device AtomicType_t<T>* data,
    float x,
    float y,
    int2 size,
    int2 stride,
    opmath_t<T> delta,
    bool align_corners,
    long NC_offset) {
  int2 pos = apply_padding_2d<Pad>(x, y, size, align_corners);
  safe_add_2d_atomic<T>(data, pos, stride, size, delta, NC_offset);
}

// Get cubic coefficients gradient
template <typename T>
static void get_cubic_coefficients_grad(T coeffs[4], T t) {
  T A = static_cast<T>(-0.75);
  T x;
  x = -1 - t;
  coeffs[0] = (-3 * A * x - 10 * A) * x - 8 * A;
  x = -t;
  coeffs[1] = (-3 * (A + 2) * x - 2 * (A + 3)) * x;
  x = 1 - t;
  coeffs[2] = (3 * (A + 2) * x - 2 * (A + 3)) * x;
  x = 2 - t;
  coeffs[3] = (3 * A * x - 10 * A) * x + 8 * A;
}

// Common preamble for all backward kernels: decompose thread id into n,h,w
// and read grid coordinates.
template <typename T>
struct BackwardPreamble {
  int32_t n, h, w;
  float x, y;

  BackwardPreamble(
      constant T* grid,
      constant GridSamplerBackwardParams<4>& params,
      uint tid) {
    auto out_H = params.forward.output_sizes[2];
    auto out_W = params.forward.output_sizes[3];
    w = tid % out_W;
    h = (tid / out_W) % out_H;
    n = tid / (out_H * out_W);
    auto grid_offset = n * params.forward.grid_strides[0] +
        h * params.forward.grid_strides[1] + w * params.forward.grid_strides[2];
    x = static_cast<float>(grid[grid_offset]);
    y = static_cast<float>(grid[grid_offset + params.forward.grid_strides[3]]);
  }
};

// Bilinear backward kernel for grad_input
template <typename T>
kernel void grid_sampler_2d_backward_bilinear_input(
    device AtomicType_t<T>* grad_input [[buffer(0)]],
    constant T* grad_output [[buffer(1)]],
    constant T* grid [[buffer(2)]],
    constant GridSamplerBackwardParams<4>& params [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  BackwardPreamble<T> p(grid, params, tid);
  auto C = params.forward.input_sizes[1];
  int2 inp_size = {
      params.forward.input_sizes[3], params.forward.input_sizes[2]};
  int2 gInp_stride = {
      params.grad_input_strides[3], params.grad_input_strides[2]};
  auto gOut_sN = params.grad_output_strides[0];
  auto gOut_sC = params.grad_output_strides[1];
  auto gOut_sH = params.grad_output_strides[2];
  auto gOut_sW = params.grad_output_strides[3];
  auto gInp_sN = params.grad_input_strides[0];
  auto gInp_sC = params.grad_input_strides[1];

  float ix = compute_source(
      p.x, inp_size.x, params.forward.align_corners, params.padding_mode);
  float iy = compute_source(
      p.y, inp_size.y, params.forward.align_corners, params.padding_mode);

  int32_t ix_nw = static_cast<int32_t>(floor(ix));
  int32_t iy_nw = static_cast<int32_t>(floor(iy));
  int32_t ix_ne = ix_nw + 1;
  int32_t iy_ne = iy_nw;
  int32_t ix_sw = ix_nw;
  int32_t iy_sw = iy_nw + 1;
  int32_t ix_se = ix_nw + 1;
  int32_t iy_se = iy_nw + 1;

  float nw = (ix_se - ix) * (iy_se - iy);
  float ne = (ix - ix_sw) * (iy_sw - iy);
  float sw = (ix_ne - ix) * (iy - iy_ne);
  float se = (ix - ix_nw) * (iy - iy_nw);

  auto gOut_ptr_NCHW =
      grad_output + p.n * gOut_sN + p.h * gOut_sH + p.w * gOut_sW;
  long NC_offset = p.n * gInp_sN;

  for (int32_t c = 0; c < C;
       ++c, NC_offset += gInp_sC, gOut_ptr_NCHW += gOut_sC) {
    opmath_t<T> gOut = static_cast<opmath_t<T>>(*gOut_ptr_NCHW);
    safe_add_2d_atomic<T>(
        grad_input,
        {ix_nw, iy_nw},
        gInp_stride,
        inp_size,
        nw * gOut,
        NC_offset);
    safe_add_2d_atomic<T>(
        grad_input,
        {ix_ne, iy_ne},
        gInp_stride,
        inp_size,
        ne * gOut,
        NC_offset);
    safe_add_2d_atomic<T>(
        grad_input,
        {ix_sw, iy_sw},
        gInp_stride,
        inp_size,
        sw * gOut,
        NC_offset);
    safe_add_2d_atomic<T>(
        grad_input,
        {ix_se, iy_se},
        gInp_stride,
        inp_size,
        se * gOut,
        NC_offset);
  }
}

// Bilinear backward kernel for grad_grid
template <typename T>
kernel void grid_sampler_2d_backward_bilinear_grid(
    device T* grad_grid [[buffer(0)]],
    constant T* grad_output [[buffer(1)]],
    constant T* input [[buffer(2)]],
    constant T* grid [[buffer(3)]],
    constant GridSamplerBackwardParams<4>& params [[buffer(4)]],
    uint tid [[thread_position_in_grid]]) {
  BackwardPreamble<T> p(grid, params, tid);
  auto C = params.forward.input_sizes[1];
  int2 inp_size = {
      params.forward.input_sizes[3], params.forward.input_sizes[2]};
  int2 inp_stride = {
      params.forward.input_strides[3], params.forward.input_strides[2]};
  auto inp_sN = params.forward.input_strides[0];
  auto inp_sC = params.forward.input_strides[1];
  auto gOut_sN = params.grad_output_strides[0];
  auto gOut_sC = params.grad_output_strides[1];
  auto gOut_sH = params.grad_output_strides[2];
  auto gOut_sW = params.grad_output_strides[3];
  auto gGrid_sW = params.grad_grid_sW;

  // .x = source index, .y = gradient multiplier from coordinate transform
  float2 ix = compute_source_index_set_grad(
      p.x, inp_size.x, params.forward.align_corners, params.padding_mode);
  float2 iy = compute_source_index_set_grad(
      p.y, inp_size.y, params.forward.align_corners, params.padding_mode);

  int32_t ix_nw = static_cast<int32_t>(floor(ix.x));
  int32_t iy_nw = static_cast<int32_t>(floor(iy.x));

  opmath_t<T> gix = 0, giy = 0;
  auto gOut_ptr_NCHW =
      grad_output + p.n * gOut_sN + p.h * gOut_sH + p.w * gOut_sW;
  auto inp_ptr_NC = input + p.n * inp_sN;

  for (int32_t c = 0; c < C;
       ++c, inp_ptr_NC += inp_sC, gOut_ptr_NCHW += gOut_sC) {
    opmath_t<T> gOut = static_cast<opmath_t<T>>(*gOut_ptr_NCHW);

    if (within_bounds_2d({ix_nw, iy_nw}, inp_size)) {
      opmath_t<T> nw_val =
          inp_ptr_NC[iy_nw * inp_stride.y + ix_nw * inp_stride.x];
      gix -= nw_val * (iy_nw + 1 - iy.x) * gOut;
      giy -= nw_val * (ix_nw + 1 - ix.x) * gOut;
    }
    if (within_bounds_2d({ix_nw + 1, iy_nw}, inp_size)) {
      opmath_t<T> ne_val =
          inp_ptr_NC[iy_nw * inp_stride.y + (ix_nw + 1) * inp_stride.x];
      gix += ne_val * (iy_nw + 1 - iy.x) * gOut;
      giy -= ne_val * (ix.x - ix_nw) * gOut;
    }
    if (within_bounds_2d({ix_nw, iy_nw + 1}, inp_size)) {
      opmath_t<T> sw_val =
          inp_ptr_NC[(iy_nw + 1) * inp_stride.y + ix_nw * inp_stride.x];
      gix -= sw_val * (iy.x - iy_nw) * gOut;
      giy += sw_val * (ix_nw + 1 - ix.x) * gOut;
    }
    if (within_bounds_2d({ix_nw + 1, iy_nw + 1}, inp_size)) {
      opmath_t<T> se_val =
          inp_ptr_NC[(iy_nw + 1) * inp_stride.y + (ix_nw + 1) * inp_stride.x];
      gix += se_val * (iy.x - iy_nw) * gOut;
      giy += se_val * (ix.x - ix_nw) * gOut;
    }
  }

  auto gGrid_ptr_NHW = grad_grid + tid * gGrid_sW;
  gGrid_ptr_NHW[0] = static_cast<T>(ix.y * gix);
  gGrid_ptr_NHW[1] = static_cast<T>(iy.y * giy);
}

// Nearest backward kernel for grad_input
template <typename T>
kernel void grid_sampler_2d_backward_nearest_input(
    device AtomicType_t<T>* grad_input [[buffer(0)]],
    constant T* grad_output [[buffer(1)]],
    constant T* grid [[buffer(2)]],
    constant GridSamplerBackwardParams<4>& params [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  BackwardPreamble<T> p(grid, params, tid);
  int2 inp_size = {
      params.forward.input_sizes[3], params.forward.input_sizes[2]};
  int2 gInp_stride = {
      params.grad_input_strides[3], params.grad_input_strides[2]};
  auto gOut_sN = params.grad_output_strides[0];
  auto gOut_sC = params.grad_output_strides[1];
  auto gOut_sH = params.grad_output_strides[2];
  auto gOut_sW = params.grad_output_strides[3];
  auto gInp_sN = params.grad_input_strides[0];
  auto gInp_sC = params.grad_input_strides[1];

  float ix = compute_source(
      p.x, inp_size.x, params.forward.align_corners, params.padding_mode);
  float iy = compute_source(
      p.y, inp_size.y, params.forward.align_corners, params.padding_mode);
  int2 nearest = {
      static_cast<int32_t>(rint(ix)), static_cast<int32_t>(rint(iy))};

  auto gOut_ptr_NCHW =
      grad_output + p.n * gOut_sN + p.h * gOut_sH + p.w * gOut_sW;
  long NC_offset = p.n * gInp_sN;
  for (int32_t c = 0; c < params.forward.input_sizes[1];
       ++c, NC_offset += gInp_sC, gOut_ptr_NCHW += gOut_sC) {
    safe_add_2d_atomic<T>(
        grad_input,
        nearest,
        gInp_stride,
        inp_size,
        static_cast<opmath_t<T>>(*gOut_ptr_NCHW),
        NC_offset);
  }
}

// Bicubic backward kernel for grad_input
template <typename Pad, typename T>
kernel void grid_sampler_2d_backward_bicubic_input(
    device AtomicType_t<T>* grad_input [[buffer(0)]],
    constant T* grad_output [[buffer(1)]],
    constant T* grid [[buffer(2)]],
    constant GridSamplerBackwardParams<4>& params [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  BackwardPreamble<T> p(grid, params, tid);
  auto C = params.forward.input_sizes[1];
  int2 inp_size = {
      params.forward.input_sizes[3], params.forward.input_sizes[2]};
  int2 gInp_stride = {
      params.grad_input_strides[3], params.grad_input_strides[2]};
  auto gOut_sN = params.grad_output_strides[0];
  auto gOut_sC = params.grad_output_strides[1];
  auto gOut_sH = params.grad_output_strides[2];
  auto gOut_sW = params.grad_output_strides[3];
  auto gInp_sN = params.grad_input_strides[0];
  auto gInp_sC = params.grad_input_strides[1];

  float ix =
      grid_sampler_unnormalize(p.x, inp_size.x, params.forward.align_corners);
  float iy =
      grid_sampler_unnormalize(p.y, inp_size.y, params.forward.align_corners);

  float ix_nw = floor(ix);
  float iy_nw = floor(iy);

  float x_coeffs[4], y_coeffs[4];
  get_cubic_coefficients(x_coeffs, ix - ix_nw);
  get_cubic_coefficients(y_coeffs, iy - iy_nw);

  auto gOut_ptr_NCHW =
      grad_output + p.n * gOut_sN + p.h * gOut_sH + p.w * gOut_sW;
  long NC_offset = p.n * gInp_sN;
  int32_t ix_nw_i = static_cast<int32_t>(ix_nw);
  int32_t iy_nw_i = static_cast<int32_t>(iy_nw);

  for (int32_t c = 0; c < C;
       ++c, gOut_ptr_NCHW += gOut_sC, NC_offset += gInp_sC) {
    opmath_t<T> gOut = static_cast<opmath_t<T>>(*gOut_ptr_NCHW);

    for (int32_t i = 0; i < 4; ++i) {
      for (int32_t j = 0; j < 4; ++j) {
        add_value_bounded_backward<Pad, T>(
            grad_input,
            ix_nw_i - 1 + i,
            iy_nw_i - 1 + j,
            inp_size,
            gInp_stride,
            gOut * x_coeffs[i] * y_coeffs[j],
            params.forward.align_corners,
            NC_offset);
      }
    }
  }
}

// Bicubic backward kernel for grad_grid
template <typename Pad, typename T>
kernel void grid_sampler_2d_backward_bicubic_grid(
    device T* grad_grid [[buffer(0)]],
    constant T* grad_output [[buffer(1)]],
    constant T* input [[buffer(2)]],
    constant T* grid [[buffer(3)]],
    constant GridSamplerBackwardParams<4>& params [[buffer(4)]],
    uint tid [[thread_position_in_grid]]) {
  BackwardPreamble<T> p(grid, params, tid);
  auto C = params.forward.input_sizes[1];
  int2 inp_size = {
      params.forward.input_sizes[3], params.forward.input_sizes[2]};
  int2 inp_stride = {
      params.forward.input_strides[3], params.forward.input_strides[2]};
  auto inp_sN = params.forward.input_strides[0];
  auto inp_sC = params.forward.input_strides[1];
  auto gOut_sN = params.grad_output_strides[0];
  auto gOut_sC = params.grad_output_strides[1];
  auto gOut_sH = params.grad_output_strides[2];
  auto gOut_sW = params.grad_output_strides[3];
  auto gGrid_sW = params.grad_grid_sW;
  auto align_corners = params.forward.align_corners;

  float2 ix = grid_sampler_unnormalize_set_grad(p.x, inp_size.x, align_corners);
  float2 iy = grid_sampler_unnormalize_set_grad(p.y, inp_size.y, align_corners);

  float ix_nw = floor(ix.x);
  float iy_nw = floor(iy.x);
  float tx = ix.x - ix_nw;
  float ty = iy.x - iy_nw;

  float x_coeffs[4], y_coeffs[4];
  float x_coeffs_grad[4], y_coeffs_grad[4];
  get_cubic_coefficients(x_coeffs, tx);
  get_cubic_coefficients(y_coeffs, ty);
  get_cubic_coefficients_grad(x_coeffs_grad, tx);
  get_cubic_coefficients_grad(y_coeffs_grad, ty);

  opmath_t<T> gix = 0, giy = 0;
  auto gOut_ptr_NCHW =
      grad_output + p.n * gOut_sN + p.h * gOut_sH + p.w * gOut_sW;
  auto inp_ptr_NC = input + p.n * inp_sN;
  int32_t ix_nw_i = static_cast<int32_t>(ix_nw);
  int32_t iy_nw_i = static_cast<int32_t>(iy_nw);

  for (int32_t c = 0; c < C;
       ++c, gOut_ptr_NCHW += gOut_sC, inp_ptr_NC += inp_sC) {
    opmath_t<T> gOut = static_cast<opmath_t<T>>(*gOut_ptr_NCHW);

    for (int32_t i = 0; i < 4; ++i) {
      for (int32_t j = 0; j < 4; ++j) {
        opmath_t<T> val = get_value_bounded_backward<Pad, T>(
            inp_ptr_NC,
            ix_nw_i - 1 + i,
            iy_nw_i - 1 + j,
            inp_size,
            inp_stride,
            align_corners);

        gix -= val * x_coeffs_grad[i] * y_coeffs[j] * gOut;
        giy -= val * y_coeffs_grad[j] * x_coeffs[i] * gOut;
      }
    }
  }

  auto gGrid_ptr_NHW = grad_grid + tid * gGrid_sW;
  gGrid_ptr_NHW[0] = static_cast<T>(ix.y * gix);
  gGrid_ptr_NHW[1] = static_cast<T>(iy.y * giy);
}

// Registration macros for backward kernels.
// Bilinear/nearest _input and bilinear _grid kernels use runtime padding
// dispatch (only templated on dtype). Bicubic keeps the Pad template because
// padding affects its inner loop (16 bounded lookups per channel).
#define REGISTER_GRID_SAMPLER_2D_BACKWARD(DTYPE, INTERP)                \
  template [[host_name("grid_sampler_2d_backward_" #INTERP              \
                       "_input_" #DTYPE)]] kernel void                  \
      grid_sampler_2d_backward_##INTERP##_input<DTYPE>(                 \
          device AtomicType_t<DTYPE> * grad_input [[buffer(0)]],        \
          constant DTYPE * grad_output [[buffer(1)]],                   \
          constant DTYPE * grid [[buffer(2)]],                          \
          constant GridSamplerBackwardParams<4> & params [[buffer(3)]], \
          uint tid [[thread_position_in_grid]]);

#define REGISTER_GRID_SAMPLER_2D_BACKWARD_BICUBIC(DTYPE, PAD, PNAME)   \
  template [[host_name("grid_sampler_2d_backward_bicubic_input_" PNAME \
                       "_" #DTYPE)]] kernel void                       \
  grid_sampler_2d_backward_bicubic_input<PAD, DTYPE>(                  \
      device AtomicType_t<DTYPE> * grad_input [[buffer(0)]],           \
      constant DTYPE * grad_output [[buffer(1)]],                      \
      constant DTYPE * grid [[buffer(2)]],                             \
      constant GridSamplerBackwardParams<4> & params [[buffer(3)]],    \
      uint tid [[thread_position_in_grid]]);                           \
  template [[host_name("grid_sampler_2d_backward_bicubic_grid_" PNAME  \
                       "_" #DTYPE)]] kernel void                       \
  grid_sampler_2d_backward_bicubic_grid<PAD, DTYPE>(                   \
      device DTYPE * grad_grid [[buffer(0)]],                          \
      constant DTYPE * grad_output [[buffer(1)]],                      \
      constant DTYPE * input [[buffer(2)]],                            \
      constant DTYPE * grid [[buffer(3)]],                             \
      constant GridSamplerBackwardParams<4> & params [[buffer(4)]],    \
      uint tid [[thread_position_in_grid]]);

#define REGISTER_GRID_SAMPLER_2D_BACKWARD_OPS(DTYPE)                    \
  REGISTER_GRID_SAMPLER_2D_BACKWARD(DTYPE, bilinear)                    \
  REGISTER_GRID_SAMPLER_2D_BACKWARD(DTYPE, nearest)                     \
  template [[host_name(                                                 \
      "grid_sampler_2d_backward_bilinear_grid_" #DTYPE)]] kernel void   \
  grid_sampler_2d_backward_bilinear_grid<DTYPE>(                        \
      device DTYPE * grad_grid [[buffer(0)]],                           \
      constant DTYPE * grad_output [[buffer(1)]],                       \
      constant DTYPE * input [[buffer(2)]],                             \
      constant DTYPE * grid [[buffer(3)]],                              \
      constant GridSamplerBackwardParams<4> & params [[buffer(4)]],     \
      uint tid [[thread_position_in_grid]]);                            \
  REGISTER_GRID_SAMPLER_2D_BACKWARD_BICUBIC(DTYPE, PadZeros, "zeros")   \
  REGISTER_GRID_SAMPLER_2D_BACKWARD_BICUBIC(DTYPE, PadBorder, "border") \
  REGISTER_GRID_SAMPLER_2D_BACKWARD_BICUBIC(DTYPE, PadReflection, "reflection")

REGISTER_GRID_SAMPLER_2D_BACKWARD_OPS(float);
REGISTER_GRID_SAMPLER_2D_BACKWARD_OPS(half);
REGISTER_GRID_SAMPLER_2D_BACKWARD_OPS(bfloat);
