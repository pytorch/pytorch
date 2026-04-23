#include <ATen/native/mps/kernels/GridSampler.h>
#include <ATen/native/mps/kernels/SamplingHelpers.h>
#include <c10/metal/atomic.h>
#include <c10/metal/utils.h>
#include <metal_array>
#include <metal_stdlib>

using namespace metal;
using namespace c10::metal;

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

// 3D trilinear interpolation matching the 2D bilinear pattern.
// Takes pre-read grid coordinates as values, returns the interpolated value.
template <typename Pad, typename T>
static T interpolate_trilinear_3d(
    constant T* input,
    opmath_t<T> ix,
    opmath_t<T> iy,
    opmath_t<T> iz,
    int32_t inp_D,
    int32_t inp_H,
    int32_t inp_W,
    int32_t inp_sD,
    int32_t inp_sH,
    int32_t inp_sW,
    bool align_corners) {
  ix = grid_sampler_unnormalize(ix, inp_W, align_corners);
  iy = grid_sampler_unnormalize(iy, inp_H, align_corners);
  iz = grid_sampler_unnormalize(iz, inp_D, align_corners);

  int32_t ix_l = static_cast<int32_t>(floor(ix));
  int32_t iy_l = static_cast<int32_t>(floor(iy));
  int32_t iz_l = static_cast<int32_t>(floor(iz));
  int32_t ix_r = ix_l + 1;
  int32_t iy_r = iy_l + 1;
  int32_t iz_r = iz_l + 1;

  opmath_t<T> sx = ix - ix_l;
  opmath_t<T> sy = iy - iy_l;
  opmath_t<T> sz = iz - iz_l;

  int32_t ix_l_p = Pad::pad(ix_l, inp_W, align_corners);
  int32_t ix_r_p = Pad::pad(ix_r, inp_W, align_corners);
  int32_t iy_l_p = Pad::pad(iy_l, inp_H, align_corners);
  int32_t iy_r_p = Pad::pad(iy_r, inp_H, align_corners);
  int32_t iz_l_p = Pad::pad(iz_l, inp_D, align_corners);
  int32_t iz_r_p = Pad::pad(iz_r, inp_D, align_corners);

  opmath_t<T> out_acc = 0;
  if (!Pad::checks_bounds ||
      (iz_l_p != IDX_ZERO && iy_l_p != IDX_ZERO && ix_l_p != IDX_ZERO)) {
    out_acc += input[iz_l_p * inp_sD + iy_l_p * inp_sH + ix_l_p * inp_sW] *
        (1 - sz) * (1 - sy) * (1 - sx);
  }
  if (!Pad::checks_bounds ||
      (iz_l_p != IDX_ZERO && iy_l_p != IDX_ZERO && ix_r_p != IDX_ZERO)) {
    out_acc += input[iz_l_p * inp_sD + iy_l_p * inp_sH + ix_r_p * inp_sW] *
        (1 - sz) * (1 - sy) * sx;
  }
  if (!Pad::checks_bounds ||
      (iz_l_p != IDX_ZERO && iy_r_p != IDX_ZERO && ix_l_p != IDX_ZERO)) {
    out_acc += input[iz_l_p * inp_sD + iy_r_p * inp_sH + ix_l_p * inp_sW] *
        (1 - sz) * sy * (1 - sx);
  }
  if (!Pad::checks_bounds ||
      (iz_l_p != IDX_ZERO && iy_r_p != IDX_ZERO && ix_r_p != IDX_ZERO)) {
    out_acc += input[iz_l_p * inp_sD + iy_r_p * inp_sH + ix_r_p * inp_sW] *
        (1 - sz) * sy * sx;
  }
  if (!Pad::checks_bounds ||
      (iz_r_p != IDX_ZERO && iy_l_p != IDX_ZERO && ix_l_p != IDX_ZERO)) {
    out_acc += input[iz_r_p * inp_sD + iy_l_p * inp_sH + ix_l_p * inp_sW] * sz *
        (1 - sy) * (1 - sx);
  }
  if (!Pad::checks_bounds ||
      (iz_r_p != IDX_ZERO && iy_l_p != IDX_ZERO && ix_r_p != IDX_ZERO)) {
    out_acc += input[iz_r_p * inp_sD + iy_l_p * inp_sH + ix_r_p * inp_sW] * sz *
        (1 - sy) * sx;
  }
  if (!Pad::checks_bounds ||
      (iz_r_p != IDX_ZERO && iy_r_p != IDX_ZERO && ix_l_p != IDX_ZERO)) {
    out_acc += input[iz_r_p * inp_sD + iy_r_p * inp_sH + ix_l_p * inp_sW] * sz *
        sy * (1 - sx);
  }
  if (!Pad::checks_bounds ||
      (iz_r_p != IDX_ZERO && iy_r_p != IDX_ZERO && ix_r_p != IDX_ZERO)) {
    out_acc += input[iz_r_p * inp_sD + iy_r_p * inp_sH + ix_r_p * inp_sW] * sz *
        sy * sx;
  }

  return static_cast<T>(out_acc);
}

// 3D nearest neighbor interpolation matching the 2D nearest pattern.
template <typename Pad, typename T>
static T interpolate_nearest_3d(
    constant T* input,
    opmath_t<T> ix,
    opmath_t<T> iy,
    opmath_t<T> iz,
    int32_t inp_D,
    int32_t inp_H,
    int32_t inp_W,
    int32_t inp_sD,
    int32_t inp_sH,
    int32_t inp_sW,
    bool align_corners) {
  ix = Pad::compute_source(ix, inp_W, align_corners);
  iy = Pad::compute_source(iy, inp_H, align_corners);
  iz = Pad::compute_source(iz, inp_D, align_corners);

  int32_t ix_nearest = static_cast<int32_t>(rint(ix));
  int32_t iy_nearest = static_cast<int32_t>(rint(iy));
  int32_t iz_nearest = static_cast<int32_t>(rint(iz));

  if (Pad::checks_bounds) {
    if (ix_nearest < 0 || ix_nearest >= inp_W || iy_nearest < 0 ||
        iy_nearest >= inp_H || iz_nearest < 0 || iz_nearest >= inp_D) {
      return static_cast<T>(0);
    }
  }

  return input[iz_nearest * inp_sD + iy_nearest * inp_sH + ix_nearest * inp_sW];
}

// Interpolation strategies for 3D kernel (matching 2D pattern).
template <typename Pad>
struct Bilinear3D {
  template <typename T>
  static T interpolate(
      constant T* input,
      opmath_t<T> ix,
      opmath_t<T> iy,
      opmath_t<T> iz,
      int32_t inp_D,
      int32_t inp_H,
      int32_t inp_W,
      int32_t inp_sD,
      int32_t inp_sH,
      int32_t inp_sW,
      bool align_corners) {
    return interpolate_trilinear_3d<Pad, T>(
        input,
        ix,
        iy,
        iz,
        inp_D,
        inp_H,
        inp_W,
        inp_sD,
        inp_sH,
        inp_sW,
        align_corners);
  }
};

template <typename Pad>
struct Nearest3D {
  template <typename T>
  static T interpolate(
      constant T* input,
      opmath_t<T> ix,
      opmath_t<T> iy,
      opmath_t<T> iz,
      int32_t inp_D,
      int32_t inp_H,
      int32_t inp_W,
      int32_t inp_sD,
      int32_t inp_sH,
      int32_t inp_sW,
      bool align_corners) {
    return interpolate_nearest_3d<Pad, T>(
        input,
        ix,
        iy,
        iz,
        inp_D,
        inp_H,
        inp_W,
        inp_sD,
        inp_sH,
        inp_sW,
        align_corners);
  }
};

// 3D grid sampler kernel: one thread per output element (n, c, d, h, w).
template <typename Interp, typename T>
kernel void grid_sampler_3d(
    device T* output [[buffer(0)]],
    constant T* input [[buffer(1)]],
    constant T* grid [[buffer(2)]],
    constant GridSamplerParams<5>& params [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  auto C = params.output_sizes[1];
  auto out_D = params.output_sizes[2];
  auto out_H = params.output_sizes[3];
  auto out_W = params.output_sizes[4];

  auto out_sN = params.output_strides[0];
  auto out_sC = params.output_strides[1];
  auto out_sD = params.output_strides[2];
  auto out_sH = params.output_strides[3];
  auto out_sW = params.output_strides[4];
  auto inp_sN = params.input_strides[0];
  auto inp_sC = params.input_strides[1];
  auto inp_sD = params.input_strides[2];
  auto inp_sH = params.input_strides[3];
  auto inp_sW = params.input_strides[4];
  auto inp_D = params.input_sizes[2];
  auto inp_H = params.input_sizes[3];
  auto inp_W = params.input_sizes[4];

  auto grid_sN = params.grid_strides[0];
  auto grid_sD = params.grid_strides[1];
  auto grid_sH = params.grid_strides[2];
  auto grid_sW = params.grid_strides[3];
  auto grid_sCoor = params.grid_strides[4];

  auto align_corners = params.align_corners;

  int32_t w = tid % out_W;
  int32_t h = (tid / out_W) % out_H;
  int32_t d = (tid / (out_W * out_H)) % out_D;
  int32_t c = (tid / (out_W * out_H * out_D)) % C;
  int32_t n = tid / (out_W * out_H * out_D * C);

  auto grid_ptr = grid + n * grid_sN + d * grid_sD + h * grid_sH + w * grid_sW;
  opmath_t<T> ix = static_cast<opmath_t<T>>(grid_ptr[0]);
  opmath_t<T> iy = static_cast<opmath_t<T>>(grid_ptr[grid_sCoor]);
  opmath_t<T> iz = static_cast<opmath_t<T>>(grid_ptr[2 * grid_sCoor]);

  auto inp_ptr_NC = input + n * inp_sN + c * inp_sC;
  auto result = Interp::template interpolate<T>(
      inp_ptr_NC,
      ix,
      iy,
      iz,
      inp_D,
      inp_H,
      inp_W,
      inp_sD,
      inp_sH,
      inp_sW,
      align_corners);
  output[n * out_sN + c * out_sC + d * out_sD + h * out_sH + w * out_sW] =
      result;
}

// Padding mode constants (must match GridSamplerPadding enum)
constant int32_t kPaddingZeros = 0;
constant int32_t kPaddingBorder = 1;
constant int32_t kPaddingReflection = 2;

// Uses opmath_t<T> for intermediate computations to avoid overflow with
// half/bfloat
template <typename T>
T grid_sampler_compute_source_index_set_grad(
    T coord,
    int32_t size,
    int32_t padding_mode,
    bool align_corners,
    thread T* grad_in) {
  using U = opmath_t<T>;
  U u_coord = static_cast<U>(coord);
  U u_grad_in = static_cast<U>(*grad_in);
  U u_size = static_cast<U>(size);

  // Unnormalize
  if (align_corners) {
    u_coord = ((u_coord + U(1.0)) / U(2.0)) * (u_size - U(1.0));
    u_grad_in = (u_size - U(1.0)) / U(2.0);
  } else {
    u_coord = ((u_coord + U(1.0)) * u_size - U(1.0)) / U(2.0);
    u_grad_in = u_size / U(2.0);
  }

  if (padding_mode == kPaddingBorder) {
    // Borders are considered out of bounds for gradient calculation
    // (matching CUDA clip_coordinates_set_grad behavior).
    U grad_clip = U(1.0);
    if (u_coord <= U(0.0)) {
      u_coord = U(0.0);
      grad_clip = U(0.0);
    } else if (u_coord >= u_size - U(1.0)) {
      u_coord = u_size - U(1.0);
      grad_clip = U(0.0);
    }
    u_grad_in = u_grad_in * grad_clip;
  } else if (padding_mode == kPaddingReflection) {
    U grad_refl = U(1.0);
    U twice_low, twice_high;
    if (align_corners) {
      twice_low = U(0.0);
      twice_high = U(2 * (size - 1));
    } else {
      twice_low = U(-1.0);
      twice_high = U(2 * size - 1);
    }

    if (twice_low != twice_high) {
      U min_val = twice_low / U(2.0);
      U span = (twice_high - twice_low) / U(2.0);
      u_coord = u_coord - min_val;

      if (u_coord < U(0.0)) {
        u_coord = -u_coord;
        grad_refl = -grad_refl;
      }

      U extra = u_coord - span * floor(u_coord / span);
      int32_t flips = static_cast<int32_t>(floor(u_coord / span));

      if (flips % 2 == 0) {
        u_coord = extra + min_val;
      } else {
        u_coord = span - extra + min_val;
        grad_refl = -grad_refl;
      }
    } else {
      u_coord = U(0.0);
    }

    // Clip after reflection (borders out of bounds for gradient)
    U grad_clip = U(1.0);
    if (u_coord <= U(0.0)) {
      u_coord = U(0.0);
      grad_clip = U(0.0);
    } else if (u_coord >= u_size - U(1.0)) {
      u_coord = u_size - U(1.0);
      grad_clip = U(0.0);
    }
    u_grad_in = u_grad_in * grad_refl * grad_clip;
  }

  coord = static_cast<T>(u_coord);
  *grad_in = static_cast<T>(u_grad_in);
  return coord;
}

inline bool within_bounds_3d(
    int32_t z,
    int32_t y,
    int32_t x,
    int32_t D,
    int32_t H,
    int32_t W) {
  return z >= 0 && z < D && y >= 0 && y < H && x >= 0 && x < W;
}

template <typename T>
kernel void grid_sampler_3d_backward(
    constant T* grad_output [[buffer(0)]],
    constant T* input [[buffer(1)]],
    constant T* grid [[buffer(2)]],
    device AtomicType_t<T>* grad_input [[buffer(3)]],
    device T* grad_grid [[buffer(4)]],
    constant GridSampler3DBackwardParams& params [[buffer(5)]],
    uint3 thread_index [[thread_position_in_grid]]) {
  const auto out_w = thread_index.x;
  const auto out_d_h_combined = thread_index.y;
  const auto n = thread_index.z;

  const auto out_d = out_d_h_combined / params.output_sizes[3];
  const auto out_h = out_d_h_combined % params.output_sizes[3];

  if (n >= params.input_sizes[0] || out_d >= params.output_sizes[2] ||
      out_h >= params.output_sizes[3] || out_w >= params.output_sizes[4]) {
    return;
  }

  const auto C = params.input_sizes[1];
  const auto inp_D = params.input_sizes[2];
  const auto inp_H = params.input_sizes[3];
  const auto inp_W = params.input_sizes[4];

  const auto grid_offset = n * params.grid_strides[0] +
      out_d * params.grid_strides[1] + out_h * params.grid_strides[2] +
      out_w * params.grid_strides[3];

  const opmath_t<T> grid_x = grid[grid_offset];
  const opmath_t<T> grid_y = grid[grid_offset + params.grid_strides[4]];
  const opmath_t<T> grid_z = grid[grid_offset + 2 * params.grid_strides[4]];

  opmath_t<T> gix_mult, giy_mult, giz_mult;
  opmath_t<T> ix = grid_sampler_compute_source_index_set_grad(
      grid_x,
      static_cast<int32_t>(inp_W),
      params.padding_mode,
      params.align_corners,
      &gix_mult);
  opmath_t<T> iy = grid_sampler_compute_source_index_set_grad(
      grid_y,
      static_cast<int32_t>(inp_H),
      params.padding_mode,
      params.align_corners,
      &giy_mult);
  opmath_t<T> iz = grid_sampler_compute_source_index_set_grad(
      grid_z,
      static_cast<int32_t>(inp_D),
      params.padding_mode,
      params.align_corners,
      &giz_mult);

  if (params.interpolation_mode == 0) { // trilinear
    const int ix_0 = static_cast<int>(floor(ix));
    const int iy_0 = static_cast<int>(floor(iy));
    const int iz_0 = static_cast<int>(floor(iz));
    const opmath_t<T> dx = ix - ix_0;
    const opmath_t<T> dy = iy - iy_0;
    const opmath_t<T> dz = iz - iz_0;
    const opmath_t<T> wx[2] = {1 - dx, dx};
    const opmath_t<T> wy[2] = {1 - dy, dy};
    const opmath_t<T> wz[2] = {1 - dz, dz};

    opmath_t<T> gix = 0, giy = 0, giz = 0;

    for (uint32_t c = 0; c < C; c++) {
      const auto grad_out_offset = n * params.grad_output_strides[0] +
          c * params.grad_output_strides[1] +
          out_d * params.grad_output_strides[2] +
          out_h * params.grad_output_strides[3] +
          out_w * params.grad_output_strides[4];
      const opmath_t<T> gOut = grad_output[grad_out_offset];
      const auto base_grad_input_offset =
          n * params.grad_input_strides[0] + c * params.grad_input_strides[1];
      const auto input_base_offset =
          n * params.input_strides[0] + c * params.input_strides[1];

      for (int i = 0; i < 8; i++) {
        const int xi = i & 1;
        const int yi = (i >> 1) & 1;
        const int zi = (i >> 2) & 1;
        const int cx = ix_0 + xi;
        const int cy = iy_0 + yi;
        const int cz = iz_0 + zi;

        if (within_bounds_3d(
                cz,
                cy,
                cx,
                static_cast<int32_t>(inp_D),
                static_cast<int32_t>(inp_H),
                static_cast<int32_t>(inp_W))) {
          const opmath_t<T> w = wx[xi] * wy[yi] * wz[zi];

          if (params.compute_grad_input) {
            AtomicType<T>::atomic_add(
                grad_input,
                base_grad_input_offset + cz * params.grad_input_strides[2] +
                    cy * params.grad_input_strides[3] +
                    cx * params.grad_input_strides[4],
                static_cast<T>(w * gOut));
          }

          if (params.compute_grad_grid) {
            const opmath_t<T> val = input
                [input_base_offset + cz * params.input_strides[2] +
                 cy * params.input_strides[3] + cx * params.input_strides[4]];
            const opmath_t<T> sign_x = xi ? 1 : -1;
            const opmath_t<T> sign_y = yi ? 1 : -1;
            const opmath_t<T> sign_z = zi ? 1 : -1;
            gix += sign_x * val * wy[yi] * wz[zi] * gOut;
            giy += sign_y * val * wx[xi] * wz[zi] * gOut;
            giz += sign_z * val * wx[xi] * wy[yi] * gOut;
          }
        }
      }
    }

    if (params.compute_grad_grid) {
      const auto grad_grid_base_offset = n * params.grad_grid_strides[0] +
          out_d * params.grad_grid_strides[1] +
          out_h * params.grad_grid_strides[2] +
          out_w * params.grad_grid_strides[3];
      grad_grid[grad_grid_base_offset] = static_cast<T>(gix_mult * gix);
      grad_grid[grad_grid_base_offset + params.grid_strides[4]] =
          static_cast<T>(giy_mult * giy);
      grad_grid[grad_grid_base_offset + 2 * params.grid_strides[4]] =
          static_cast<T>(giz_mult * giz);
    }
  } else if (params.compute_grad_input) { // nearest
    int32_t ix_n = static_cast<int32_t>(rint(ix));
    int32_t iy_n = static_cast<int32_t>(rint(iy));
    int32_t iz_n = static_cast<int32_t>(rint(iz));

    if (params.padding_mode == kPaddingBorder) {
      ix_n = clamp(ix_n, 0, static_cast<int32_t>(inp_W - 1));
      iy_n = clamp(iy_n, 0, static_cast<int32_t>(inp_H - 1));
      iz_n = clamp(iz_n, 0, static_cast<int32_t>(inp_D - 1));
    } else if (params.padding_mode == kPaddingReflection) {
      if (params.align_corners) {
        ix_n = static_cast<int32_t>(rint(reflect_coordinates(
            static_cast<float>(ix_n), 0.0f, 2.0f * (inp_W - 1))));
        iy_n = static_cast<int32_t>(rint(reflect_coordinates(
            static_cast<float>(iy_n), 0.0f, 2.0f * (inp_H - 1))));
        iz_n = static_cast<int32_t>(rint(reflect_coordinates(
            static_cast<float>(iz_n), 0.0f, 2.0f * (inp_D - 1))));
      } else {
        ix_n = static_cast<int32_t>(rint(reflect_coordinates(
            static_cast<float>(ix_n), -1.0f, 2.0f * inp_W - 1)));
        iy_n = static_cast<int32_t>(rint(reflect_coordinates(
            static_cast<float>(iy_n), -1.0f, 2.0f * inp_H - 1)));
        iz_n = static_cast<int32_t>(rint(reflect_coordinates(
            static_cast<float>(iz_n), -1.0f, 2.0f * inp_D - 1)));
      }
      ix_n = clamp(ix_n, 0, static_cast<int32_t>(inp_W - 1));
      iy_n = clamp(iy_n, 0, static_cast<int32_t>(inp_H - 1));
      iz_n = clamp(iz_n, 0, static_cast<int32_t>(inp_D - 1));
    }

    bool in_bounds = params.padding_mode != kPaddingZeros ||
        within_bounds_3d(iz_n,
                         iy_n,
                         ix_n,
                         static_cast<int32_t>(inp_D),
                         static_cast<int32_t>(inp_H),
                         static_cast<int32_t>(inp_W));

    if (in_bounds) {
      const auto base_offset = n * params.grad_input_strides[0] +
          iz_n * params.grad_input_strides[2] +
          iy_n * params.grad_input_strides[3] +
          ix_n * params.grad_input_strides[4];

      for (uint32_t c = 0; c < C; c++) {
        const auto grad_out_offset = n * params.grad_output_strides[0] +
            c * params.grad_output_strides[1] +
            out_d * params.grad_output_strides[2] +
            out_h * params.grad_output_strides[3] +
            out_w * params.grad_output_strides[4];
        const opmath_t<T> gOut = grad_output[grad_out_offset];
        AtomicType<T>::atomic_add(
            grad_input,
            base_offset + c * params.grad_input_strides[1],
            static_cast<T>(gOut));
      }
    }
  }
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

#define REGISTER_GRID_SAMPLER_3D(DTYPE, INTERP, INAME, PAD, PNAME)      \
  template [[host_name("grid_sampler_3d_" INAME "_" PNAME "_" #DTYPE)]] \
  kernel void grid_sampler_3d<INTERP<PAD>, DTYPE>(                      \
      device DTYPE * output [[buffer(0)]],                              \
      constant DTYPE * input [[buffer(1)]],                             \
      constant DTYPE * grid [[buffer(2)]],                              \
      constant GridSamplerParams<5> & params [[buffer(3)]],             \
      uint tid [[thread_position_in_grid]]);

#define REGISTER_GRID_SAMPLER_3D_INTERP(DTYPE, INTERP, INAME)         \
  REGISTER_GRID_SAMPLER_3D(DTYPE, INTERP, INAME, PadZeros, "zeros")   \
  REGISTER_GRID_SAMPLER_3D(DTYPE, INTERP, INAME, PadBorder, "border") \
  REGISTER_GRID_SAMPLER_3D(DTYPE, INTERP, INAME, PadReflection, "reflection")

#define REGISTER_GRID_SAMPLER_BACKWARD(DTYPE)                      \
  template [[host_name("grid_sampler_3d_backward_" #DTYPE)]]       \
  kernel void grid_sampler_3d_backward<DTYPE>(                     \
      constant DTYPE * grad_output [[buffer(0)]],                  \
      constant DTYPE * input [[buffer(1)]],                        \
      constant DTYPE * grid [[buffer(2)]],                         \
      device AtomicType_t<DTYPE> * grad_input [[buffer(3)]],       \
      device DTYPE * grad_grid [[buffer(4)]],                      \
      constant GridSampler3DBackwardParams & params [[buffer(5)]], \
      uint3 thread_index [[thread_position_in_grid]]);

#define REGISTER_GRID_SAMPLER_OPS(DTYPE)                         \
  REGISTER_GRID_SAMPLER_2D_INTERP(DTYPE, Bilinear2D, "bilinear") \
  REGISTER_GRID_SAMPLER_2D_INTERP(DTYPE, Nearest2D, "nearest")   \
  REGISTER_GRID_SAMPLER_2D_INTERP(DTYPE, Bicubic2D, "bicubic")   \
  REGISTER_GRID_SAMPLER_3D_INTERP(DTYPE, Bilinear3D, "bilinear") \
  REGISTER_GRID_SAMPLER_3D_INTERP(DTYPE, Nearest3D, "nearest")   \
  REGISTER_GRID_SAMPLER_BACKWARD(DTYPE)

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
