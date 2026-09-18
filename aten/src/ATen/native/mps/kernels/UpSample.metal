#include <ATen/native/mps/kernels/SamplingHelpers.h>
#include <ATen/native/mps/kernels/UpSample.h>
#include <c10/metal/atomic.h>
#include <metal_stdlib>

using namespace metal;
using namespace c10::metal;

template <typename accscalar_t>
accscalar_t area_pixel_compute_source_index(
    accscalar_t scale,
    int dst_index,
    bool align_corners,
    bool cubic) {
  if (align_corners) {
    return scale * dst_index;
  } else {
    accscalar_t src_idx = scale * (dst_index + static_cast<accscalar_t>(0.5)) -
        static_cast<accscalar_t>(0.5);
    // See Note[Follow Opencv resize logic]
    return (!cubic && src_idx < static_cast<accscalar_t>(0))
        ? static_cast<accscalar_t>(0)
        : src_idx;
  }
}

template <typename scalar_t>
scalar_t upsample_get_value_bounded(
    constant scalar_t* data,
    uint3 dim,
    ::metal::array<long, 5> strides,
    uint n,
    uint c,
    uint z,
    uint y,
    uint x) {
  auto access_z = max(min(z, dim.z - 1), 0U);
  auto access_y = max(min(y, dim.y - 1), 0U);
  auto access_x = max(min(x, dim.x - 1), 0U);
  return data
      [n * strides[0] + c * strides[1] + access_z * strides[2] +
       access_y * strides[3] + access_x * strides[4]];
}

// 1D/2D bounded gather. The spatial coords (index >= 2) are clamped to
// [0, size-1]; the leading batch/channel coords (0, 1) are always in range.
// 3D has its own array<long, 5> overload below since Metal vectors cap at 4
// lanes.
template <typename scalar_t, uint N>
scalar_t upsample_get_value_bounded(
    constant scalar_t* data,
    vec<long, N> sizes,
    vec<long, N> strides,
    vec<long, N> coords) {
  long offset = 0;
  for (uint i = 0; i < N; i++) {
    const auto idx = i < 2 ? coords[i] : clamp(coords[i], 0L, sizes[i] - 1);
    offset += idx * strides[i];
  }
  return data[offset];
}

// Store counterpart of upsample_get_value_bounded: output coordinates are
// always in range, so no clamping is needed.
template <typename scalar_t, uint N>
void upsample_set_value(
    device scalar_t* data,
    vec<long, N> strides,
    vec<long, N> coords,
    scalar_t value) {
  long offset = 0;
  for (uint i = 0; i < N; i++) {
    offset += coords[i] * strides[i];
  }
  data[offset] = value;
}

// Unpack a UpsampleParams stride/size array into the Metal vector the 1D/2D
// kernel bodies index with (.x/.y/.z/.w). One template covers the 1D (<3>) and
// 2D (<4>) ranks; the per-lane copy unrolls since N is a compile-time constant.
template <typename T, size_t N>
inline vec<T, N> to_vec(constant ::c10::metal::array<T, N>& a) {
  vec<T, N> v;
  for (size_t i = 0; i < N; i++) {
    v[i] = a[i];
  }
  return v;
}

template <typename scalar_t, uint N>
void upsample_increment_value_bounded(
    device AtomicType_t<scalar_t>* data,
    vec<long, N> sizes,
    vec<long, N> strides,
    vec<long, N> coords,
    float value) {
  long offset = 0;
  for (uint i = 0; i < N; i++) {
    const auto idx = i < 2 ? coords[i] : clamp(coords[i], 0L, sizes[i] - 1);
    offset += idx * strides[i];
  }
  AtomicType<scalar_t>::atomic_add(data, offset, static_cast<scalar_t>(value));
}

template <typename scalar_t>
void upsample_increment_value_bounded(
    device AtomicType_t<scalar_t>* data,
    uint3 dim,
    ::metal::array<long, 5> strides,
    uint n,
    uint c,
    uint z,
    uint y,
    uint x,
    float value) {
  auto access_z = max(min(z, dim.z - 1), 0U);
  auto access_y = max(min(y, dim.y - 1), 0U);
  auto access_x = max(min(x, dim.x - 1), 0U);
  AtomicType<scalar_t>::atomic_add(
      data,
      n * strides[0] + c * strides[1] + access_z * strides[2] +
          access_y * strides[3] + access_x * strides[4],
      static_cast<scalar_t>(value));
}

template <typename T>
struct linear_return_type {
  typedef float type;
};
template <>
struct linear_return_type<uchar> {
  typedef uchar type;
};
template <typename T>
using linear_return_t = typename linear_return_type<T>::type;

template <typename T>
inline linear_return_t<T> linear_interp(T v0, T v1, float x) {
  return x * v1 + (1 - x) * v0;
}

/* 3D interpolation kernels and helper functions */
inline uint3 coords_from_threadidx(
    constant UpsampleParams<5>& params,
    uint thread_index) {
  const auto size_x = static_cast<uint>(params.output_sizes[4]);
  const auto size_xy = static_cast<uint>(params.output_sizes[3]) * size_x;
  auto output_xy = thread_index % size_xy;
  return uint3(output_xy % size_x, output_xy / size_x, thread_index / size_xy);
}

inline float3 coords_to_real_coords(
    constant UpsampleParams<5>& params,
    uint3 output,
    bool align_corners) {
  auto real_x = area_pixel_compute_source_index(
      params.scales[0], output.x, align_corners, /*cubic=*/false);
  auto real_y = area_pixel_compute_source_index(
      params.scales[1], output.y, align_corners, /*cubic=*/false);
  auto real_z = area_pixel_compute_source_index(
      params.scales[2], output.z, align_corners, /*cubic=*/false);
  return float3(real_x, real_y, real_z);
}

template <typename T>
kernel void upsample_nearest_exact_3d(
    constant T* inputData [[buffer(0)]],
    device T* outputData [[buffer(1)]],
    constant UpsampleParams<5>& params [[buffer(2)]],
    uint thread_index [[thread_position_in_grid]]) {
  const auto input_sizes = uint3(
      params.input_sizes[4], params.input_sizes[3], params.input_sizes[2]);
  const auto output = coords_from_threadidx(params, thread_index);
  const auto real = coords_to_real_coords(params, output, false);
  for (uint n = 0; n < params.output_sizes[0]; n++) {
    for (uint c = 0; c < params.output_sizes[1]; c++) {
      auto res = upsample_get_value_bounded<T>(
          inputData,
          input_sizes,
          params.input_strides,
          n,
          c,
          real.z + .5,
          real.y + .5,
          real.x + .5);
      outputData
          [n * params.output_strides[0] + c * params.output_strides[1] +
           output.z * params.output_strides[2] +
           output.y * params.output_strides[3] +
           output.x * params.output_strides[4]] = static_cast<T>(res);
    }
  }
}

template <typename T>
kernel void upsample_nearest_exact_3d_backward(
    device AtomicType_t<T>* gradInputData [[buffer(0)]],
    constant T* gradOutputData [[buffer(1)]],
    constant UpsampleParams<5>& params [[buffer(2)]],
    uint thread_index [[thread_position_in_grid]]) {
  const auto input_sizes = uint3(
      params.input_sizes[4], params.input_sizes[3], params.input_sizes[2]);
  const auto output = coords_from_threadidx(params, thread_index);
  const auto real = coords_to_real_coords(params, output, false);
  for (uint n = 0; n < params.output_sizes[0]; n++) {
    for (uint c = 0; c < params.output_sizes[1]; c++) {
      auto res = gradOutputData
          [n * params.output_strides[0] + c * params.output_strides[1] +
           output.z * params.output_strides[2] +
           output.y * params.output_strides[3] +
           output.x * params.output_strides[4]];
      upsample_increment_value_bounded<T>(
          gradInputData,
          input_sizes,
          params.input_strides,
          n,
          c,
          real.z + .5,
          real.y + .5,
          real.x + .5,
          res);
    }
  }
}

template <typename T>
kernel void upsample_nearest_3d(
    constant T* inputData [[buffer(0)]],
    device T* outputData [[buffer(1)]],
    constant UpsampleParams<5>& params [[buffer(2)]],
    uint thread_index [[thread_position_in_grid]]) {
  const auto input_sizes = uint3(
      params.input_sizes[4], params.input_sizes[3], params.input_sizes[2]);
  const auto output = coords_from_threadidx(params, thread_index);
  const auto real = coords_to_real_coords(params, output, true);
  for (uint n = 0; n < params.output_sizes[0]; n++) {
    for (uint c = 0; c < params.output_sizes[1]; c++) {
      auto res = upsample_get_value_bounded<T>(
          inputData,
          input_sizes,
          params.input_strides,
          n,
          c,
          real.z,
          real.y,
          real.x);
      outputData
          [n * params.output_strides[0] + c * params.output_strides[1] +
           output.z * params.output_strides[2] +
           output.y * params.output_strides[3] +
           output.x * params.output_strides[4]] = static_cast<T>(res);
    }
  }
}

template <typename T>
kernel void upsample_nearest_3d_backward(
    device AtomicType_t<T>* gradInputData [[buffer(0)]],
    constant T* gradOutputData [[buffer(1)]],
    constant UpsampleParams<5>& params [[buffer(2)]],
    uint thread_index [[thread_position_in_grid]]) {
  const auto input_sizes = uint3(
      params.input_sizes[4], params.input_sizes[3], params.input_sizes[2]);
  const auto output = coords_from_threadidx(params, thread_index);
  const auto real = coords_to_real_coords(params, output, true);
  for (uint n = 0; n < params.output_sizes[0]; n++) {
    for (uint c = 0; c < params.output_sizes[1]; c++) {
      auto res = gradOutputData
          [n * params.output_strides[0] + c * params.output_strides[1] +
           output.z * params.output_strides[2] +
           output.y * params.output_strides[3] +
           output.x * params.output_strides[4]];
      upsample_increment_value_bounded<T>(
          gradInputData,
          input_sizes,
          params.input_strides,
          n,
          c,
          real.z,
          real.y,
          real.x,
          res);
    }
  }
}

template <typename T>
kernel void upsample_trilinear(
    constant T* inputData [[buffer(0)]],
    device T* outputData [[buffer(1)]],
    constant UpsampleParams<5>& params [[buffer(2)]],
    uint thread_index [[thread_position_in_grid]]) {
  const auto input_sizes = uint3(
      params.input_sizes[4], params.input_sizes[3], params.input_sizes[2]);
  const auto output = coords_from_threadidx(params, thread_index);
  const auto real = coords_to_real_coords(params, output, params.align_corners);
  auto t = fract(real);
  for (uint n = 0; n < params.output_sizes[0]; n++) {
    for (uint c = 0; c < params.output_sizes[1]; c++) {
      auto i000 = upsample_get_value_bounded<T>(
          inputData,
          input_sizes,
          params.input_strides,
          n,
          c,
          real.z,
          real.y,
          real.x);
      auto i001 = upsample_get_value_bounded<T>(
          inputData,
          input_sizes,
          params.input_strides,
          n,
          c,
          real.z,
          real.y,
          real.x + 1);
      auto i010 = upsample_get_value_bounded<T>(
          inputData,
          input_sizes,
          params.input_strides,
          n,
          c,
          real.z,
          real.y + 1,
          real.x);
      auto i011 = upsample_get_value_bounded<T>(
          inputData,
          input_sizes,
          params.input_strides,
          n,
          c,
          real.z,
          real.y + 1,
          real.x + 1);
      auto i100 = upsample_get_value_bounded<T>(
          inputData,
          input_sizes,
          params.input_strides,
          n,
          c,
          real.z + 1,
          real.y,
          real.x);
      auto i101 = upsample_get_value_bounded<T>(
          inputData,
          input_sizes,
          params.input_strides,
          n,
          c,
          real.z + 1,
          real.y,
          real.x + 1);
      auto i110 = upsample_get_value_bounded<T>(
          inputData,
          input_sizes,
          params.input_strides,
          n,
          c,
          real.z + 1,
          real.y + 1,
          real.x);
      auto i111 = upsample_get_value_bounded<T>(
          inputData,
          input_sizes,
          params.input_strides,
          n,
          c,
          real.z + 1,
          real.y + 1,
          real.x + 1);
      auto i00_l = linear_interp(i000, i001, t.x);
      auto i01_l = linear_interp(i010, i011, t.x);
      auto i10_l = linear_interp(i100, i101, t.x);
      auto i11_l = linear_interp(i110, i111, t.x);
      auto i0_l = linear_interp(i00_l, i01_l, t.y);
      auto i1_l = linear_interp(i10_l, i11_l, t.y);
      auto res = linear_interp(i0_l, i1_l, t.z);
      outputData
          [n * params.output_strides[0] + c * params.output_strides[1] +
           output.z * params.output_strides[2] +
           output.y * params.output_strides[3] +
           output.x * params.output_strides[4]] = static_cast<T>(res);
    }
  }
}

template <typename T>
kernel void upsample_trilinear_backward(
    device AtomicType_t<T>* gradInputData [[buffer(0)]],
    constant T* gradOutputData [[buffer(1)]],
    constant UpsampleParams<5>& params [[buffer(2)]],
    uint thread_index [[thread_position_in_grid]]) {
  const auto input_sizes = uint3(
      params.input_sizes[4], params.input_sizes[3], params.input_sizes[2]);
  const auto output = coords_from_threadidx(params, thread_index);
  const auto real = coords_to_real_coords(params, output, params.align_corners);
  auto t = fract(real);
  for (uint n = 0; n < params.output_sizes[0]; n++) {
    for (uint c = 0; c < params.output_sizes[1]; c++) {
      auto res = gradOutputData
          [n * params.output_strides[0] + c * params.output_strides[1] +
           output.z * params.output_strides[2] +
           output.y * params.output_strides[3] +
           output.x * params.output_strides[4]];
      for (int d = 0; d < 8; d++) {
        const auto w = (d & 1 ? t.x : 1.0 - t.x) * (d & 2 ? t.y : 1.0 - t.y) *
            (d & 4 ? t.z : 1.0 - t.z);
        upsample_increment_value_bounded<T>(
            gradInputData,
            input_sizes,
            params.input_strides,
            n,
            c,
            real.z + ((d & 4) >> 2),
            real.y + ((d & 2) >> 1),
            real.x + (d & 1),
            res * w);
      }
    }
  }
}

// See Note [ Weights computation for uint8_t and multiplication trick ]
// Essentially fall back to fixed floating point arithmetic during uint8
// interpolation, which is not necessarily more accurate (see example below),
// but matches closes to what CPU can deliver
// I.e. mid-point 152+249+172+35 is 152, but algorithm yields 153 as horizontal
// and vertical interpolation is done in separate steps and results are rounded
// to uint8 Also, as Metal is currently limited to 32-bit floats, results will
// never match those on CPU especially for 1/3, 2/3 scale
template <>
inline uchar linear_interp(uchar v0, uchar v1, float x) {
  constexpr auto PRECISION_BITS = 15;
  constexpr auto one = 1L << (PRECISION_BITS);
  constexpr auto onehalf = 1L << (PRECISION_BITS - 1);
  auto ix = static_cast<long>(x * one + .5);
  auto iomx = static_cast<long>((1.0 - x) * one + .5);
  return (onehalf + v0 * iomx + v1 * ix) >> PRECISION_BITS;
}

template <typename T>
kernel void upsample_linear1d(
    constant T* inputData [[buffer(0)]],
    device T* outputData [[buffer(1)]],
    constant UpsampleParams<3>& params [[buffer(2)]],
    uint thread_index [[thread_position_in_grid]]) {
  const auto input_strides = to_vec(params.input_strides);
  const auto output_strides = to_vec(params.output_strides);
  const auto input_sizes = to_vec(params.input_sizes);
  const auto output_sizes = to_vec(params.output_sizes);
  const float2 scales = float2(params.scales[0], 0.0f);
  const bool align_corners = params.align_corners;
  auto output_x = thread_index;
  auto real_x = area_pixel_compute_source_index(
      scales.x, output_x, align_corners, /*cubic=*/false);
  auto t_x = fract(real_x);

  for (int n = 0; n < output_sizes.x; n++) {
    for (int c = 0; c < output_sizes.y; c++) {
      auto i00 = upsample_get_value_bounded<T>(
          inputData, input_sizes, input_strides, long3(n, c, real_x));
      auto i01 = upsample_get_value_bounded<T>(
          inputData, input_sizes, input_strides, long3(n, c, real_x + 1));
      auto res = linear_interp(i00, i01, t_x);
      upsample_set_value(
          outputData,
          output_strides,
          long3(n, c, output_x),
          static_cast<T>(res));
    }
  }
}
template <typename T>
kernel void upsample_bilinear2d(
    constant T* inputData [[buffer(0)]],
    device T* outputData [[buffer(1)]],
    constant UpsampleParams<4>& params [[buffer(2)]],
    uint thread_index [[thread_position_in_grid]]) {
  const auto input_strides = to_vec(params.input_strides);
  const auto output_strides = to_vec(params.output_strides);
  const auto input_sizes = to_vec(params.input_sizes);
  const auto output_sizes = to_vec(params.output_sizes);
  const float2 scales = float2(params.scales[0], params.scales[1]);
  const bool align_corners = params.align_corners;
  auto output_x = thread_index % static_cast<uint>(output_sizes.w);
  auto output_y = thread_index / static_cast<uint>(output_sizes.w);
  auto real_x = area_pixel_compute_source_index(
      scales.x, output_x, align_corners, /*cubic=*/false);
  auto t_x = fract(real_x);

  auto real_y = area_pixel_compute_source_index(
      scales.y, output_y, align_corners, /*cubic=*/false);
  auto t_y = fract(real_y);
  for (int n = 0; n < output_sizes.x; n++) {
    for (int c = 0; c < output_sizes.y; c++) {
      auto i00 = upsample_get_value_bounded<T>(
          inputData, input_sizes, input_strides, long4(n, c, real_y, real_x));
      auto i01 = upsample_get_value_bounded<T>(
          inputData,
          input_sizes,
          input_strides,
          long4(n, c, real_y, real_x + 1));
      auto i10 = upsample_get_value_bounded<T>(
          inputData,
          input_sizes,
          input_strides,
          long4(n, c, real_y + 1, real_x));
      auto i11 = upsample_get_value_bounded<T>(
          inputData,
          input_sizes,
          input_strides,
          long4(n, c, real_y + 1, real_x + 1));
      auto i0_l = linear_interp(i00, i01, t_x);
      auto i1_l = linear_interp(i10, i11, t_x);
      auto res = linear_interp(i0_l, i1_l, t_y);
      upsample_set_value(
          outputData,
          output_strides,
          long4(n, c, output_y, output_x),
          static_cast<T>(res));
    }
  }
}

// Nearest source index: floor(scale * dst) for nearest, floor(scale * (dst +
// 0.5)) for nearest-exact. scale is the input/output ratio, so truncation
// (== floor for the non-negative product) matches the CPU reference. Computing
// this directly in Metal (built with -fno-fast-math) is bit-stable across macOS
// versions, unlike the MPSGraph resize which mis-rounds exact boundaries on
// macOS 14.
template <bool exact>
inline long nearest_src_index(float scale, uint dst) {
  return static_cast<long>(exact ? scale * (dst + 0.5f) : scale * dst);
}

template <typename T, bool exact>
kernel void upsample_nearest1d(
    constant T* inputData [[buffer(0)]],
    device T* outputData [[buffer(1)]],
    constant UpsampleParams<3>& params [[buffer(2)]],
    uint thread_index [[thread_position_in_grid]]) {
  const auto input_strides = to_vec(params.input_strides);
  const auto output_strides = to_vec(params.output_strides);
  const auto input_sizes = to_vec(params.input_sizes);
  const auto output_sizes = to_vec(params.output_sizes);
  const float2 scales = float2(params.scales[0], 0.0f);
  const auto output_x = thread_index;
  const auto src_x = nearest_src_index<exact>(scales.x, output_x);
  for (int n = 0; n < output_sizes.x; n++) {
    for (int c = 0; c < output_sizes.y; c++) {
      upsample_set_value(
          outputData,
          output_strides,
          long3(n, c, output_x),
          upsample_get_value_bounded<T>(
              inputData, input_sizes, input_strides, long3(n, c, src_x)));
    }
  }
}

template <typename T, bool exact>
kernel void upsample_nearest2d(
    constant T* inputData [[buffer(0)]],
    device T* outputData [[buffer(1)]],
    constant UpsampleParams<4>& params [[buffer(2)]],
    uint thread_index [[thread_position_in_grid]]) {
  const auto input_strides = to_vec(params.input_strides);
  const auto output_strides = to_vec(params.output_strides);
  const auto input_sizes = to_vec(params.input_sizes);
  const auto output_sizes = to_vec(params.output_sizes);
  const float2 scales = float2(params.scales[0], params.scales[1]);
  const auto output_x = thread_index % static_cast<uint>(output_sizes.w);
  const auto output_y = thread_index / static_cast<uint>(output_sizes.w);
  const auto src_x = nearest_src_index<exact>(scales.x, output_x);
  const auto src_y = nearest_src_index<exact>(scales.y, output_y);
  for (int n = 0; n < output_sizes.x; n++) {
    for (int c = 0; c < output_sizes.y; c++) {
      upsample_set_value(
          outputData,
          output_strides,
          long4(n, c, output_y, output_x),
          upsample_get_value_bounded<T>(
              inputData,
              input_sizes,
              input_strides,
              long4(n, c, src_y, src_x)));
    }
  }
}

struct BilinearFunctor {
  inline float operator()(float x) {
    x = abs(x);
    return x < 1.0 ? 1.0 - x : x;
  }
  static constant constexpr float area_factor = 1.0;
};

struct BicubicFunctor {
  inline float operator()(float x) {
    // https://en.wikipedia.org/wiki/Bicubic_interpolation#Bicubic_convolution_algorithm
    x = abs(x);
    if (x < 1.0) {
      return 1.0 + (1.5 * x - 2.5) * x * x;
    }
    if (x < 2.0) {
      return 2.0 - 0.5 * ((x - 5.0) * x + 8.0) * x;
    }
    return 0;
  }
  static constant constexpr float area_factor = 2.0;
};

template <typename T, typename F>
kernel void upsample_2d_aa(
    constant T* inputData [[buffer(0)]],
    device T* outputData [[buffer(1)]],
    constant UpsampleParams<4>& params [[buffer(2)]],
    uint thread_index [[thread_position_in_grid]]) {
  const auto input_strides = to_vec(params.input_strides);
  const auto output_strides = to_vec(params.output_strides);
  const auto input_sizes = to_vec(params.input_sizes);
  const auto output_sizes = to_vec(params.output_sizes);
  const float2 scales = float2(params.scales[0], params.scales[1]);
  auto output_x = thread_index % static_cast<uint>(output_sizes.w);
  auto output_y = thread_index / static_cast<uint>(output_sizes.w);
  F f;
  auto x_center = area_pixel_compute_source_index(
      scales.x,
      output_x,
      /*align_corners=*/false,
      /*cubic=*/F::area_factor == 2.0);
  auto y_center = area_pixel_compute_source_index(
      scales.y,
      output_y,
      /*align_corners=*/false,
      /*cubic=*/F::area_factor == 2.0);
  auto clamped_scales = max(1.0, scales);
  auto x_min =
      max(0L, long(floor(x_center - f.area_factor * clamped_scales.x + 1)));
  auto x_max = min(
      input_sizes.w, long(ceil(x_center + f.area_factor * clamped_scales.x)));
  auto y_min =
      max(0L, long(floor(y_center - f.area_factor * clamped_scales.y + 1)));
  auto y_max = min(
      input_sizes.z, long(ceil(y_center + f.area_factor * clamped_scales.y)));
  for (int n = 0; n < output_sizes.x; n++) {
    for (int c = 0; c < output_sizes.y; c++) {
      float res = 0.0;
      float ws = 0.0;
      constant auto* input =
          inputData + n * input_strides.x + c * input_strides.y;
      for (auto y = y_min; y < y_max; ++y) {
        auto dy = f((y - y_center) / clamped_scales.y);
        for (auto x = x_min; x < x_max; ++x) {
          auto dx = f((x - x_center) / clamped_scales.x);
          auto val = input[x * input_strides.w + y * input_strides.z];
          res += val * dx * dy;
          ws += dx * dy;
        }
      }
      upsample_set_value(
          outputData,
          output_strides,
          long4(n, c, output_y, output_x),
          static_cast<T>(res / ws));
    }
  }
}

template <typename T, typename F>
kernel void upsample_2d_aa_backward(
    device AtomicType_t<T>* gradInputData [[buffer(0)]],
    constant T* gradOutputData [[buffer(1)]],
    constant UpsampleParams<4>& params [[buffer(2)]],
    uint thread_index [[thread_position_in_grid]]) {
  const auto input_strides = to_vec(params.input_strides);
  const auto output_strides = to_vec(params.output_strides);
  const auto input_sizes = to_vec(params.input_sizes);
  const auto output_sizes = to_vec(params.output_sizes);
  const float2 scales = float2(params.scales[0], params.scales[1]);
  auto output_x = thread_index % static_cast<uint>(output_sizes.w);
  auto output_y = thread_index / static_cast<uint>(output_sizes.w);
  F f;
  auto x_center = area_pixel_compute_source_index(
      scales.x,
      output_x,
      /*align_corners=*/false,
      /*cubic=*/F::area_factor == 2.0);
  auto y_center = area_pixel_compute_source_index(
      scales.y,
      output_y,
      /*align_corners=*/false,
      /*cubic=*/F::area_factor == 2.0);
  auto clamped_scales = max(1.0, scales);
  auto x_min =
      max(0L, long(floor(x_center - f.area_factor * clamped_scales.x + 1)));
  auto x_max = min(
      input_sizes.w, long(ceil(x_center + f.area_factor * clamped_scales.x)));
  auto y_min =
      max(0L, long(floor(y_center - f.area_factor * clamped_scales.y + 1)));
  auto y_max = min(
      input_sizes.z, long(ceil(y_center + f.area_factor * clamped_scales.y)));
  float ws = 0.0;
  auto clamped_scales_recip = 1 / clamped_scales;
  for (auto y = y_min; y < y_max; ++y) {
    auto dy = f((y - y_center) * clamped_scales_recip.y);
    for (auto x = x_min; x < x_max; ++x) {
      ws += f((x - x_center) * clamped_scales_recip.x) * dy;
    }
  }
  for (int n = 0; n < output_sizes.x; n++) {
    for (int c = 0; c < output_sizes.y; c++) {
      auto grad_out_value = static_cast<float>(
          gradOutputData
              [n * output_strides.x + c * output_strides.y +
               output_y * output_strides.z + output_x * output_strides.w]);
      for (auto y = y_min; y < y_max; ++y) {
        auto dy = f((y - y_center) * clamped_scales_recip.y);
        for (auto x = x_min; x < x_max; ++x) {
          auto dx = f((x - x_center) * clamped_scales_recip.x);
          upsample_increment_value_bounded<T>(
              gradInputData,
              input_sizes,
              input_strides,
              long4(n, c, y, x),
              static_cast<T>(grad_out_value * dx * dy / ws));
        }
      }
    }
  }
}

template <typename T>
kernel void upsample_bicubic2d(
    constant T* inputData [[buffer(0)]],
    device T* outputData [[buffer(1)]],
    constant UpsampleParams<4>& params [[buffer(2)]],
    uint thread_index [[thread_position_in_grid]]) {
  const auto input_strides = to_vec(params.input_strides);
  const auto output_strides = to_vec(params.output_strides);
  const auto input_sizes = to_vec(params.input_sizes);
  const auto output_sizes = to_vec(params.output_sizes);
  const float2 scales = float2(params.scales[0], params.scales[1]);
  const bool align_corners = params.align_corners;
  auto output_x = thread_index % static_cast<uint>(output_sizes.w);
  auto output_y = thread_index / static_cast<uint>(output_sizes.w);
  auto real_x = area_pixel_compute_source_index(
      scales.x, output_x, align_corners, /*cubic=*/true);
  int in_x = floor(real_x);
  auto t_x = real_x - in_x;

  auto real_y = area_pixel_compute_source_index(
      scales.y, output_y, align_corners, /*cubic=*/true);
  int in_y = floor(real_y);
  auto t_y = real_y - in_y;
  for (int n = 0; n < output_sizes.x; n++) {
    for (int c = 0; c < output_sizes.y; c++) {
      float coefficients[4];
      for (int k = 0; k < 4; k++) {
        coefficients[k] = cubic_interp1d(
            upsample_get_value_bounded<T>(
                inputData,
                input_sizes,
                input_strides,
                long4(n, c, in_y - 1 + k, in_x - 1)),
            upsample_get_value_bounded<T>(
                inputData,
                input_sizes,
                input_strides,
                long4(n, c, in_y - 1 + k, in_x + 0)),
            upsample_get_value_bounded<T>(
                inputData,
                input_sizes,
                input_strides,
                long4(n, c, in_y - 1 + k, in_x + 1)),
            upsample_get_value_bounded<T>(
                inputData,
                input_sizes,
                input_strides,
                long4(n, c, in_y - 1 + k, in_x + 2)),
            t_x);
      }
      auto inp = static_cast<T>(cubic_interp1d(
          coefficients[0],
          coefficients[1],
          coefficients[2],
          coefficients[3],
          t_y));
      upsample_set_value(
          outputData, output_strides, long4(n, c, output_y, output_x), inp);
    }
  }
}

template <typename T>
kernel void upsample_bicubic2d_backward(
    device AtomicType_t<T>* gradInputData [[buffer(0)]],
    constant T* gradOutputData [[buffer(1)]],
    constant UpsampleParams<4>& params [[buffer(2)]],
    uint thread_index [[thread_position_in_grid]]) {
  const auto input_strides = to_vec(params.input_strides);
  const auto output_strides = to_vec(params.output_strides);
  const auto input_sizes = to_vec(params.input_sizes);
  const auto output_sizes = to_vec(params.output_sizes);
  const float2 scales = float2(params.scales[0], params.scales[1]);
  const bool align_corners = params.align_corners;
  auto output_x = thread_index % output_sizes.w;
  auto output_y = thread_index / output_sizes.w;
  auto real_x = area_pixel_compute_source_index<float>(
      scales.x, output_x, align_corners, /*cubic=*/true);
  int input_x = floor(real_x);
  float t_x = real_x - input_x;

  auto real_y = area_pixel_compute_source_index<float>(
      scales.y, output_y, align_corners, /*cubic=*/true);
  int input_y = floor(real_y);
  float t_y = real_y - input_y;

  float x_coeffs[4];
  float y_coeffs[4];

  get_cubic_coefficients(x_coeffs, t_x);
  get_cubic_coefficients(y_coeffs, t_y);

  for (int n = 0; n < output_sizes.x; n++) {
    for (int c = 0; c < output_sizes.y; ++c) {
      auto out_value = gradOutputData
          [n * output_strides.x + c * output_strides.y +
           output_y * output_strides.z + output_x * output_strides.w];
      for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
          upsample_increment_value_bounded<T>(
              gradInputData,
              input_sizes,
              input_strides,
              long4(n, c, input_y - 1 + i, input_x - 1 + j),
              out_value * y_coeffs[i] * x_coeffs[j]);
        }
      }
    }
  }
}

#define INSTANTIATE_UPSAMPLE_2D(NAME, DTYPE)                       \
  template [[host_name("upsample_" #NAME "_" #DTYPE)]] kernel void \
      upsample_##NAME<DTYPE>(                                      \
          constant DTYPE * inputData [[buffer(0)]],                \
          device DTYPE * outputData [[buffer(1)]],                 \
          constant UpsampleParams<4> & params [[buffer(2)]],       \
          uint thread_index [[thread_position_in_grid]])

#define INSTANTIATE_UPSAMPLE_2D_AA(NAME, FUNCTOR, DTYPE)           \
  template [[host_name("upsample_" #NAME "_" #DTYPE)]] kernel void \
  upsample_2d_aa<DTYPE, FUNCTOR>(                                  \
      constant DTYPE * inputData [[buffer(0)]],                    \
      device DTYPE * outputData [[buffer(1)]],                     \
      constant UpsampleParams<4> & params [[buffer(2)]],           \
      uint thread_index [[thread_position_in_grid]])

#define INSTANTIATE_UPSAMPLE_2D_AA_BACKWARD(NAME, FUNCTOR, DTYPE)           \
  template [[host_name("upsample_" #NAME "_backward_" #DTYPE)]] kernel void \
  upsample_2d_aa_backward<DTYPE, FUNCTOR>(                                  \
      device AtomicType_t<DTYPE> * gradInputData [[buffer(0)]],             \
      constant DTYPE * gradOutputData [[buffer(1)]],                        \
      constant UpsampleParams<4> & params [[buffer(2)]],                    \
      uint thread_index [[thread_position_in_grid]])

#define INSTANTIATE_UPSAMPLE_2D_BACKWARD(NAME, DTYPE)                       \
  template [[host_name("upsample_" #NAME "_backward_" #DTYPE)]] kernel void \
      upsample_##NAME##_backward<DTYPE>(                                    \
          device AtomicType_t<DTYPE> * gradInputData [[buffer(0)]],         \
          constant DTYPE * gradOutputData [[buffer(1)]],                    \
          constant UpsampleParams<4> & params [[buffer(2)]],                \
          uint thread_index [[thread_position_in_grid]])

#define INSTANTIATE_UPSAMPLE_LINEAR(DTYPE)                        \
  template [[host_name("upsample_linear1d_" #DTYPE)]] kernel void \
  upsample_linear1d<DTYPE>(                                       \
      constant DTYPE * inputData [[buffer(0)]],                   \
      device DTYPE * outputData [[buffer(1)]],                    \
      constant UpsampleParams<3> & params [[buffer(2)]],          \
      uint thread_index [[thread_position_in_grid]])

#define INSTANTIATE_UPSAMPLE_NEAREST_1D(NAME, EXACT, DTYPE)        \
  template [[host_name("upsample_" #NAME "_" #DTYPE)]] kernel void \
  upsample_nearest1d<DTYPE, EXACT>(                                \
      constant DTYPE * inputData [[buffer(0)]],                    \
      device DTYPE * outputData [[buffer(1)]],                     \
      constant UpsampleParams<3> & params [[buffer(2)]],           \
      uint thread_index [[thread_position_in_grid]])

#define INSTANTIATE_UPSAMPLE_NEAREST_2D(NAME, EXACT, DTYPE)        \
  template [[host_name("upsample_" #NAME "_" #DTYPE)]] kernel void \
  upsample_nearest2d<DTYPE, EXACT>(                                \
      constant DTYPE * inputData [[buffer(0)]],                    \
      device DTYPE * outputData [[buffer(1)]],                     \
      constant UpsampleParams<4> & params [[buffer(2)]],           \
      uint thread_index [[thread_position_in_grid]])

#define INSTANTIATE_UPSAMPLE_NEAREST(DTYPE)                      \
  INSTANTIATE_UPSAMPLE_NEAREST_1D(nearest1d, false, DTYPE);      \
  INSTANTIATE_UPSAMPLE_NEAREST_1D(nearest_exact1d, true, DTYPE); \
  INSTANTIATE_UPSAMPLE_NEAREST_2D(nearest2d, false, DTYPE);      \
  INSTANTIATE_UPSAMPLE_NEAREST_2D(nearest_exact2d, true, DTYPE)

#define INSTANTIATE_UPSAMPLE_3D(DTYPE)                                    \
  template [[host_name("upsample_nearest_3d_" #DTYPE)]] kernel void       \
  upsample_nearest_3d<DTYPE>(                                             \
      constant DTYPE * inputData [[buffer(0)]],                           \
      device DTYPE * outputData [[buffer(1)]],                            \
      constant UpsampleParams<5> & params [[buffer(2)]],                  \
      uint thread_index [[thread_position_in_grid]]);                     \
  template [[host_name("upsample_nearest_exact_3d_" #DTYPE)]] kernel void \
  upsample_nearest_exact_3d<DTYPE>(                                       \
      constant DTYPE * inputData [[buffer(0)]],                           \
      device DTYPE * outputData [[buffer(1)]],                            \
      constant UpsampleParams<5> & params [[buffer(2)]],                  \
      uint thread_index [[thread_position_in_grid]]);                     \
  template [[host_name("upsample_trilinear_" #DTYPE)]] kernel void        \
  upsample_trilinear<DTYPE>(                                              \
      constant DTYPE * inputData [[buffer(0)]],                           \
      device DTYPE * outputData [[buffer(1)]],                            \
      constant UpsampleParams<5> & params [[buffer(2)]],                  \
      uint thread_index [[thread_position_in_grid]])

#define INSTANTIATE_UPSAMPLE_3D_BACKWARD(DTYPE)                               \
  template [[host_name("upsample_nearest_3d_backward_" #DTYPE)]] kernel void  \
  upsample_nearest_3d_backward<DTYPE>(                                        \
      device AtomicType_t<DTYPE> * gradInputData [[buffer(0)]],               \
      constant DTYPE * gradOutputData [[buffer(1)]],                          \
      constant UpsampleParams<5> & params [[buffer(2)]],                      \
      uint thread_index [[thread_position_in_grid]]);                         \
  template                                                                    \
      [[host_name("upsample_nearest_exact_3d_backward_" #DTYPE)]] kernel void \
      upsample_nearest_exact_3d_backward<DTYPE>(                              \
          device AtomicType_t<DTYPE> * gradInputData [[buffer(0)]],           \
          constant DTYPE * gradOutputData [[buffer(1)]],                      \
          constant UpsampleParams<5> & params [[buffer(2)]],                  \
          uint thread_index [[thread_position_in_grid]]);                     \
  template [[host_name("upsample_trilinear_backward_" #DTYPE)]] kernel void   \
  upsample_trilinear_backward<DTYPE>(                                         \
      device AtomicType_t<DTYPE> * gradInputData [[buffer(0)]],               \
      constant DTYPE * gradOutputData [[buffer(1)]],                          \
      constant UpsampleParams<5> & params [[buffer(2)]],                      \
      uint thread_index [[thread_position_in_grid]]);

#define INSTANTIATE_UPSAMPLE_ALL(DTYPE)                                       \
  INSTANTIATE_UPSAMPLE_2D(bicubic2d, DTYPE);                                  \
  INSTANTIATE_UPSAMPLE_2D_AA(bicubic2d_aa, BicubicFunctor, DTYPE);            \
  INSTANTIATE_UPSAMPLE_2D_AA_BACKWARD(bicubic2d_aa, BicubicFunctor, DTYPE);   \
  INSTANTIATE_UPSAMPLE_2D_BACKWARD(bicubic2d, DTYPE);                         \
  INSTANTIATE_UPSAMPLE_2D(bilinear2d, DTYPE);                                 \
  INSTANTIATE_UPSAMPLE_2D_AA(bilinear2d_aa, BilinearFunctor, DTYPE);          \
  INSTANTIATE_UPSAMPLE_2D_AA_BACKWARD(bilinear2d_aa, BilinearFunctor, DTYPE); \
  INSTANTIATE_UPSAMPLE_LINEAR(DTYPE);                                         \
  INSTANTIATE_UPSAMPLE_3D_BACKWARD(DTYPE);                                    \
  INSTANTIATE_UPSAMPLE_3D(DTYPE)

INSTANTIATE_UPSAMPLE_2D(bilinear2d, uchar);
INSTANTIATE_UPSAMPLE_3D(uchar);
INSTANTIATE_UPSAMPLE_ALL(float);
INSTANTIATE_UPSAMPLE_ALL(half);
INSTANTIATE_UPSAMPLE_ALL(bfloat);

// Nearest 1d/2d forward is a pure gather, so cover every dtype the MPSGraph
// path supported to avoid a coverage regression.
INSTANTIATE_UPSAMPLE_NEAREST(float);
INSTANTIATE_UPSAMPLE_NEAREST(half);
INSTANTIATE_UPSAMPLE_NEAREST(bfloat);
INSTANTIATE_UPSAMPLE_NEAREST(uchar);
INSTANTIATE_UPSAMPLE_NEAREST(char);
INSTANTIATE_UPSAMPLE_NEAREST(short);
INSTANTIATE_UPSAMPLE_NEAREST(int);
INSTANTIATE_UPSAMPLE_NEAREST(long);
INSTANTIATE_UPSAMPLE_NEAREST(bool);
