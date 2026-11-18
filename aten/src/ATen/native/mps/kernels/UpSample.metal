#include <ATen/native/mps/kernels/SamplingHelpers.h>
#include <ATen/native/mps/kernels/UpSample.h>
#include <c10/metal/atomic.h>
#include <metal_stdlib>

using namespace metal;
using namespace c10::metal;

template <typename accscalar_t, typename dst_t>
accscalar_t area_pixel_compute_source_index(
    accscalar_t scale,
    dst_t dst_index,
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
    device const scalar_t* data,
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
    device const T* inputData [[buffer(0)]],
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
void upsample_nearest_exact_3d_backward_impl(
    device AtomicType_t<T>* gradInputData,
    device const T* gradOutputData,
    constant UpsampleParams<5>& params,
    uint thread_index,
    uint n_begin,
    uint n_end,
    uint c_begin,
    uint c_end) {
  const auto input_sizes = uint3(
      params.input_sizes[4], params.input_sizes[3], params.input_sizes[2]);
  const auto output = coords_from_threadidx(params, thread_index);
  const auto real = coords_to_real_coords(params, output, false);
  for (uint n = n_begin; n < n_end; n++) {
    for (uint c = c_begin; c < c_end; c++) {
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
    device const T* inputData [[buffer(0)]],
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
void upsample_nearest_3d_backward_impl(
    device AtomicType_t<T>* gradInputData,
    device const T* gradOutputData,
    constant UpsampleParams<5>& params,
    uint thread_index,
    uint n_begin,
    uint n_end,
    uint c_begin,
    uint c_end) {
  const auto input_sizes = uint3(
      params.input_sizes[4], params.input_sizes[3], params.input_sizes[2]);
  const auto output = coords_from_threadidx(params, thread_index);
  const auto real = coords_to_real_coords(params, output, true);
  for (uint n = n_begin; n < n_end; n++) {
    for (uint c = c_begin; c < c_end; c++) {
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
    device const T* inputData [[buffer(0)]],
    device T* outputData [[buffer(1)]],
    constant UpsampleParams<5>& params [[buffer(2)]],
    uint thread_index [[thread_position_in_grid]]) {
  const auto input_sizes = uint3(
      params.input_sizes[4], params.input_sizes[3], params.input_sizes[2]);
  const auto output = coords_from_threadidx(params, thread_index);
  const auto real = coords_to_real_coords(params, output, params.align_corners);
  auto t = fract(real);
  const auto input = uint3(floor(real));
  for (uint n = 0; n < params.output_sizes[0]; n++) {
    for (uint c = 0; c < params.output_sizes[1]; c++) {
      auto i000 = upsample_get_value_bounded<T>(
          inputData,
          input_sizes,
          params.input_strides,
          n,
          c,
          input.z,
          input.y,
          input.x);
      auto i001 = upsample_get_value_bounded<T>(
          inputData,
          input_sizes,
          params.input_strides,
          n,
          c,
          input.z,
          input.y,
          input.x + 1);
      auto i010 = upsample_get_value_bounded<T>(
          inputData,
          input_sizes,
          params.input_strides,
          n,
          c,
          input.z,
          input.y + 1,
          input.x);
      auto i011 = upsample_get_value_bounded<T>(
          inputData,
          input_sizes,
          params.input_strides,
          n,
          c,
          input.z,
          input.y + 1,
          input.x + 1);
      auto i100 = upsample_get_value_bounded<T>(
          inputData,
          input_sizes,
          params.input_strides,
          n,
          c,
          input.z + 1,
          input.y,
          input.x);
      auto i101 = upsample_get_value_bounded<T>(
          inputData,
          input_sizes,
          params.input_strides,
          n,
          c,
          input.z + 1,
          input.y,
          input.x + 1);
      auto i110 = upsample_get_value_bounded<T>(
          inputData,
          input_sizes,
          params.input_strides,
          n,
          c,
          input.z + 1,
          input.y + 1,
          input.x);
      auto i111 = upsample_get_value_bounded<T>(
          inputData,
          input_sizes,
          params.input_strides,
          n,
          c,
          input.z + 1,
          input.y + 1,
          input.x + 1);
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
void upsample_trilinear_backward_impl(
    device AtomicType_t<T>* gradInputData,
    device const T* gradOutputData,
    constant UpsampleParams<5>& params,
    uint thread_index,
    uint n_begin,
    uint n_end,
    uint c_begin,
    uint c_end) {
  const auto input_sizes = uint3(
      params.input_sizes[4], params.input_sizes[3], params.input_sizes[2]);
  const auto output = coords_from_threadidx(params, thread_index);
  const auto real = coords_to_real_coords(params, output, params.align_corners);
  auto t = fract(real);
  const auto input = uint3(floor(real));
  for (uint n = n_begin; n < n_end; n++) {
    for (uint c = c_begin; c < c_end; c++) {
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
            input.z + ((d & 4) >> 2),
            input.y + ((d & 2) >> 1),
            input.x + (d & 1),
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

// The forward reads landing on an input index: their output run [lo, hi) and
// tap weights. The first and last rows also take the reads clamped onto them.
struct LinearSource {
  float scale;
  bool align_corners;
  LinearSource(float scale_, bool align_corners_)
      : scale(scale_), align_corners(align_corners_) {}
  inline float operator()(int dst) const {
    return area_pixel_compute_source_index(
        scale, dst, align_corners, /*cubic=*/false);
  }
  // Analytic inverse of [idx - 1, idx + 1), grown a step where the forward's
  // rounding still gives a nonzero tap; scale 0 (align_corners, size-1 dim)
  // sends every output to source 0.
  inline int2 run(int idx, int input_size, int output_size) const {
    if (scale == 0.0f) {
      return int2(0, idx == 0 ? output_size : 0);
    }
    const float inv_scale = 1.0f / scale;
    const float offset = align_corners ? 0.0f : 0.5f * inv_scale - 0.5f;
    int lo = static_cast<int>(floor((idx - 1) * inv_scale + offset)) + 1;
    int hi = static_cast<int>(ceil((idx + 1) * inv_scale + offset));
    lo -= (*this)(lo - 1) > idx - 1;
    hi += (*this)(hi) < idx + 1;
    lo = idx == 0 ? 0 : lo;
    hi = idx == input_size - 1 ? output_size : hi;
    return int2(clamp(lo, 0, output_size), clamp(hi, 0, output_size));
  }
  // 1 - t on the lower tap, t on the upper, 1 where both collapse onto the
  // last row.
  inline float weight(int dst, int idx, int input_size) const {
    const auto src =
        min(area_pixel_compute_source_index(scale, dst, align_corners, false),
            static_cast<float>(input_size - 1));
    return max(0.0f, 1.0f - abs(idx - src));
  }
};

template <bool exact, typename index_t = int>
struct NearestSource {
  float scale;
  NearestSource(float scale_, bool) : scale(scale_) {}
  // floor(scale * dst) for nearest, floor(scale * (dst + 0.5)) for
  // nearest-exact. scale is the input/output ratio, so truncation (== floor for
  // the non-negative product) matches the CPU reference. Computing this in
  // Metal (built with -fno-fast-math) is bit-stable across macOS versions,
  // unlike the MPSGraph resize which mis-rounds exact boundaries on macOS 14.
  // The float to index_t cast saturates, so callers can clamp in index_t.
  template <typename dst_t>
  inline index_t operator()(dst_t dst) const {
    return static_cast<index_t>(exact ? scale * (dst + 0.5f) : scale * dst);
  }
  // Analytic inverse of [idx, idx + 1), moved a step to match the forward's
  // rounding.
  inline int2 run(int idx, int input_size, int output_size) const {
    const float inv_scale = 1.0f / scale;
    const float offset = exact ? -0.5f : 0.0f;
    int lo =
        clamp(static_cast<int>(ceil(idx * inv_scale + offset)), 1, output_size);
    int hi = clamp(
        static_cast<int>(ceil((idx + 1) * inv_scale + offset)), 1, output_size);
    lo += ((*this)(lo) < idx) - ((*this)(lo - 1) >= idx);
    hi += ((*this)(hi) < idx + 1) - ((*this)(hi - 1) >= idx + 1);
    lo = idx == 0 ? 0 : lo;
    hi = idx == input_size - 1 ? output_size : hi;
    return int2(clamp(lo, 0, output_size), clamp(hi, 0, output_size));
  }
  static inline float weight(int, int, int) {
    return 1.0f;
  }
};

// A thread owns one (y, x) of the output and strides over the (n, c) planes
// left out of the grid, like upsample_gather_backward.
template <typename T, bool exact, typename index_t>
kernel void upsample_nearest2d(
    device const T* inputData [[buffer(0)]],
    device T* outputData [[buffer(1)]],
    constant UpsampleParams<4>& params [[buffer(2)]],
    constant ::c10::IntDivider<uint>& channels_div [[buffer(3)]],
    constant long2& origin [[buffer(4)]],
    uint3 thread_index [[thread_position_in_grid]],
    uint3 grid_size [[threads_per_grid]]) {
  const auto input_strides = vec<index_t, 4>(to_vec(params.input_strides));
  const auto output_strides = vec<index_t, 4>(to_vec(params.output_strides));
  const ::c10::IntDivider<uint> channels = channels_div;
  const auto planes =
      static_cast<uint>(params.output_sizes[0]) * channels.divisor;
  const NearestSource<exact, index_t> source_x(params.scales[0], false);
  const NearestSource<exact, index_t> source_y(params.scales[1], false);
  const index_t x = static_cast<index_t>(origin.x) + thread_index.x;
  const index_t y = static_cast<index_t>(origin.y) + thread_index.y;
  const auto src_x =
      min(source_x(x), static_cast<index_t>(params.input_sizes[3]) - 1);
  const auto src_y =
      min(source_y(y), static_cast<index_t>(params.input_sizes[2]) - 1);
  device const auto* input =
      inputData + src_y * input_strides.z + src_x * input_strides.w;
  device auto* output =
      outputData + y * output_strides.z + x * output_strides.w;
  for (auto plane = thread_index.z; plane < planes; plane += grid_size.z) {
    const auto nc = channels.divmod(plane);
    const auto n = nc.div;
    const auto c = nc.mod;
    output[n * output_strides.x + c * output_strides.y] =
        input[n * input_strides.x + c * input_strides.y];
  }
}

// Like upsample_nearest2d, with the four taps and their weights computed once
// per (y, x) and reused for every plane.
template <typename T, typename index_t>
kernel void upsample_bilinear2d(
    device const T* inputData [[buffer(0)]],
    device T* outputData [[buffer(1)]],
    constant UpsampleParams<4>& params [[buffer(2)]],
    constant ::c10::IntDivider<uint>& channels_div [[buffer(3)]],
    constant long2& origin [[buffer(4)]],
    uint3 thread_index [[thread_position_in_grid]],
    uint3 grid_size [[threads_per_grid]]) {
  const auto input_strides = vec<index_t, 4>(to_vec(params.input_strides));
  const auto output_strides = vec<index_t, 4>(to_vec(params.output_strides));
  const ::c10::IntDivider<uint> channels = channels_div;
  const auto planes =
      static_cast<uint>(params.output_sizes[0]) * channels.divisor;
  const index_t x = static_cast<index_t>(origin.x) + thread_index.x;
  const index_t y = static_cast<index_t>(origin.y) + thread_index.y;
  const auto real_x = area_pixel_compute_source_index(
      params.scales[0], x, params.align_corners, /*cubic=*/false);
  const auto real_y = area_pixel_compute_source_index(
      params.scales[1], y, params.align_corners, /*cubic=*/false);
  const auto t_x = fract(real_x);
  const auto t_y = fract(real_y);
  const index_t max_x = params.input_sizes[3] - 1;
  const index_t max_y = params.input_sizes[2] - 1;
  const auto x0 = static_cast<index_t>(floor(real_x));
  const auto y0 = static_cast<index_t>(floor(real_y));
  const index_t col0 = clamp(x0, index_t(0), max_x) * input_strides.w;
  const index_t col1 = clamp(x0 + 1, index_t(0), max_x) * input_strides.w;
  device const T* row0 =
      inputData + clamp(y0, index_t(0), max_y) * input_strides.z;
  device const T* row1 =
      inputData + clamp(y0 + 1, index_t(0), max_y) * input_strides.z;
  device auto* output =
      outputData + y * output_strides.z + x * output_strides.w;
  for (auto plane = thread_index.z; plane < planes; plane += grid_size.z) {
    const auto nc = channels.divmod(plane);
    const index_t in_off = nc.div * input_strides.x + nc.mod * input_strides.y;
    const auto i0 =
        linear_interp(row0[in_off + col0], row0[in_off + col1], t_x);
    const auto i1 =
        linear_interp(row1[in_off + col0], row1[in_off + col1], t_x);
    output[nc.div * output_strides.x + nc.mod * output_strides.y] =
        static_cast<T>(linear_interp(i0, i1, t_y));
  }
}

// A thread owns one (y, x) of grad_input, strides over the (n, c) planes left
// out of the grid and sums the grad_output run that read each element in float.
template <typename T, typename Source, typename index_t>
kernel void upsample_gather_backward(
    device T* gradInputData [[buffer(0)]],
    device const T* gradOutputData [[buffer(1)]],
    constant UpsampleParams<4>& params [[buffer(2)]],
    constant ::c10::IntDivider<uint>& channels_div [[buffer(3)]],
    constant long2& origin [[buffer(4)]],
    uint3 thread_index [[thread_position_in_grid]],
    uint3 grid_size [[threads_per_grid]]) {
  const auto input_strides = vec<index_t, 4>(to_vec(params.input_strides));
  const auto output_strides = vec<index_t, 4>(to_vec(params.output_strides));
  const auto input_sizes = to_vec(params.input_sizes);
  const auto output_sizes = to_vec(params.output_sizes);
  const int input_x = static_cast<int>(origin.x) + thread_index.x;
  const int input_y = static_cast<int>(origin.y) + thread_index.y;
  const ::c10::IntDivider<uint> channels = channels_div;
  const auto planes = static_cast<uint>(input_sizes.x) * channels.divisor;
  const Source source_x(params.scales[0], params.align_corners);
  const Source source_y(params.scales[1], params.align_corners);
  const auto x_run = source_x.run(input_x, input_sizes.w, output_sizes.w);
  const auto y_run = source_y.run(input_y, input_sizes.z, output_sizes.z);
  device auto* grad_input =
      gradInputData + input_y * input_strides.z + input_x * input_strides.w;
  for (auto plane = thread_index.z; plane < planes; plane += grid_size.z) {
    const auto nc = channels.divmod(plane);
    const auto n = nc.div;
    const auto c = nc.mod;
    device const auto* grad_output =
        gradOutputData + n * output_strides.x + c * output_strides.y;
    float res = 0;
    for (auto y = y_run.x; y < y_run.y; y++) {
      const auto w_y = source_y.weight(y, input_y, input_sizes.z);
      device const auto* row =
          grad_output + y * output_strides.z + x_run.x * output_strides.w;
      for (auto x = x_run.x; x < x_run.y; x++, row += output_strides.w) {
        res += w_y * source_x.weight(x, input_x, input_sizes.w) *
            static_cast<float>(*row);
      }
    }
    grad_input[n * input_strides.x + c * input_strides.y] = static_cast<T>(res);
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

// Like upsample_bilinear2d, with the filter window and its weight sum computed
// once per (y, x) and reused for every plane.
template <typename T, typename F, typename index_t>
kernel void upsample_2d_aa(
    device const T* inputData [[buffer(0)]],
    device T* outputData [[buffer(1)]],
    constant UpsampleParams<4>& params [[buffer(2)]],
    constant ::c10::IntDivider<uint>& channels_div [[buffer(3)]],
    constant long2& origin [[buffer(4)]],
    uint3 thread_index [[thread_position_in_grid]],
    uint3 grid_size [[threads_per_grid]]) {
  const auto input_strides = vec<index_t, 4>(to_vec(params.input_strides));
  const auto output_strides = vec<index_t, 4>(to_vec(params.output_strides));
  const ::c10::IntDivider<uint> channels = channels_div;
  const auto planes =
      static_cast<uint>(params.output_sizes[0]) * channels.divisor;
  const index_t output_x = static_cast<index_t>(origin.x) + thread_index.x;
  const index_t output_y = static_cast<index_t>(origin.y) + thread_index.y;
  F f;
  const auto x_center = area_pixel_compute_source_index(
      params.scales[0],
      output_x,
      /*align_corners=*/false,
      /*cubic=*/F::area_factor == 2.0);
  const auto y_center = area_pixel_compute_source_index(
      params.scales[1],
      output_y,
      /*align_corners=*/false,
      /*cubic=*/F::area_factor == 2.0);
  const auto clamped_scales =
      max(1.0, float2(params.scales[0], params.scales[1]));
  const auto x_min =
      max(index_t(0),
          static_cast<index_t>(
              floor(x_center - f.area_factor * clamped_scales.x + 1)));
  const auto x_max = min(
      static_cast<index_t>(params.input_sizes[3]),
      static_cast<index_t>(ceil(x_center + f.area_factor * clamped_scales.x)));
  const auto y_min =
      max(index_t(0),
          static_cast<index_t>(
              floor(y_center - f.area_factor * clamped_scales.y + 1)));
  const auto y_max = min(
      static_cast<index_t>(params.input_sizes[2]),
      static_cast<index_t>(ceil(y_center + f.area_factor * clamped_scales.y)));
  float ws = 0.0;
  for (auto y = y_min; y < y_max; ++y) {
    const auto dy = f((y - y_center) / clamped_scales.y);
    for (auto x = x_min; x < x_max; ++x) {
      ws += f((x - x_center) / clamped_scales.x) * dy;
    }
  }
  device auto* output =
      outputData + output_y * output_strides.z + output_x * output_strides.w;
  for (auto plane = thread_index.z; plane < planes; plane += grid_size.z) {
    const auto nc = channels.divmod(plane);
    device const auto* input =
        inputData + nc.div * input_strides.x + nc.mod * input_strides.y;
    float res = 0.0;
    for (auto y = y_min; y < y_max; ++y) {
      const auto dy = f((y - y_center) / clamped_scales.y);
      device const auto* row = input + y * input_strides.z;
      for (auto x = x_min; x < x_max; ++x) {
        const auto dx = f((x - x_center) / clamped_scales.x);
        res += static_cast<float>(row[x * input_strides.w]) * dx * dy;
      }
    }
    output[nc.div * output_strides.x + nc.mod * output_strides.y] =
        static_cast<T>(res / ws);
  }
}

template <typename T, typename F>
void upsample_2d_aa_backward_impl(
    device AtomicType_t<T>* gradInputData,
    device const T* gradOutputData,
    constant UpsampleParams<4>& params,
    uint thread_index,
    uint n_begin,
    uint n_end,
    uint c_begin,
    uint c_end) {
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
  for (uint n = n_begin; n < n_end; n++) {
    for (uint c = c_begin; c < c_end; c++) {
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

// Like upsample_bilinear2d, with the 4x4 clamped taps and their cubic
// coefficients computed once per (y, x) and reused for every plane.
template <typename T, typename index_t>
kernel void upsample_bicubic2d(
    device const T* inputData [[buffer(0)]],
    device T* outputData [[buffer(1)]],
    constant UpsampleParams<4>& params [[buffer(2)]],
    constant ::c10::IntDivider<uint>& channels_div [[buffer(3)]],
    constant long2& origin [[buffer(4)]],
    uint3 thread_index [[thread_position_in_grid]],
    uint3 grid_size [[threads_per_grid]]) {
  const auto input_strides = vec<index_t, 4>(to_vec(params.input_strides));
  const auto output_strides = vec<index_t, 4>(to_vec(params.output_strides));
  const ::c10::IntDivider<uint> channels = channels_div;
  const auto planes =
      static_cast<uint>(params.output_sizes[0]) * channels.divisor;
  const index_t x = static_cast<index_t>(origin.x) + thread_index.x;
  const index_t y = static_cast<index_t>(origin.y) + thread_index.y;
  const auto real_x = area_pixel_compute_source_index(
      params.scales[0], x, params.align_corners, /*cubic=*/true);
  const auto real_y = area_pixel_compute_source_index(
      params.scales[1], y, params.align_corners, /*cubic=*/true);
  const auto in_x = static_cast<index_t>(floor(real_x));
  const auto in_y = static_cast<index_t>(floor(real_y));
  float x_coeffs[4];
  float y_coeffs[4];
  get_cubic_coefficients(x_coeffs, real_x - in_x);
  get_cubic_coefficients(y_coeffs, real_y - in_y);
  const index_t max_x = params.input_sizes[3] - 1;
  const index_t max_y = params.input_sizes[2] - 1;
  index_t cols[4];
  index_t rows[4];
  for (int k = 0; k < 4; k++) {
    cols[k] = clamp(in_x - 1 + k, index_t(0), max_x) * input_strides.w;
    rows[k] = clamp(in_y - 1 + k, index_t(0), max_y) * input_strides.z;
  }
  device auto* output =
      outputData + y * output_strides.z + x * output_strides.w;
  for (auto plane = thread_index.z; plane < planes; plane += grid_size.z) {
    const auto nc = channels.divmod(plane);
    device const T* input =
        inputData + nc.div * input_strides.x + nc.mod * input_strides.y;
    float res = 0;
    for (int k = 0; k < 4; k++) {
      device const T* row = input + rows[k];
      res += y_coeffs[k] *
          (row[cols[0]] * x_coeffs[0] + row[cols[1]] * x_coeffs[1] +
           row[cols[2]] * x_coeffs[2] + row[cols[3]] * x_coeffs[3]);
    }
    output[nc.div * output_strides.x + nc.mod * output_strides.y] =
        static_cast<T>(res);
  }
}

template <typename T>
void upsample_bicubic2d_backward_impl(
    device AtomicType_t<T>* gradInputData,
    device const T* gradOutputData,
    constant UpsampleParams<4>& params,
    uint thread_index,
    uint n_begin,
    uint n_end,
    uint c_begin,
    uint c_end) {
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

  for (uint n = n_begin; n < n_end; n++) {
    for (uint c = c_begin; c < c_end; ++c) {
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

template <
    typename T,
    unsigned N,
    void (*impl)(
        device AtomicType_t<T>*,
        device const T*,
        constant UpsampleParams<N>&,
        uint,
        uint,
        uint,
        uint,
        uint),
    bool serial>
kernel void upsample_backward(
    device AtomicType_t<T>* gradInputData [[buffer(0)]],
    device const T* gradOutputData [[buffer(1)]],
    constant UpsampleParams<N>& params [[buffer(2)]],
    uint thread_index [[thread_position_in_grid]]) {
  const uint n_size = params.output_sizes[0];
  const uint c_size = params.output_sizes[1];
  if IF_CONSTEXPR (!serial) {
    impl(
        gradInputData,
        gradOutputData,
        params,
        thread_index,
        0,
        n_size,
        0,
        c_size);
    return;
  }
  uint spatial_size = 1;
  for (unsigned dim = 2; dim < N; dim++) {
    spatial_size *= params.output_sizes[dim];
  }
  const uint n = thread_index / c_size;
  const uint c = thread_index % c_size;
  for (uint spatial_idx = 0; spatial_idx < spatial_size; spatial_idx++) {
    impl(
        gradInputData, gradOutputData, params, spatial_idx, n, n + 1, c, c + 1);
  }
}

#define INSTANTIATE_UPSAMPLE_BACKWARD(NAME, N, IMPL, DTYPE, SERIAL)      \
  template[[host_name(                                                   \
      "upsample_" #NAME                                                  \
      "_backward_" #DTYPE C10_METAL_SERIAL_SUFFIX(SERIAL))]] kernel void \
  upsample_backward<DTYPE, N, IMPL, SERIAL>(                             \
      device AtomicType_t<DTYPE> * gradInputData [[buffer(0)]],          \
      device const DTYPE* gradOutputData [[buffer(1)]],                  \
      constant UpsampleParams<N>& params [[buffer(2)]],                  \
      uint thread_index [[thread_position_in_grid]])

#define INSTANTIATE_UPSAMPLE_2D_AA_BACKWARD(NAME, FUNCTOR, DTYPE, SERIAL) \
  INSTANTIATE_UPSAMPLE_BACKWARD(                                          \
      NAME, 4, (upsample_2d_aa_backward_impl<DTYPE, FUNCTOR>), DTYPE, SERIAL)

#define INSTANTIATE_UPSAMPLE_GATHER_BACKWARD_IDX(                   \
    NAME, SOURCE, DTYPE, IDX, SUFFIX)                               \
  template [[host_name("upsample_" #NAME "_backward_" #DTYPE        \
                       "_" #SUFFIX)]] kernel void                   \
  upsample_gather_backward<DTYPE, SOURCE, IDX>(                     \
      device DTYPE * gradInputData [[buffer(0)]],                   \
      device const DTYPE* gradOutputData [[buffer(1)]],             \
      constant UpsampleParams<4>& params [[buffer(2)]],             \
      constant ::c10::IntDivider<uint>& channels_div [[buffer(3)]], \
      constant long2& origin [[buffer(4)]],                         \
      uint3 thread_index [[thread_position_in_grid]],               \
      uint3 grid_size [[threads_per_grid]])

#define INSTANTIATE_UPSAMPLE_GATHER_BACKWARD(NAME, SOURCE, DTYPE)          \
  INSTANTIATE_UPSAMPLE_GATHER_BACKWARD_IDX(NAME, SOURCE, DTYPE, int, u32); \
  INSTANTIATE_UPSAMPLE_GATHER_BACKWARD_IDX(NAME, SOURCE, DTYPE, long, u64)

#define INSTANTIATE_UPSAMPLE_2D_BACKWARD(NAME, DTYPE, SERIAL) \
  INSTANTIATE_UPSAMPLE_BACKWARD(                              \
      NAME, 4, upsample_##NAME##_backward_impl<DTYPE>, DTYPE, SERIAL)

// Forward kernels on the plane grid; the template arguments follow KERNEL.
#define INSTANTIATE_UPSAMPLE_PLANE_GRID(NAME, DTYPE, SUFFIX, KERNEL, ...) \
  template [[host_name("upsample_" #NAME "_" #DTYPE "_" #SUFFIX)]]        \
  kernel void KERNEL<__VA_ARGS__>(                                        \
      device const DTYPE* inputData [[buffer(0)]],                        \
      device DTYPE* outputData [[buffer(1)]],                             \
      constant UpsampleParams<4>& params [[buffer(2)]],                   \
      constant ::c10::IntDivider<uint>& channels_div [[buffer(3)]],       \
      constant long2& origin [[buffer(4)]],                               \
      uint3 thread_index [[thread_position_in_grid]],                     \
      uint3 grid_size [[threads_per_grid]])

#define INSTANTIATE_UPSAMPLE_BILINEAR(DTYPE)                    \
  INSTANTIATE_UPSAMPLE_PLANE_GRID(                              \
      bilinear2d, DTYPE, u32, upsample_bilinear2d, DTYPE, int); \
  INSTANTIATE_UPSAMPLE_PLANE_GRID(                              \
      bilinear2d, DTYPE, u64, upsample_bilinear2d, DTYPE, long)

#define INSTANTIATE_UPSAMPLE_BICUBIC(DTYPE)                   \
  INSTANTIATE_UPSAMPLE_PLANE_GRID(                            \
      bicubic2d, DTYPE, u32, upsample_bicubic2d, DTYPE, int); \
  INSTANTIATE_UPSAMPLE_PLANE_GRID(                            \
      bicubic2d, DTYPE, u64, upsample_bicubic2d, DTYPE, long)

#define INSTANTIATE_UPSAMPLE_2D_AA(NAME, FUNCTOR, DTYPE)      \
  INSTANTIATE_UPSAMPLE_PLANE_GRID(                            \
      NAME, DTYPE, u32, upsample_2d_aa, DTYPE, FUNCTOR, int); \
  INSTANTIATE_UPSAMPLE_PLANE_GRID(                            \
      NAME, DTYPE, u64, upsample_2d_aa, DTYPE, FUNCTOR, long)

#define INSTANTIATE_UPSAMPLE_NEAREST(DTYPE)                               \
  INSTANTIATE_UPSAMPLE_PLANE_GRID(                                        \
      nearest2d, DTYPE, u32, upsample_nearest2d, DTYPE, false, int);      \
  INSTANTIATE_UPSAMPLE_PLANE_GRID(                                        \
      nearest2d, DTYPE, u64, upsample_nearest2d, DTYPE, false, long);     \
  INSTANTIATE_UPSAMPLE_PLANE_GRID(                                        \
      nearest_exact2d, DTYPE, u32, upsample_nearest2d, DTYPE, true, int); \
  INSTANTIATE_UPSAMPLE_PLANE_GRID(                                        \
      nearest_exact2d, DTYPE, u64, upsample_nearest2d, DTYPE, true, long)

#define INSTANTIATE_UPSAMPLE_3D(DTYPE)                                    \
  template [[host_name("upsample_nearest_3d_" #DTYPE)]] kernel void       \
  upsample_nearest_3d<DTYPE>(                                             \
      device const DTYPE* inputData [[buffer(0)]],                        \
      device DTYPE* outputData [[buffer(1)]],                             \
      constant UpsampleParams<5>& params [[buffer(2)]],                   \
      uint thread_index [[thread_position_in_grid]]);                     \
  template [[host_name("upsample_nearest_exact_3d_" #DTYPE)]] kernel void \
  upsample_nearest_exact_3d<DTYPE>(                                       \
      device const DTYPE* inputData [[buffer(0)]],                        \
      device DTYPE* outputData [[buffer(1)]],                             \
      constant UpsampleParams<5>& params [[buffer(2)]],                   \
      uint thread_index [[thread_position_in_grid]]);                     \
  template [[host_name("upsample_trilinear_" #DTYPE)]] kernel void        \
  upsample_trilinear<DTYPE>(                                              \
      device const DTYPE* inputData [[buffer(0)]],                        \
      device DTYPE* outputData [[buffer(1)]],                             \
      constant UpsampleParams<5>& params [[buffer(2)]],                   \
      uint thread_index [[thread_position_in_grid]])

#define INSTANTIATE_UPSAMPLE_3D_BACKWARD(DTYPE, SERIAL)                        \
  INSTANTIATE_UPSAMPLE_BACKWARD(                                               \
      nearest_3d, 5, upsample_nearest_3d_backward_impl<DTYPE>, DTYPE, SERIAL); \
  INSTANTIATE_UPSAMPLE_BACKWARD(                                               \
      nearest_exact_3d,                                                        \
      5,                                                                       \
      upsample_nearest_exact_3d_backward_impl<DTYPE>,                          \
      DTYPE,                                                                   \
      SERIAL);                                                                 \
  INSTANTIATE_UPSAMPLE_BACKWARD(                                               \
      trilinear, 5, upsample_trilinear_backward_impl<DTYPE>, DTYPE, SERIAL)

#define INSTANTIATE_UPSAMPLE_ALL(DTYPE)                                  \
  INSTANTIATE_UPSAMPLE_BICUBIC(DTYPE);                                   \
  INSTANTIATE_UPSAMPLE_2D_AA(bicubic2d_aa, BicubicFunctor, DTYPE);       \
  INSTANTIATE_UPSAMPLE_2D_AA_BACKWARD(                                   \
      bicubic2d_aa, BicubicFunctor, DTYPE, false);                       \
  INSTANTIATE_UPSAMPLE_2D_BACKWARD(bicubic2d, DTYPE, false);             \
  INSTANTIATE_UPSAMPLE_BILINEAR(DTYPE);                                  \
  INSTANTIATE_UPSAMPLE_GATHER_BACKWARD(bilinear2d, LinearSource, DTYPE); \
  INSTANTIATE_UPSAMPLE_GATHER_BACKWARD(                                  \
      nearest2d, NearestSource<false>, DTYPE);                           \
  INSTANTIATE_UPSAMPLE_GATHER_BACKWARD(                                  \
      nearest_exact2d, NearestSource<true>, DTYPE);                      \
  INSTANTIATE_UPSAMPLE_2D_AA(bilinear2d_aa, BilinearFunctor, DTYPE);     \
  INSTANTIATE_UPSAMPLE_2D_AA_BACKWARD(                                   \
      bilinear2d_aa, BilinearFunctor, DTYPE, false);                     \
  INSTANTIATE_UPSAMPLE_3D_BACKWARD(DTYPE, false);                        \
  INSTANTIATE_UPSAMPLE_3D(DTYPE)

INSTANTIATE_UPSAMPLE_BILINEAR(uchar);
INSTANTIATE_UPSAMPLE_3D(uchar);
INSTANTIATE_UPSAMPLE_ALL(float);
INSTANTIATE_UPSAMPLE_ALL(half);
INSTANTIATE_UPSAMPLE_ALL(bfloat);

INSTANTIATE_UPSAMPLE_2D_AA_BACKWARD(bicubic2d_aa, BicubicFunctor, float, true);
INSTANTIATE_UPSAMPLE_2D_BACKWARD(bicubic2d, float, true);
INSTANTIATE_UPSAMPLE_2D_AA_BACKWARD(
    bilinear2d_aa,
    BilinearFunctor,
    float,
    true);
INSTANTIATE_UPSAMPLE_3D_BACKWARD(float, true);

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
